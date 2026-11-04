"""Isolated MERT music frontend using the released MAPSS scoring functions.

MERT operates at 24 kHz. Activity uses 480-sample windows and a 324-sample
hop; each activity frame selects the nearest available native MERT feature
center (400-sample convolution receptive field, 320-sample stride).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from numbers import Integral
from os import PathLike
from pathlib import Path
import random
import threading
from typing import Any

import numpy as np
import pandas as pd
import torch

from .api import MAPSSResult
from .audio import compute_speaker_activity_masks, loudness_normalize
from .config import DEFAULT_DELTA_CI
from .distortions import apply_pm_distortions, apply_ps_distortions
from .metrics import compute_pm, compute_ps, diffusion_map_torch
from ._mert_ci import pm_ci_components, ps_ci_components


MERT_ID = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_SR = 24_000
ACTIVITY_WINDOW = 480
ACTIVITY_HOP = 324
FEATURE_WINDOW = 400
FEATURE_HOP = 320
_RNG_LOCK = threading.RLock()


@dataclass(frozen=True)
class _MERTResult(MAPSSResult):
    """MAPSS result with MERT provenance and optional confidence plots."""

    metadata: dict[str, Any]

    def save(
        self,
        directory: str | PathLike[str],
        *,
        plot: bool = False,
        plot_filename: str | PathLike[str] = "mapss_over_time.png",
        plot_dpi: int = 180,
    ) -> Path:
        if not isinstance(plot, bool):
            raise TypeError("plot must be True or False.")
        relative_plot = None
        if plot:
            try:
                relative_plot = Path(plot_filename)
            except TypeError as exc:
                raise TypeError("plot_filename must be a string or path-like value.") from exc
            if relative_plot.is_absolute() or ".." in relative_plot.parts:
                raise ValueError("plot_filename must be a relative path inside the result directory.")
            if relative_plot.suffix.lower() not in {".png", ".pdf", ".svg"}:
                raise ValueError("plot_filename must use a .png, .pdf, or .svg extension.")
            if not isinstance(plot_dpi, Integral) or isinstance(plot_dpi, (bool, np.bool_)) or plot_dpi <= 0:
                raise ValueError("plot_dpi must be a positive integer.")
        destination = super().save(directory, plot=False)
        if self.ci is None:
            # A later score-only save must not leave stale confidence data that
            # the standalone plotter would silently associate with new scores.
            (destination / "confidence.csv").unlink(missing_ok=True)
        (destination / "mert_config.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if plot:
            from .plotting import plot_tables

            plot_tables(self.ps, self.pm, self.ci, destination / relative_plot,
                        dpi=int(plot_dpi), require_confidence=False)
        return destination


def _as_mert_mono_array(source, input_sample_rate: int, label: str) -> np.ndarray:
    """Apply the public API's audio guardrails with an explicit 24 kHz target."""
    if isinstance(source, (str, PathLike)):
        import soundfile as sf

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"{label} must be an audio file, not a directory: {path}")
        try:
            waveform, source_rate = sf.read(path, dtype="float32", always_2d=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(
                f"{label} could not be decoded as audio: {path}. "
                "Use a format supported by libsndfile, such as WAV or FLAC."
            ) from exc
    else:
        if hasattr(source, "detach") and hasattr(source, "cpu"):
            source = source.detach().cpu().numpy()
        try:
            waveform = np.asarray(source, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(f"{label} must contain numeric audio samples or be an audio path.") from exc
        source_rate = input_sample_rate

    if waveform.ndim == 2:
        if waveform.shape[0] == 0 or waveform.shape[1] == 0:
            raise ValueError(f"{label} cannot contain an empty channel or sample axis.")
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            raise ValueError(
                f"{label} appears to have shape (channels, samples). Individual "
                "multi-channel waveforms must use shape (samples, channels); transpose it."
            )
        waveform = waveform.mean(axis=1)
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError(
            f"{label} must be a non-empty one-dimensional waveform or have shape (samples, channels)."
        )
    if not np.isfinite(waveform).all():
        raise ValueError(f"{label} contains NaN or infinite samples.")
    if float(np.max(np.abs(waveform))) <= 1e-12:
        raise ValueError(f"{label} is entirely silent.")
    if int(source_rate) <= 0:
        raise ValueError(f"{label} has an invalid sample rate: {source_rate}.")
    if int(source_rate) != MERT_SR:
        import librosa

        try:
            # Integer duration arithmetic avoids a floating-point ceil adding a
            # spurious sample (e.g. 22050 samples at 44100 Hz -> 12001).
            converted_length = (len(waveform)*MERT_SR+int(source_rate)-1)//int(source_rate)
            waveform = librosa.resample(waveform, orig_sr=int(source_rate),
                                        target_sr=MERT_SR, res_type="soxr_hq", fix=False)
            waveform = librosa.util.fix_length(waveform, size=converted_length)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"{label} could not be resampled from {source_rate} Hz to {MERT_SR} Hz."
            ) from exc
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"{label} became empty or non-finite during audio conversion.")
    if float(np.max(np.abs(waveform))) <= 1e-12:
        raise ValueError(f"{label} is entirely silent after audio conversion.")
    return np.asarray(waveform, dtype=np.float32)


@contextmanager
def _scoped_random_state(seed: int, device: str):
    """Seed this call while restoring Python, NumPy, CPU and used-CUDA RNGs."""
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        cuda_devices = [0] if str(device).startswith("cuda") else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                np.random.seed(seed)
                random.seed(seed)
                # torch.manual_seed also touches CUDA generators; seed only those used.
                torch.random.default_generator.manual_seed(seed)
                if cuda_devices:
                    with torch.cuda.device(0):
                        torch.cuda.manual_seed(seed)
                yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _load_mert_model(device):
    """Load the pinned official checkpoint and custom code; no global cache."""
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        raise RuntimeError(
            "MERT requires PyTorch >=2.6 to load its official .bin checkpoint. "
            "Upgrade PyTorch before using model='mert'."
        )
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(MERT_ID, revision=MERT_REVISION)
    model = AutoModel.from_pretrained(
        MERT_ID, revision=MERT_REVISION, trust_remote_code=True,
        use_safetensors=False, torch_dtype=torch.float32,
        attn_implementation="eager", output_hidden_states=True,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, extractor


def _extract_features(waveforms, model, extractor, layer, device, *, batch_size=2, diagnostics=None):
    """Return one native ``(frames, features)`` float32 array per waveform.

    Equal-length waveforms can share a batch. A first full forward verifies that
    capturing the next encoder block's input exactly equals hidden_states[layer]
    before enabling an early stop; unavailable or unequal captures use full forwards.
    """
    if batch_size not in {1, 2}:
        raise ValueError("MERT feature batch_size must be 1 or 2.")
    report = diagnostics if diagnostics is not None else {}
    report.update(batch_size=batch_size, dtype="float32", prefix_validation="pending")
    result = [None] * len(waveforms)
    pending = list(range(len(waveforms)))
    prefix_checked, prefix_validated = False, False

    class _Captured(Exception):
        def __init__(self, hidden):
            self.hidden = hidden

    def capture_prefix(values):
        layers = getattr(getattr(model, "encoder", None), "layers", [])
        if not 0 <= layer < len(layers):
            raise RuntimeError("No following encoder block is available for this hidden-state index.")

        def capture(module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise RuntimeError("Encoder block did not receive positional hidden states.")
            raise _Captured(inputs[0].detach())

        hook = layers[layer].register_forward_pre_hook(capture)
        try:
            model(values, output_hidden_states=True)
        except _Captured as captured:
            return captured.hidden
        finally:
            hook.remove()
        raise RuntimeError("Encoder capture hook was not reached.")

    with torch.inference_mode():
        while pending:
            first = pending.pop(0)
            indices = [first]
            if batch_size == 2:
                same = next((i for i, index in enumerate(pending)
                             if len(waveforms[index]) == len(waveforms[first])), None)
                if same is not None:
                    indices.append(pending.pop(same))
            values = extractor([waveforms[i] for i in indices], sampling_rate=MERT_SR,
                               return_tensors="pt", padding=True).input_values.to(device=device, dtype=torch.float32)
            if prefix_validated:
                hidden = capture_prefix(values)
            else:
                full = model(values, output_hidden_states=True)
                hidden = full.hidden_states[layer].detach()
                if not prefix_checked:
                    try:
                        candidate = capture_prefix(values)
                        prefix_validated = torch.equal(hidden, candidate)
                        report.update(prefix_validation="exact" if prefix_validated else "unequal_full_forward",
                                      prefix_torch_equal=prefix_validated,
                                      prefix_max_abs_difference=float((hidden-candidate).abs().max()))
                    except (RuntimeError, AttributeError, IndexError, TypeError) as exc:
                        report.update(prefix_validation="unavailable_full_forward", prefix_reason=str(exc))
                    prefix_checked = True
                del full
            for offset, index in enumerate(indices):
                feature = hidden[offset].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
                if feature.ndim != 2 or not feature.size or not np.isfinite(feature).all():
                    raise RuntimeError("MERT returned empty or non-finite native features.")
                result[index] = feature
            del hidden, values
    return result


def _nearest_native_indices(activity_indices, native_frames):
    """Choose nearest available native convolution centers, including boundaries."""
    if native_frames < 1:
        raise RuntimeError("MERT returned no native feature frames.")
    proposed = np.rint((np.asarray(activity_indices)*ACTIVITY_HOP
                        + (ACTIVITY_WINDOW-FEATURE_WINDOW)/2) / FEATURE_HOP).astype(np.int64)
    selected = np.clip(proposed, 0, native_frames-1)
    return selected, int(np.count_nonzero(selected != proposed))


def _versions():
    values = {}
    for name in ["torch", "transformers", "numpy", "scipy", "librosa", "soundfile", "pyloudnorm"]:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = "unavailable"
    return values


def evaluate_mert(references, outputs, *, sample_rate, source_names, layer, alpha,
                  add_ci, seed, max_gpus, length_policy, verbose):
    """Evaluate ordered music sources using a scoped, standalone MERT frontend."""
    names = list(source_names)
    refs = [_as_mert_mono_array(item, sample_rate, f"reference[{i}]") for i, item in enumerate(references)]
    outs = [_as_mert_mono_array(item, sample_rate, f"output[{i}]") for i, item in enumerate(outputs)]
    lengths = [len(wave) for wave in refs+outs]
    if len(set(lengths)) != 1:
        if length_policy == "error":
            labels = [*(f"reference[{i}]" for i in range(len(refs))), *(f"output[{i}]" for i in range(len(outs)))]
            durations = ", ".join(f"{label}={length/MERT_SR:.6f}s ({length} samples)" for label, length in zip(labels, lengths))
            raise ValueError(
                f"All references and outputs must have equal duration after conversion to {MERT_SR} Hz. "
                f"Received: {durations}. Pass length_policy='trim' only when trimming every input to the shortest duration is intentional."
            )
        shortest = min(lengths)
        refs, outs = [wave[:shortest] for wave in refs], [wave[:shortest] for wave in outs]
    if len(refs[0]) < int(0.4*MERT_SR):
        raise ValueError("Audio must be at least 0.400 seconds long.")
    use_cuda = max_gpus != 0 and torch.cuda.is_available() and torch.cuda.device_count() > 0
    if max_gpus is not None and max_gpus > 0 and not use_cuda:
        raise RuntimeError("MERT cannot access a CUDA GPU. Set max_gpus=0 for CPU.")
    device = "cuda:0" if use_cuda else "cpu"
    internal_ids = [f"src{i:03d}" for i in range(len(refs))]
    model = extractor = native_features = None

    with _scoped_random_state(int(seed), device):
        try:
            refs = [loudness_normalize(wave, sr=MERT_SR) for wave in refs]
            outs = [loudness_normalize(wave, sr=MERT_SR) for wave in outs]
            multi, individual = compute_speaker_activity_masks(
                [torch.from_numpy(wave) for wave in refs], ACTIVITY_WINDOW, ACTIVITY_HOP)
            individual = torch.stack(individual).cpu().numpy()
            valid = torch.where(multi)[0].cpu().numpy()
            inactive = [name for name, mask in zip(names, individual) if not mask[valid].any()]
            if inactive:
                raise RuntimeError(
                    f"MAPSS produced no valid frames for {', '.join(inactive)}. Ensure every source participates in at "
                    "least one frame where two or more references are simultaneously active."
                )

            # Complete PS banks precede complete PM banks, before model loading.
            waveforms, waveform_keys, key_to_index, banks, bank_counts = [], [], {}, {}, {}

            def register(wave):
                wave = np.ascontiguousarray(wave, dtype=np.float32)
                key = hashlib.sha256(b"float32-mono-sr24000\0"+wave.tobytes()).hexdigest()
                if key not in key_to_index:
                    key_to_index[key] = len(waveforms)
                    waveforms.append(wave)
                    waveform_keys.append(key)
                return key_to_index[key]

            for metric, make_bank in [("PS", apply_ps_distortions), ("PM", apply_pm_distortions)]:
                banks[metric], bank_counts[metric] = {}, {}
                for source, reference, estimate in zip(internal_ids, refs, outs):
                    distortions = [loudness_normalize(wave, sr=MERT_SR)
                                   for wave in make_bank(reference, "all", sr=MERT_SR)]
                    if len(distortions) < 2:
                        raise RuntimeError(f"{metric} distortion bank for {source} contains fewer than two distortions.")
                    labels = ["ref", "out"]+[f"d{i}" for i in range(len(distortions))]
                    banks[metric][source] = [(f"{source}-{label}", register(wave))
                                            for label, wave in zip(labels, [reference, estimate]+distortions)]
                    bank_counts[metric][source] = len(distortions)
            if verbose:
                print(f"MERT: extracting {len(waveforms)} unique waveforms at 24000 Hz on {device}.", flush=True)
            model, extractor = _load_mert_model(device)
            extraction = {}
            native_features = _extract_features(waveforms, model, extractor, layer, device,
                                                batch_size=2, diagnostics=extraction)
            if len(native_features) != len(waveforms):
                raise RuntimeError("MERT returned an unexpected number of waveform embeddings.")
            selected, max_center_error, clipped_mappings, native_counts = [], 0.0, 0, set()
            width = None
            for wave, feature in zip(waveforms, native_features):
                feature = np.asarray(feature)
                expected = (len(wave)-FEATURE_WINDOW)//FEATURE_HOP+1
                if feature.ndim != 2 or feature.shape[0] != expected or not feature.shape[1] or not np.isfinite(feature).all():
                    raise RuntimeError("MERT returned invalid native feature shape or non-finite values.")
                if width is not None and feature.shape[1] != width:
                    raise RuntimeError("MERT returned inconsistent feature widths across waveforms.")
                width = feature.shape[1]
                indices, clipped = _nearest_native_indices(valid, len(feature))
                selected.append(indices)
                clipped_mappings += clipped
                native_counts.add(len(feature))
                error = np.abs(indices*FEATURE_HOP+FEATURE_WINDOW/2-(valid*ACTIVITY_HOP+ACTIVITY_WINDOW/2))/MERT_SR
                if error.size:
                    max_center_error = max(max_center_error, float(error.max()))
            waveforms.clear()

            tables, score_counts, dimensions = {}, {}, {}
            timestamps = np.arange(len(multi), dtype=float)*ACTIVITY_HOP/MERT_SR*1000
            confidence = pd.DataFrame({"timestamp_ms": timestamps}) if add_ci else None
            ci_diagnostics = {"PS": {}, "PM": {}}
            for metric in ["PS", "PM"]:
                values = np.full((len(multi), len(names)), np.nan)
                biases = np.full_like(values, np.nan) if add_ci else None
                probabilities = np.full_like(values, np.nan) if add_ci else None
                dimensions[metric] = []
                for ordinal, frame in enumerate(valid):
                    active = [i for i in range(len(names)) if individual[i, frame]]
                    entries = [entry for i in active for entry in banks[metric][internal_ids[i]]]
                    labels = [label for label, _ in entries]
                    cloud = np.stack([native_features[index][selected[index][ordinal]] for _, index in entries])
                    try:
                        embedding = diffusion_map_torch(
                            cloud, labels, device=device, alpha=float(alpha),
                            cutoff=.99, diffusion_time=1, eig_solver="full",
                            return_complement=add_ci, return_eigs=add_ci,
                        )
                        if add_ci:
                            coords, complement, eigenvalues = embedding
                        else:
                            coords = embedding
                        scores = (compute_ps(coords, labels, max_gpus=1 if use_cuda else 0) if metric == "PS" else
                                  compute_pm(coords, labels, "gamma", max_gpus=1 if use_cuda else 0))
                        if add_ci:
                            ci_function = ps_ci_components if metric == "PS" else pm_ci_components
                            bias, probability = ci_function(
                                coords, complement, eigenvalues, labels,
                                delta=DEFAULT_DELTA_CI, device=device, scores=scores,
                                diagnostics=ci_diagnostics[metric],
                            )
                        for index in active:
                            value = float(scores[internal_ids[index]])
                            if not np.isfinite(value) or not -1e-6 <= value <= 1+1e-6:
                                raise ValueError(f"invalid {metric} score for {names[index]}: {value}")
                            values[frame, index] = value
                            if add_ci:
                                radius = float(bias[internal_ids[index]])
                                tail = float(probability[internal_ids[index]])
                                if not (np.isfinite(radius) and np.isfinite(tail)
                                        and 0 <= radius <= 1 and 0 <= tail <= 1):
                                    raise ValueError(f"invalid {metric} confidence components for {names[index]}")
                                biases[frame, index], probabilities[frame, index] = radius, tail
                        dimensions[metric].append(int(coords.shape[1]))
                    except Exception as exc:
                        raise RuntimeError(f"MERT {metric} failed at activity frame {frame}: {exc}") from exc
                table = pd.DataFrame(values, columns=names)
                table.insert(0, "timestamp_ms", timestamps)
                tables[metric] = table
                if add_ci:
                    for index, name in enumerate(names):
                        confidence[f"{name}_{metric.lower()}_bias"] = biases[:, index]
                        confidence[f"{name}_{metric.lower()}_prob"] = probabilities[:, index]
                score_counts[metric] = {name: int(np.isfinite(values[:,i]).sum()) for i, name in enumerate(names)}
                if any(count == 0 for count in score_counts[metric].values()):
                    raise RuntimeError(f"MERT {metric} produced a source without finite scores.")
                if verbose:
                    print(f"MERT: computed {metric} over {len(valid)} eligible frames.", flush=True)
            metadata = {
                "model": MERT_ID, "revision": MERT_REVISION, "layer": int(layer), "hidden_states_index": int(layer),
                "sample_rate": MERT_SR, "seed": int(seed), "device": device, "dtype": "float32",
                "attention": "eager", "max_gpus_requested": None if max_gpus is None else int(max_gpus),
                "gpu_policy": "at most one GPU, cuda:0; max_gpus=0 forces CPU; no multi-GPU execution",
                "source_names": dict(zip(internal_ids, names)), "normalization": "released -23 LUFS normalization with peak limiting at 24000 Hz",
                "length_policy": length_policy, "input_samples_at_24000_before_trim": lengths, "evaluated_samples": len(refs[0]),
                "timing": {"activity_window_samples": ACTIVITY_WINDOW, "activity_hop_samples": ACTIVITY_HOP,
                           "native_receptive_field_samples": FEATURE_WINDOW, "native_stride_samples": FEATURE_HOP,
                           "timestamp": "activity-window start, in milliseconds", "alignment": "nearest available native feature center; rounded index clipped only at feature boundaries",
                           "mapping": "clip(rint((i*324+(480-400)/2)/320),0,native_frames-1)",
                           "historical_change": "corrects the original interpolated-mask compaction and ordinal pairing",
                           "max_center_error_seconds": max_center_error, "boundary_clipped_waveform_frame_pairs": clipped_mappings,
                           "activity_frames": len(multi), "eligible_frames": len(valid), "native_frame_counts": sorted(native_counts)},
                "bank_counts": bank_counts, "unique_waveforms": len(waveform_keys), "normalized_waveform_sha256": waveform_keys,
                "banks": "complete released standard PS bank, then complete released adaptive PM bank, each in source order",
                "graph": "joint active-source graph separately for each metric/frame, containing references, estimates and all metric-specific distortions",
                "scoring": {"alpha": float(alpha), "diffusion_time": 1, "eigensolver": "full", "cutoff": .99, "pm_method": "gamma",
                            "kernel": "exp(-D_squared/(2*median_off_diagonal_D_squared))",
                            "coordinates": "released symmetric-normalized-kernel eigenvectors scaled by eigenvalues^t",
                            "covariance": "released unconditional 1e-6 ridge; PM reference-centered cross-products",
                            "retained_dimension_range": {metric: [min(ds), max(ds)] for metric, ds in dimensions.items()}},
                "score_counts": score_counts, "extraction": extraction, "versions": _versions(),
                "confidence_intervals": {
                    "supported": True, "computed": bool(add_ci), "delta": DEFAULT_DELTA_CI,
                    "construction": "paper-derived plug-in deterministic truncation and finite-sample probability radii",
                    "pm_moments": "reference-centered cross-products, matching the unchanged PM score",
                    "device": device, "diagnostics": ci_diagnostics if add_ci else {},
                    "coverage": "nominal 1-delta under the paper's assumptions; empirical coverage not established by implementation tests",
                },
                "limitations": ["Uses released numerical conventions; not an exact reproduction of the original paper experiments.",
                                "Released adaptive PM processing skips reverb/echo variants whose lengths exceed its 20 ms frames.",
                                "CPU and CUDA use float32; original experimental extraction used GPU float16."],
            }
            if add_ci:
                columns = [f"{name}_{metric}_{component}" for name in names
                           for metric in ("ps", "pm") for component in ("bias", "prob")]
                confidence = confidence[["timestamp_ms"] + columns]
            return _MERTResult(ps=tables["PS"], pm=tables["PM"], ci=confidence, model="mert", layer=int(layer),
                               sample_rate=MERT_SR, metadata=metadata)
        finally:
            # No model or embedding cache survives this evaluation.
            del model, extractor, native_features
