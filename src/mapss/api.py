"""Practitioner-facing Python API for MAPSS."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import MODEL_DEFAULT_LAYER, MODEL_MAX_LAYER, SR

Waveform = str | PathLike[str] | np.ndarray | Sequence[float] | Any


@dataclass(frozen=True)
class MAPSSResult:
    """Frame-level MAPSS scores and an utterance-level convenience summary.

    ``ps`` and ``pm`` contain a ``timestamp_ms`` column followed by one column
    per source. Inactive frames are represented by ``NaN``. The ``summary``
    property averages each measure over its finite, active frames.
    """

    ps: pd.DataFrame
    pm: pd.DataFrame
    ci: pd.DataFrame | None
    model: str
    layer: int
    sample_rate: int

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(c for c in self.ps.columns if c != "timestamp_ms")

    @property
    def summary(self) -> pd.DataFrame:
        rows = []
        for source in self.source_names:
            ps_values = pd.to_numeric(self.ps[source], errors="coerce")
            pm_values = pd.to_numeric(self.pm[source], errors="coerce")
            rows.append(
                {
                    "source": source,
                    "ps": float(ps_values.mean(skipna=True)),
                    "pm": float(pm_values.mean(skipna=True)),
                    "ps_frames": int(ps_values.notna().sum()),
                    "pm_frames": int(pm_values.notna().sum()),
                }
            )
        return pd.DataFrame(rows).set_index("source")

    def save(
        self,
        directory: str | PathLike[str],
        *,
        plot: bool = False,
        plot_filename: str | PathLike[str] = "mapss_over_time.png",
        plot_dpi: int = 180,
    ) -> Path:
        """Save this evaluation's tables and optionally its paper-style figure.

        ``plot=True`` writes a six-panel, time-aligned view of PM, PS, their
        deterministic error radii, and their probabilistic 95% bounds. It
        therefore requires a result computed with ``add_ci=True`` and the
        optional plotting dependencies.
        """
        if not isinstance(plot, bool):
            raise TypeError("plot must be True or False.")
        if plot and self.ci is None:
            raise ValueError(
                "plot=True requires confidence data. Run mapss(..., add_ci=True), "
                "then save the result with plotting enabled."
            )
        relative_plot = None
        if plot:
            try:
                relative_plot = Path(plot_filename)
            except TypeError as exc:
                raise TypeError("plot_filename must be a string or path-like value.") from exc
            if relative_plot.is_absolute() or ".." in relative_plot.parts:
                raise ValueError(
                    "plot_filename must be a relative path inside the result directory."
                )
            if relative_plot.suffix.lower() not in {".png", ".pdf", ".svg"}:
                raise ValueError(
                    "plot_filename must use a .png, .pdf, or .svg extension."
                )
            if (
                not isinstance(plot_dpi, Integral)
                or isinstance(plot_dpi, (bool, np.bool_))
                or plot_dpi <= 0
            ):
                raise ValueError("plot_dpi must be a positive integer.")

        try:
            destination = Path(directory)
        except TypeError as exc:
            raise TypeError("directory must be a string or path-like value.") from exc
        if destination.exists() and not destination.is_dir():
            raise ValueError(
                f"MAPSS results destination must be a directory: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        self.ps.to_csv(destination / "ps_scores.csv", index=False)
        self.pm.to_csv(destination / "pm_scores.csv", index=False)
        self.summary.to_csv(destination / "summary.csv")
        if self.ci is not None:
            self.ci.to_csv(destination / "confidence.csv", index=False)
        if plot:
            from .plotting import plot_tables

            plot_tables(
                self.ps,
                self.pm,
                self.ci,
                destination / relative_plot,
                dpi=plot_dpi,
                require_confidence=True,
            )
        return destination


def _is_scalar_sequence(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    return all(np.isscalar(item) and not isinstance(item, (str, bytes)) for item in value)


def _source_collection(value: Any, argument: str) -> list[Waveform]:
    if isinstance(value, (str, PathLike)):
        return [value]

    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()

    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [value]
        if value.ndim == 2:
            return [value[index] for index in range(value.shape[0])]
        raise ValueError(f"{argument} must be one- or two-dimensional.")

    if _is_scalar_sequence(value):
        return [np.asarray(value)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            raise ValueError(f"{argument} cannot be empty.")
        return list(value)
    raise TypeError(
        f"{argument} must be a waveform, path, or sequence of source waveforms."
    )


def _as_mono_array(source: Waveform, input_sample_rate: int, label: str) -> np.ndarray:
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
            raise TypeError(
                f"{label} must contain numeric audio samples or be an audio path."
            ) from exc
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
            f"{label} must be a non-empty one-dimensional waveform or have "
            "shape (samples, channels)."
        )
    if not np.isfinite(waveform).all():
        raise ValueError(f"{label} contains NaN or infinite samples.")
    if float(np.max(np.abs(waveform))) <= 1e-12:
        raise ValueError(f"{label} is entirely silent.")
    if int(source_rate) <= 0:
        raise ValueError(f"{label} has an invalid sample rate: {source_rate}.")

    if int(source_rate) != SR:
        import librosa

        try:
            waveform = librosa.resample(
                waveform, orig_sr=int(source_rate), target_sr=SR, res_type="soxr_hq"
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"{label} could not be resampled from {source_rate} Hz to {SR} Hz."
            ) from exc
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"{label} became empty or non-finite during audio conversion.")
    if float(np.max(np.abs(waveform))) <= 1e-12:
        raise ValueError(f"{label} is entirely silent after audio conversion.")
    return np.asarray(waveform, dtype=np.float32)


def _resolve_options(model: str, layer: int | None, alpha: float, max_gpus: int | None):
    model = str(model).lower()
    if model not in MODEL_MAX_LAYER:
        raise ValueError(
            f"Unknown model {model!r}. Choose one of {sorted(MODEL_MAX_LAYER)}."
        )
    if layer is not None and (
        not isinstance(layer, Integral) or isinstance(layer, (bool, np.bool_))
    ):
        raise TypeError("layer must be an integer or None.")
    resolved_layer = MODEL_DEFAULT_LAYER[model] if layer is None else int(layer)
    if not 0 <= resolved_layer <= MODEL_MAX_LAYER[model]:
        raise ValueError(
            f"layer must be in [0, {MODEL_MAX_LAYER[model]}] for {model!r}."
        )
    if not isinstance(alpha, Real) or isinstance(alpha, (bool, np.bool_)):
        raise TypeError("alpha must be a real number in [0, 1].")
    if not np.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1].")
    if max_gpus is not None and (
        not isinstance(max_gpus, Integral)
        or isinstance(max_gpus, (bool, np.bool_))
        or max_gpus < 0
    ):
        raise ValueError("max_gpus must be a non-negative integer or None.")
    return model, resolved_layer


def _rename_score_columns(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    score_columns = [column for column in frame.columns if column != "timestamp_ms"]
    if len(score_columns) != len(names):
        raise RuntimeError(
            "MAPSS returned an unexpected number of source columns: "
            f"expected {len(names)}, received {len(score_columns)}."
        )
    return frame.rename(columns=dict(zip(score_columns, names)))


def mapss(
    reference: Waveform | Sequence[Waveform],
    output: Waveform | Sequence[Waveform],
    *,
    sample_rate: int = SR,
    source_names: Sequence[str] | None = None,
    model: str = "wav2vec2",
    layer: int | None = None,
    alpha: float = 1.0,
    add_ci: bool = True,
    seed: int = 42,
    max_gpus: int | None = None,
    length_policy: str = "error",
    verbose: bool = False,
) -> MAPSSResult:
    """Compute Perceptual Separation (PS) and Perceptual Match (PM).

    Parameters
    ----------
    reference, output:
        Ordered source waveforms. Each may be a sequence of paths/arrays or a
        two-dimensional array shaped ``(sources, samples)``. Output source ``i``
        must correspond to reference source ``i``.
    sample_rate:
        Sample rate for in-memory arrays. File inputs carry their own sample rate.
        Inputs are resampled to the paper's 16 kHz operating rate.
    model, layer, alpha, add_ci, seed, max_gpus:
        MAPSS settings. The default is the paper-selected wav2vec 2.0 Large,
        transformer layer 2, with diffusion-map ``alpha=1``.
    length_policy:
        ``"error"`` rejects unequal durations. ``"trim"`` trims every source
        and output to the shortest duration.

    Returns
    -------
    MAPSSResult
        Frame-level scores in ``[0, 1]``; higher is better.
    """
    if (
        not isinstance(sample_rate, Integral)
        or isinstance(sample_rate, (bool, np.bool_))
        or sample_rate <= 0
    ):
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(add_ci, bool):
        raise TypeError("add_ci must be True or False.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False.")
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed must be an integer.")
    if length_policy not in {"error", "trim"}:
        raise ValueError("length_policy must be 'error' or 'trim'.")
    model, resolved_layer = _resolve_options(model, layer, alpha, max_gpus)
    if max_gpus is not None and max_gpus > 0:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError(
                f"max_gpus={max_gpus} requested CUDA execution, but PyTorch cannot "
                "access a CUDA GPU. Set max_gpus=0 for CPU, or install a compatible "
                "CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
            )

    references = _source_collection(reference, "reference")
    outputs = _source_collection(output, "output")
    if len(references) < 2:
        raise ValueError(
            f"MAPSS requires at least two ordered source pairs; received "
            f"{len(references)} reference source(s) and {len(outputs)} output source(s)."
        )
    if len(references) != len(outputs):
        raise ValueError(
            "reference and output must contain the same number of ordered sources; "
            f"received {len(references)} reference source(s) and {len(outputs)} "
            "output source(s). Output[i] must estimate reference[i]."
        )

    if source_names is None:
        # Preserve the established zero-based default names for compatibility.
        names = [f"source_{index}" for index in range(len(references))]
    else:
        if isinstance(source_names, (str, bytes)):
            raise TypeError(
                "source_names must be a sequence with one name per source, not one string."
            )
        names = [str(name).strip() for name in source_names]
        if len(names) != len(references):
            raise ValueError(
                f"source_names must contain {len(references)} names; received {len(names)}."
            )
        if any(not name for name in names):
            raise ValueError("source_names cannot contain an empty or whitespace-only name.")
        if len(set(names)) != len(names):
            raise ValueError("source_names must be unique after surrounding spaces are removed.")
        if "timestamp_ms" in names:
            raise ValueError("source_names cannot use the reserved name 'timestamp_ms'.")

    ref_arrays = [
        _as_mono_array(item, sample_rate, f"reference[{index}]")
        for index, item in enumerate(references)
    ]
    out_arrays = [
        _as_mono_array(item, sample_rate, f"output[{index}]")
        for index, item in enumerate(outputs)
    ]
    lengths = [len(item) for item in ref_arrays + out_arrays]
    if len(set(lengths)) != 1:
        if length_policy == "error":
            labels = [
                *(f"reference[{index}]" for index in range(len(ref_arrays))),
                *(f"output[{index}]" for index in range(len(out_arrays))),
            ]
            durations = ", ".join(
                f"{item_label}={length / SR:.6f}s ({length} samples)"
                for item_label, length in zip(labels, lengths)
            )
            raise ValueError(
                "All references and outputs must have equal duration after conversion "
                f"to {SR} Hz. Received: {durations}. Pass length_policy='trim' only "
                "when trimming every input to the shortest duration is intentional."
            )
        shortest = min(lengths)
        ref_arrays = [item[:shortest] for item in ref_arrays]
        out_arrays = [item[:shortest] for item in out_arrays]

    # EBU R128 integrated loudness uses a 400 ms analysis block.
    minimum_samples = int(0.4 * SR)
    if len(ref_arrays[0]) < minimum_samples:
        raise ValueError(
            f"Audio must be at least {minimum_samples / SR:.3f} seconds long."
        )

    import soundfile as sf

    from .engine import compute_mapss_measures

    with TemporaryDirectory(prefix="mapss-") as temporary:
        temporary_path = Path(temporary)
        reference_paths = []
        output_paths = []
        for index, (ref_array, out_array) in enumerate(zip(ref_arrays, out_arrays)):
            reference_path = temporary_path / f"reference_{index:02d}.wav"
            output_path = temporary_path / f"output_{index:02d}.wav"
            sf.write(reference_path, ref_array, SR, subtype="FLOAT")
            sf.write(output_path, out_array, SR, subtype="FLOAT")
            reference_paths.append(reference_path)
            output_paths.append(output_path)

        manifest = [
            {
                "mixture_id": "api",
                "references": reference_paths,
                "systems": {"system": output_paths},
            }
        ]
        experiment_path = Path(
            compute_mapss_measures(
                models=[model],
                mixtures=manifest,
                experiment_id="api",
                layer=resolved_layer,
                add_ci=bool(add_ci),
                alpha=float(alpha),
                seed=int(seed),
                on_missing="error",
                verbose=bool(verbose),
                max_gpus=max_gpus,
                results_root=temporary_path / "results",
            )
        )
        mixture_path = experiment_path / "api"
        ps = _rename_score_columns(
            pd.read_csv(mixture_path / f"ps_scores_{model}.csv"), names
        )
        pm = _rename_score_columns(
            pd.read_csv(mixture_path / f"pm_scores_{model}.csv"), names
        )
        ci_path = mixture_path / f"ci_{model}.csv"
        ci = pd.read_csv(ci_path) if add_ci and ci_path.is_file() else None
        if ci is not None:
            raw_prefixes = [f"system_api__reference_{index:02d}" for index in range(len(names))]
            ci = ci.rename(
                columns={
                    column: column.replace(raw_prefix, name, 1)
                    for column in ci.columns
                    for raw_prefix, name in zip(raw_prefixes, names)
                    if column.startswith(raw_prefix)
                }
            )

    missing_frames = []
    for measure, table in (("PS", ps), ("PM", pm)):
        for source in names:
            values = pd.to_numeric(table[source], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).any():
                missing_frames.append(f"{measure}[{source}]")
    if missing_frames:
        raise RuntimeError(
            "MAPSS produced no valid frames for "
            f"{', '.join(missing_frames)}. Ensure every source participates in at "
            "least one frame where two or more references are simultaneously active."
        )
    numeric = pd.concat(
        [ps.drop(columns="timestamp_ms"), pm.drop(columns="timestamp_ms")], axis=1
    ).to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if np.any((finite < -1e-6) | (finite > 1.0 + 1e-6)):
        raise RuntimeError("MAPSS produced a score outside the documented [0, 1] range.")

    return MAPSSResult(
        ps=ps,
        pm=pm,
        ci=ci,
        model=model,
        layer=resolved_layer,
        sample_rate=SR,
    )


evaluate = mapss
