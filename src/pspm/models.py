from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    WavLMModel,
    Wav2Vec2Model,
)

from .logging_utils import setup_logger

logger = setup_logger(__name__)


def get_device() -> torch.device:
    """Return a best-effort device (cuda if available, else cpu)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _model_registry(layer: int) -> dict[str, Tuple[str | None, Any | None, int | None]]:
    return {
        "raw": (None, None, None),
        "wavlm": ("microsoft/wavlm-large", WavLMModel, layer),
        "wav2vec2": ("facebook/wav2vec2-large-lv60", Wav2Vec2Model, layer),
        "hubert": ("facebook/hubert-large-ll60k", HubertModel, layer),
        "wav2vec2_xlsr": ("facebook/wav2vec2-large-xlsr-53", Wav2Vec2Model, layer),
    }


class ModelRunner:
    """Unified CPU/single-GPU/multi-GPU runner with a *single* consistent API.

    This intentionally avoids experimental stream juggling and focuses on
    readability, stability, and clear resource ownership.
    """

    def __init__(self, model_name: str, layer: int):
        self.layer = layer
        self.model_name = model_name
        ckpt, cls, _ = _model_registry(layer)[model_name]

        if model_name == "raw":
            self.extractor = None
            self.model = None
            self.devices = [get_device()]
            return

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)

        # Load the base model
        use_cuda = torch.cuda.is_available()
        dtype = torch.float16 if use_cuda else torch.float32
        base_model = cls.from_pretrained(
            ckpt,
            output_hidden_states=True,
            use_safetensors=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()

        if use_cuda and torch.cuda.device_count() > 1:
            # DataParallel for simplicity and robustness
            self.model = nn.DataParallel(base_model).to(get_device())
            self.devices = list(self.model.device_ids) if hasattr(self.model, "device_ids") else [0]
        else:
            self.model = base_model.to(get_device())
            self.devices = [get_device()]

        logger.info("Loaded %s on %s", model_name, "CUDA" if use_cuda else "CPU")

    def __repr__(self) -> str:  # pragma: no cover - utility
        return f"ModelRunner(name={self.model_name}, layer={self.layer}, devices={self.devices})"

    def process_batch(self, signals: List, masks: List, use_mlm: bool = False) -> torch.Tensor:
        """Return embeddings shaped (B, L, D) for a batch of signals.

        Parameters
        ----------
        signals : list of np.ndarray
            Waveforms (mono) as float arrays.
        masks : list of torch.Tensor[bool]
            Booleans per-frame (voiced) masks; will be interpolated to model timebase.
        use_mlm : bool
            If True, temporarily switch the model to train() to enable masked-LM stochasticity.

        Returns
        -------
        torch.Tensor
            Embedding tensor of shape (B, L, D) (padded on L to the maximum frames in batch).
        """
        if self.model_name == "raw":
            return _embed_batch_raw(signals, masks)

        # Feature extraction on CPU for stability; tensors moved later
        inputs = self.extractor(signals, sampling_rate=16_000, return_tensors="pt", padding=True)
        device = get_device()
        input_values = inputs.input_values.to(device, non_blocking=torch.cuda.is_available())

        # Mode switch
        orig_mode = self.model.training
        self.model.train() if use_mlm else self.model.eval()

        with torch.no_grad():
            autocast = torch.cuda.amp.autocast if device.type == "cuda" else _nullcontext
            with autocast():
                out = self.model(input_values, output_hidden_states=True)
                hs = out.hidden_states[self.layer]  # (B, T, D)

        # restore mode
        self.model.train(orig_mode)

        B, T, D = hs.shape
        keep_list = []
        for b in range(B):
            mask_b = masks[b].float().unsqueeze(0).unsqueeze(0).to(device)
            mask_t = F.interpolate(mask_b, size=T, mode="nearest")[0, 0].bool()
            keep_list.append(hs[b][mask_t])

        if keep_list:
            L_max = max(x.shape[0] for x in keep_list)
            keep_padded = [F.pad(x, (0, 0, 0, L_max - x.shape[0])) for x in keep_list]
            result = torch.stack(keep_padded, dim=0).cpu()
        else:
            result = torch.empty(0, 0, 0)

        return result


class _nullcontext:
    """Fallback context manager (no-op) when CUDA autocast is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _embed_batch_raw(signals, masks) -> torch.Tensor:
    """Represent waveforms as framed tensors mimicking (L, win) embeddings."""
    # Parameters mirror the research defaults
    SR = 16_000
    ENERGY_WIN_MS = 20
    ENERGY_HOP_MS = 20
    win = int(ENERGY_WIN_MS * SR / 1000)
    hop = int(ENERGY_HOP_MS * SR / 1000)

    reps = []
    L_max = 0
    for sig_np, mask_np in zip(signals, masks):
        x = torch.as_tensor(sig_np[:-1], dtype=torch.float32)
        frames = x.unfold(0, win, hop)
        mask = torch.as_tensor(mask_np[: len(frames)], dtype=torch.bool)
        keep = frames[mask] if mask.any() else frames[:1]
        reps.append(keep)
        L_max = max(L_max, keep.size(0))
    reps = [F.pad(r, (0, 0, 0, L_max - r.size(0))) for r in reps]
    return torch.stack(reps, dim=0)


@lru_cache(maxsize=4)
def load_model(model_name: str, layer: int) -> Tuple[ModelRunner, int]:
    """Cache and return a ready-to-run model wrapper plus the effective layer index."""
    runner = ModelRunner(model_name, layer)
    return runner, layer
