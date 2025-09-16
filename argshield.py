from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePath
import importlib.util

from config import DEFAULT_ALPHA
from models import get_model_config

MODEL_DEFAULT_LAYER = {
    "raw": None,
    "wavlm": 24,
    "wav2vec2": 24,
    "hubert": 24,
    "wavlm_base": 12,
    "wav2vec2_base": 12,
    "hubert_base": 12,
    "wav2vec2_xlsr": 24,
}

def _read_manifest_json(path: Path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Manifest must be JSON. Failed to parse: {e}")

def _read_manifest_py(path: Path):
    spec = importlib.util.spec_from_file_location("manifest_mod", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load Python manifest: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "MANIFEST"):
        raise SystemExit(f"Python manifest {path} must define a top-level variable MANIFEST")

    manifest = mod.MANIFEST

    def _to_str(p):
        if isinstance(p, (Path, PurePath)):
            return str(p)
        if isinstance(p, str):
            return p
        raise TypeError(f"Path entry must be str or Path, got {type(p)}: {p}")

    normalized = []
    try:
        for item in manifest:
            mix_id = item["mixture_id"]
            refs = [_to_str(x) for x in item["references"]]
            systems = {}
            for sys_name, lst in item["systems"].items():
                systems[sys_name] = [_to_str(x) for x in lst]
            normalized.append({
                "mixture_id": mix_id,
                "references": refs,
                "systems": systems,
            })
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f"Malformed MANIFEST in {path}: {e}")
    return normalized

def _read_manifest(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".py"}:
        return _read_manifest_py(path)
    elif suffix in {".json", ".txt"}:
        return _read_manifest_json(path)
    else:
        raise SystemExit(f"Unsupported manifest type '{suffix}'. Use .py, .json, or .txt")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run PS/PM experiment from a manifest file."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to manifest (.py with MANIFEST or .json/.txt with JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=("Embedding model. Choices: "
              "raw, wavlm, wav2vec2, hubert, wavlm_base, wav2vec2_base, "
              "hubert_base, wav2vec2_xlsr"),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer (validated per model). Omit to use the model default.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional diffusion-maps alpha in [0,1] (default: config DEFAULT_ALPHA).",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument("--max-gpus", type=int, default=None, help="Limit GPUs to use (must be >= 0).")
    return parser.parse_args()

def _validate_and_resolve(model: str, layer_opt: int|None, alpha_opt: float|None):
    allowed_models = set(get_model_config(0).keys())
    if model not in allowed_models:
        raise SystemExit(f"Unknown --model '{model}'. Allowed: {sorted(allowed_models)}")

    max_layer = MODEL_DEFAULT_LAYER.get(model)
    if model == "raw":
        layer_final = 0 if layer_opt is None else int(layer_opt)
    else:
        if layer_opt is None:
            if max_layer is None:
                raise SystemExit(f"--layer must be provided for model '{model}'.")
            layer_final = max_layer
        else:
            layer_final = int(layer_opt)
            if max_layer is not None and not (0 <= layer_final <= max_layer):
                raise SystemExit(
                    f"--layer {layer_final} is out of range for '{model}'. "
                    f"Expected 0..{max_layer} (or omit to use default {max_layer})."
                )

    alpha_final = DEFAULT_ALPHA if alpha_opt is None else float(alpha_opt)
    if not (0.0 <= alpha_final <= 1.0):
        raise SystemExit("--alpha must be in [0, 1].")
    return layer_final, alpha_final

def _validate_gpus(max_gpus_opt):
    if max_gpus_opt is None:
        return None
    try:
        mg = int(max_gpus_opt)
    except Exception:
        raise SystemExit("--max-gpus must be an integer >= 0.")
    if mg < 0:
        raise SystemExit("--max-gpus must be >= 0.")
    return mg