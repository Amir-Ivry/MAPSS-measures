"""MERT-only adapter for the established manifest command/output layout."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


_CI_SUFFIXES = ("ps_bias", "ps_prob", "pm_bias", "pm_prob")


def _directory_name(value, label):
    name = str(value)
    if not name.strip() or name in {".", ".."} or any(c in name for c in ("/", "\\", "\0")):
        raise ValueError(f"{label} must be a non-empty directory name without path separators.")
    return name


def compute_mert_manifest(
    mixtures,
    *,
    systems=None,
    algos=None,
    experiment_id=None,
    layer,
    add_ci,
    alpha,
    seed,
    on_missing,
    verbose,
    max_gpus,
    results_root,
):
    """Evaluate each music mixture/system through the public MERT API.

    Each pair has its own seeded distortion banks. Existing non-MERT manifest
    execution and its random-number sequence never enter this adapter.
    """
    from .api import _resolve_options, mapss
    from .utils import canonicalize_mixtures

    _resolve_options("mert", layer, alpha, max_gpus)
    if not isinstance(add_ci, (bool, np.bool_)):
        raise TypeError("add_ci must be True or False.")
    if on_missing not in {"skip", "error"}:
        raise ValueError("on_missing must be 'skip' or 'error'.")
    canonical = canonicalize_mixtures(mixtures, systems=systems)
    if not canonical:
        raise ValueError("mixtures must contain at least one mixture.")
    names = [_directory_name(m.mixture_id, "mixture_id") for m in canonical]
    if len(set(names)) != len(names):
        raise ValueError("MERT manifest mixture_id values must be unique.")
    for mixture in canonical:
        if len(mixture.refs) < 2:
            raise ValueError(f"Mixture {mixture.mixture_id}: MAPSS requires at least two references.")
    selected_algos = (
        sorted({algo for m in canonical for algo in m.systems}) if algos is None else list(algos)
    )
    if not selected_algos:
        raise ValueError("No output systems were provided.")

    exp_id = _directory_name(
        experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f"), "experiment_id"
    )
    root = Path(results_root) / f"experiment_{exp_id}"
    root.mkdir(parents=True, exist_ok=True)
    params = {
        "models": ["mert"], "layer": int(layer), "alpha": float(alpha),
        "add_ci": bool(add_ci), "seed": int(seed), "max_gpus": max_gpus,
        "sample_rate": 24000, "rng_scope": "independent seed per mixture/system",
    }
    (root / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
    manifest = [
        {"mixture_id": m.mixture_id, "references": [str(p) for p in m.refs],
         "systems": {a: [str(p) for p in paths] for a, paths in m.systems.items()},
         "speaker_ids": m.speaker_ids}
        for m in canonical
    ]
    (root / "manifest_canonical.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    evaluated = 0
    for mixture in canonical:
        ps, pm, ci, metadata = None, None, None, {}
        for algo in selected_algos:
            if algo not in mixture.systems:
                continue
            estimates = mixture.systems[algo]
            problem = None
            if len(estimates) != len(mixture.refs):
                problem = (
                    f"[{algo}] Number of outputs ({len(estimates)}) does not match "
                    f"number of references ({len(mixture.refs)}) for mixture {mixture.mixture_id}"
                )
            else:
                missing = [str(p) for p in list(mixture.refs) + list(estimates) if not p.is_file()]
                if missing:
                    problem = f"[{algo}] Missing audio in mixture {mixture.mixture_id}: {missing}"
            if problem is not None:
                if on_missing == "error":
                    raise ValueError(problem)
                if verbose:
                    warnings.warn(problem + " Skipping this system.", stacklevel=2)
                continue

            result = mapss(
                mixture.refs, estimates, source_names=mixture.speaker_ids,
                model="mert", layer=layer, alpha=alpha, add_ci=bool(add_ci), seed=seed,
                max_gpus=max_gpus, verbose=verbose,
            )
            timestamps = result.ps["timestamp_ms"].to_numpy()
            if not np.array_equal(result.pm["timestamp_ms"].to_numpy(), timestamps):
                raise RuntimeError("MERT manifest PS and PM returned different frame timestamps.")
            if add_ci:
                if result.ci is None:
                    raise RuntimeError("MERT manifest evaluation did not return requested confidence data.")
                if not np.array_equal(result.ci["timestamp_ms"].to_numpy(), timestamps):
                    raise RuntimeError("MERT manifest confidence data returned different frame timestamps.")
            if ps is None:
                ps = pd.DataFrame({"timestamp_ms": timestamps})
                pm = pd.DataFrame({"timestamp_ms": timestamps})
                if add_ci:
                    ci = pd.DataFrame({"timestamp_ms": timestamps})
            elif not np.array_equal(ps["timestamp_ms"].to_numpy(), timestamps):
                raise RuntimeError("MERT manifest systems returned different frame timestamps.")
            for source in mixture.speaker_ids:
                column = f"{algo}_{source}"
                if column in ps:
                    raise ValueError(f"MERT manifest produced a duplicate result column: {column}")
                ps[column] = result.ps[source].to_numpy()
                pm[column] = result.pm[source].to_numpy()
                if add_ci:
                    for suffix in _CI_SUFFIXES:
                        ci_column = f"{source}_{suffix}"
                        if ci_column not in result.ci:
                            raise RuntimeError(
                                f"MERT manifest confidence data is missing column: {ci_column}"
                            )
                        ci[f"{column}_{suffix}"] = result.ci[ci_column].to_numpy()
            metadata[algo] = result.metadata
            evaluated += 1
        if ps is not None:
            destination = root / mixture.mixture_id
            destination.mkdir(parents=True, exist_ok=True)
            ps.to_csv(destination / "ps_scores_mert.csv", index=False)
            pm.to_csv(destination / "pm_scores_mert.csv", index=False)
            ci_path = destination / "ci_mert.csv"
            if ci is not None:
                ci.to_csv(ci_path, index=False)
            else:
                # A score-only rerun must not leave confidence from an older run.
                ci_path.unlink(missing_ok=True)
            (destination / "mert_config.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    if evaluated == 0:
        raise ValueError("No valid MERT mixture/system pairs were available to evaluate.")
    return str(root)
