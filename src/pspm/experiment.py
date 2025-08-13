from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch

from .audio import loudness_normalize, make_union_voiced_mask
from .config import ExperimentConfig
from .diffusion import diffusion_map_torch
from .logging_utils import setup_logger
from .metrics import compute_pm, compute_ps, pm_ci_components_full, ps_ci_components_full
from .models import load_model

logger = setup_logger(__name__)


ALGOS_DEFAULT = [
    "Algo10_SAOC", "Algo10_SAOC_SAOC", "Algo10_SASSEC", "Algo14_SASSEC", "Algo1_SASSEC",
    "Algo21_SiSEC08", "Algo22_SiSEC08", "Algo24_SiSEC08", "Algo25_SiSEC08", "Algo27_SiSEC08",
    "Algo2_SAOC", "Algo2_SAOC_SAOC", "Algo2_SASSEC", "Algo30_SiSEC08", "Algo32_SiSEC08",
    "Algo34_SiSEC08", "Algo3_SAOC", "Algo3_SAOC_SAOC", "Algo3_SASSEC", "Algo4_SAOC",
    "Algo4_SAOC_SAOC", "Algo4_SASSEC", "Algo6_SAOC", "Algo6_SAOC_SAOC", "Algo6_SASSEC",
    "Algo7_SASSEC", "Algo8_SASSEC", "Algo8_SiSEC08", "Algo9_SAOC", "Algo9_SAOC_SAOC",
    "Algo9_SASSEC", "SAOC_SASSEC"
]


def _random_misalign(sig: np.ndarray, sr: int, max_ms: int) -> np.ndarray:
    """Deterministic max shift used for robustness tests (0 disables)."""
    max_samples = int(sr * max_ms / 1000)
    if max_samples <= 0:
        return sig
    shift = max_samples
    if shift == 0:
        return sig
    z = np.zeros(abs(shift), dtype=sig.dtype)
    return np.concatenate([z, sig[:-shift]]) if shift > 0 else np.concatenate([sig[-shift:], z])


def run_experiment(cfg: ExperimentConfig, mixtures: List[List[Dict]]) -> None:
    """Execute the full PM/PS pipeline with diffusion maps.

    This implements the same logic as the original `run_experiment` research
    script, with clearer structure, logging, and robust CPU/GPU fallbacks.
    """
    # Seeding
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    exp_root = os.path.join(cfg.results_root, f"experiment_{cfg.seed}")
    os.makedirs(exp_root, exist_ok=True)

    # Persist params
    with open(os.path.join(exp_root, "params.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    logger.info("Starting experiment — results → %s", exp_root)

    # Load references
    logger.info("Loading reference signals…")
    all_refs: Dict[str, np.ndarray] = {}
    for mix in mixtures:
        for e in mix:
            if e["id"] in all_refs:
                continue
            wav, _ = librosa.load(e["ref"], sr=16_000)
            all_refs[e["id"]] = loudness_normalize(wav)

    # Precompute voiced masks
    logger.info("Computing voiced masks…")
    win = int(cfg.energy_win_ms * 16_000 / 1000)
    hop = int(cfg.energy_hop_ms * 16_000 / 1000)
    voiced_mask_mix = []
    for k, mix in enumerate(mixtures):
        refs_for_mix = [all_refs[e["id"]] for e in mix]
        mask = make_union_voiced_mask(refs_for_mix, win, hop, cfg.silence_ratio)
        voiced_mask_mix.append(mask)

    ordered_speakers = [e["id"] for mix in mixtures for e in mix]
    logger.info("Speakers: %s", ordered_speakers)

    algos = ALGOS_DEFAULT

    for algo in algos:
        logger.info("Processing algorithm: %s", algo)
        algo_dir = os.path.join(exp_root, algo)
        os.makedirs(algo_dir, exist_ok=True)

        # Load outputs
        logger.info("Loading algorithm outputs…")
        all_outs: Dict[str, np.ndarray] = {}
        for mix in mixtures:
            for e in mix:
                wav_path = Path(cfg.data_root) / algo / f"{e['id']}.wav"
                wav, _ = librosa.load(str(wav_path), sr=16_000)
                if cfg.misalign_max_ms:
                    wav = _random_misalign(wav, 16_000, cfg.misalign_max_ms)
                all_outs[e["id"]] = loudness_normalize(wav)

        # Results accumulators
        ps_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        pm_gamma_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        pm_rank_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        ps_bias_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        ps_prob_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        pm_bias_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}
        pm_prob_ts = {m: {s: [] for s in ordered_speakers} for m in cfg.models}

        # Models
        for mname in cfg.models:
            logger.info("Embedding model: %s", mname)
            model_wrapper, layer_eff = load_model(mname, cfg.layer)

            embs_by_mix, labels_by_mix = {}, {}
            for k, mix in enumerate(mixtures):
                # Build inputs for this mixture
                signals, masks, labels = [], [], []
                for e in mix:
                    s = e["id"]
                    # ref/out + distortions
                    ref_sig = all_refs[s]
                    out_sig = all_outs[s]
                    if cfg.advanced_distortions:
                        from .distortions import apply_adv_distortions
                        dists = [loudness_normalize(d, 16_000) for d in apply_adv_distortions(ref_sig, cfg.distortion_keys, 16_000, cfg.energy_win_ms)]
                    else:
                        from .distortions import apply_distortions
                        dists = [loudness_normalize(d, 16_000) for d in apply_distortions(ref_sig, cfg.distortion_keys, 16_000)]
                    sigs = [ref_sig, out_sig] + dists
                    lab = [f"{s}-ref", f"{s}-out"] + [f"{s}-d{i}" for i in range(len(dists))]
                    signals.extend(sigs)
                    labels.extend(lab)
                    masks.extend([voiced_mask_mix[k]] * len(sigs))

                # Embed
                embeddings = model_wrapper.process_batch(signals, masks, use_mlm=cfg.use_mlm)
                if embeddings.numel() == 0:
                    logger.warning("No embeddings produced for mixture %d", k + 1)
                    continue
                embs_by_mix[k] = embeddings
                labels_by_mix[k] = labels

            # Compute PS/PM per frame
            for k in range(len(mixtures)):
                if k not in embs_by_mix:
                    continue
                E, L, D = embs_by_mix[k].shape

                def process_frame(f: int):
                    coords_d, coords_c, eigvals = diffusion_map_torch(
                        embs_by_mix[k][:, f, :].numpy(),
                        labels_by_mix[k],
                        alpha=cfg.alpha_dm,
                        eig_solver=cfg.dm_solver,
                        return_eigs=True,
                        return_complement=True,
                        return_cval=False,
                    )

                    ps = compute_ps(coords_d, labels_by_mix[k])
                    pm_gamma = compute_pm(coords_d, labels_by_mix[k], "gamma")
                    pm_rank = compute_pm(coords_d, labels_by_mix[k], "rank")

                    ps_bias, ps_prob, pm_bias, pm_prob = None, None, None, None
                    if cfg.add_ci:
                        ps_bias, ps_prob = ps_ci_components_full(coords_d, coords_c, eigvals, labels_by_mix[k], delta=cfg.delta_ci)
                        pm_bias, pm_prob = pm_ci_components_full(coords_d, coords_c, eigvals, labels_by_mix[k], delta=cfg.delta_ci)
                    return f, ps, pm_gamma, pm_rank, ps_bias, ps_prob, pm_bias, pm_prob

                with ThreadPoolExecutor(max_workers=min(4, max(1, torch.cuda.device_count() * 2))) as ex:
                    futures = [ex.submit(process_frame, f) for f in range(L)]
                    for fut in futures:
                        f, ps, pm_gamma, pm_rank, ps_bias, ps_prob, pm_bias, pm_prob = fut.result()

                        for sp in ps:
                            ps_ts[mname][sp].append(ps[sp])
                            if cfg.add_ci:
                                ps_bias_ts[mname][sp].append(ps_bias[sp])
                                ps_prob_ts[mname][sp].append(ps_prob[sp])
                        for sp in pm_gamma:
                            pm_gamma_ts[mname][sp].append(pm_gamma[sp])
                            if cfg.add_ci:
                                pm_bias_ts[mname][sp].append(pm_bias[sp])
                                pm_prob_ts[mname][sp].append(pm_prob[sp])
                        for sp in pm_rank:
                            pm_rank_ts[mname][sp].append(pm_rank[sp])

            # Save results per model
            max_len = 0
            for s in ordered_speakers:
                max_len = max(
                    max_len, len(ps_ts[mname][s]), len(pm_gamma_ts[mname][s]), len(pm_rank_ts[mname][s])
                )

            def _pad(arr): return arr + [np.nan] * (max_len - len(arr))

            ps_data = {s: _pad(ps_ts[mname][s]) for s in ordered_speakers}
            pd.DataFrame(ps_data).to_csv(os.path.join(algo_dir, f"ps_scores_{mname}.csv"), index=False)

            pm_gamma_data = {s: _pad(pm_gamma_ts[mname][s]) for s in ordered_speakers}
            pd.DataFrame(pm_gamma_data).to_csv(os.path.join(algo_dir, f"pm_scores_gamma_{mname}.csv"), index=False)

            pm_rank_data = {s: _pad(pm_rank_ts[mname][s]) for s in ordered_speakers}
            pd.DataFrame(pm_rank_data).to_csv(os.path.join(algo_dir, f"pm_scores_rank_{mname}.csv"), index=False)

            pm_def = pm_gamma_ts if cfg.pm_method == "gamma" else pm_rank_ts
            pm_def_data = {s: _pad(pm_def[mname][s]) for s in ordered_speakers}
            pd.DataFrame(pm_def_data).to_csv(os.path.join(algo_dir, f"pm_scores_{mname}.csv"), index=False)

            if cfg.add_ci:
                ci_cols = {}
                for s in ordered_speakers:
                    ci_cols[f"{s}_ps_bias"] = _pad(ps_bias_ts[mname][s])
                    ci_cols[f"{s}_ps_prob"] = _pad(ps_prob_ts[mname][s])
                    ci_cols[f"{s}_pm_bias"] = _pad(pm_bias_ts[mname][s])
                    ci_cols[f"{s}_pm_prob"] = _pad(pm_prob_ts[mname][s])
                pd.DataFrame(ci_cols).to_csv(os.path.join(algo_dir, f"ci_{mname}.csv"), index=False)

            logger.info("Saved CSVs for model %s", mname)

    logger.info("Experiment complete. See %s", exp_root)
