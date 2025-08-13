"""
A convenience script that mirrors the original grid in main_exp_reproduce.py.
Edit the lists below to sweep different combinations.

Usage:
    python scripts/run_grid.py --mixtures configs/mixtures.example.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from pspm.config import ExperimentConfig
from pspm.experiment import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixtures", required=True, help="Path to mixtures JSON (same format as configs/mixtures.example.json)")
    args = ap.parse_args()

    with open(args.mixtures, "r") as f:
        MIXTURES = json.load(f)["mixtures"]

    # Define your grid here (mirrors the original defaults)
    model_list = ["wav2vec2"]
    distortion_sets = ["all"]
    alpha_vals = [1.0]
    dm_solvers = ["full"]
    pm_methods = ["gamma"]
    use_mlm_vals = [False]
    mlm_oversample_vals = [False]
    advanced_distortions_vals = [False]
    layers = [2]
    misalign_max_ms_vals = [0]
    add_ci_vals = [True]

    for (model_name, distortion_keys, alpha_dm, dm_solver, pm_method, use_mlm, mlm_oversample, layer, adv_distortions, misalign_max_ms, add_ci) in itertools.product(
        model_list, distortion_sets, alpha_vals, dm_solvers, pm_methods, use_mlm_vals, mlm_oversample_vals, layers, advanced_distortions_vals, misalign_max_ms_vals, add_ci_vals
    ):
        cfg = ExperimentConfig(
            models=[model_name],
            distortion_keys=distortion_keys,
            alpha_dm=alpha_dm,
            dm_solver=dm_solver,
            pm_method=pm_method,
            use_mlm=use_mlm,
            mlm_oversample=mlm_oversample,
            advanced_distortions=adv_distortions,
            layer=layer,
            misalign_max_ms=misalign_max_ms,
            add_ci=add_ci,
            seed=42,
        )
        cfg.mixtures_file = args.mixtures
        run_experiment(cfg, MIXTURES)


if __name__ == "__main__":
    main()
