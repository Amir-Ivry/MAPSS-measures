from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .config import ExperimentConfig
from .experiment import run_experiment


def _parse_args(argv=None) -> ExperimentConfig:
    p = argparse.ArgumentParser(description="Run PM/PS diffusion‑map experiment.")
    p.add_argument("--config", type=str, help="YAML config file.", default=None)
    p.add_argument("--mixtures-file", type=str, help="JSON file with mixtures.", default=None)

    # Quick overrides for common fields
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--results-root", type=str, default=None)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--distortions", dest="distortion_keys", type=str, default=None)
    p.add_argument("--alpha-dm", dest="alpha_dm", type=float, default=None)
    p.add_argument("--dm-solver", type=str, default=None, choices=["full", "lobpcg"])
    p.add_argument("--pm-method", type=str, default=None, choices=["gamma", "rank"])
    p.add_argument("--use-mlm", action="store_true")
    p.add_argument("--mlm-oversample", action="store_true")
    p.add_argument("--advanced-distortions", action="store_true")
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--misalign-max-ms", type=int, default=None)
    p.add_argument("--add-ci", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    args = p.parse_args(argv)

    if args.config:
        with open(args.config, "r") as f:
            d = yaml.safe_load(f)
        cfg = ExperimentConfig(**d)
    else:
        cfg = ExperimentConfig()

    # Apply CLI overrides if provided
    for k in ["data_root", "results_root", "alpha_dm", "dm_solver", "pm_method", "layer", "misalign_max_ms", "seed"]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v)

    if args.models is not None:
        cfg.models = args.models
    if args.distortion_keys is not None:
        cfg.distortion_keys = args.distortion_keys
    if args.use_mlm:
        cfg.use_mlm = True
    if args.mlm_oversample:
        cfg.mlm_oversample = True
    if args.advanced_distortions:
        cfg.advanced_distortions = True
    if args.add_ci:
        cfg.add_ci = True

    if args.mixtures_file:
        cfg.mixtures_file = args.mixtures_file
    elif not cfg.mixtures_file:
        raise SystemExit("You must provide --mixtures-file or set mixtures_file in YAML.")

    # Load mixtures
    with open(cfg.mixtures_file, "r") as f:
        mixtures = json.load(f)["mixtures"]

    return cfg, mixtures


def main(argv=None) -> None:
    cfg, mixtures = _parse_args(argv)
    run_experiment(cfg, mixtures)


if __name__ == "__main__":
    main(sys.argv[1:])
