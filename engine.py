import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import librosa
import pandas as pd

from audio import (
    assign_outputs_to_refs_by_corr,
    loudness_normalize,
    make_union_voiced_mask,
)
from config import *
from distortions import apply_adv_distortions, apply_distortions
from metrics import (
    compute_pm,
    compute_ps,
    diffusion_map_torch,
    pm_ci_components_full,
    ps_ci_components_full,
)
from models import embed_batch, load_model
from utils import *


def run_experiment(
    models,
    mixtures,
    *,
    systems=None,
    algos=None,
    experiment_id=None,
    layer=DEFAULT_LAYER,
    add_ci=DEFAULT_ADD_CI,
    seed=42,
    on_missing="skip",
    verbose=False,
    max_gpus=None,
):

    # Initialize
    gpu_distributor = GPUWorkDistributor(max_gpus)
    ngpu = get_gpu_count(max_gpus)

    if on_missing not in {"skip", "error"}:
        raise ValueError("on_missing must be 'skip' or 'error'.")

    # Set seeds
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Canonicalize manifest
    canon_mix = canonicalize_mixtures(mixtures, systems=systems)

    # Build per-speaker entries
    mixture_entries = []
    for m in canon_mix:
        entries = []
        for i, refp in enumerate(m.refs):
            sid = m.speaker_ids[i]
            entries.append(
                {"id": sid, "ref": Path(refp), "mixture": m.mixture_id, "outs": {}}
            )
        mixture_entries.append(entries)

    # Assignment of system outputs
    for m, mix_entries in zip(canon_mix, mixture_entries):
        for algo, out_list in (m.systems or {}).items():
            mapping = assign_outputs_to_refs_by_corr(
                [e["ref"] for e in mix_entries], out_list
            )
            for idx, e in enumerate(mix_entries):
                j = mapping[idx]
                if j is not None:
                    e["outs"][algo] = out_list[j]

    # Algorithms to run
    if algos is None:
        algos_to_run = sorted(
            {algo for m in canon_mix for algo in (m.systems or {}).keys()}
        )
    else:
        algos_to_run = list(algos)

    # Experiment folder
    exp_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(RESULTS_ROOT, f"experiment_{exp_id}")
    os.makedirs(exp_root, exist_ok=True)

    # Save parameters
    params = {
        "models": models,
        "layer": layer,
        "add_ci": add_ci,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "ngpu": ngpu,
        "max_gpus": max_gpus,
    }

    with open(os.path.join(exp_root, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # Save canonical manifest
    canon_struct = [
        {
            "mixture_id": m.mixture_id,
            "references": [str(p) for p in m.refs],
            "systems": {
                a: [str(p) for p in outs] for a, outs in (m.systems or {}).items()
            },
            "speaker_ids": m.speaker_ids,
        }
        for m in canon_mix
    ]

    with open(os.path.join(exp_root, "manifest_canonical.json"), "w") as f:
        json.dump(canon_struct, f, indent=2)

    print(f"Starting experiment {exp_id} with {ngpu} GPUs")
    print(f"Results will be saved to: {exp_root}")

    clear_gpu_memory()
    get_gpu_memory_info(verbose)

    # Flatten and load refs
    flat_entries = [e for mix in mixture_entries for e in mix]
    all_refs = {}

    if verbose:
        print("Loading reference signals...")
    for e in flat_entries:
        wav, _ = librosa.load(str(e["ref"]), sr=SR)
        all_refs[e["id"]] = torch.from_numpy(loudness_normalize(wav)).pin_memory()

    # Per-mixture voiced masks
    if verbose:
        print("Computing voiced masks...")

    win = int(ENERGY_WIN_MS * SR / 1000)
    hop = int(ENERGY_HOP_MS * SR / 1000)
    voiced_mask_mix = []

    for i, mix in enumerate(mixture_entries):
        if verbose:
            print(f"  Computing mask for mixture {i+1}/{len(mixture_entries)}")

        if ngpu > 0:
            with torch.cuda.device(0):
                refs_for_mix = [all_refs[e["id"]].cuda() for e in mix]
                mask = make_union_voiced_mask(refs_for_mix, win, hop)
                voiced_mask_mix.append(mask.cpu().pin_memory())
                for ref in refs_for_mix:
                    del ref
                torch.cuda.empty_cache()
        else:
            refs_for_mix = [all_refs[e["id"]].cpu() for e in mix]
            mask = make_union_voiced_mask(refs_for_mix, win, hop)
            voiced_mask_mix.append(mask.cpu())

    ordered_speakers = [e["id"] for e in flat_entries]

    # Process algorithms
    for algo_idx, algo in enumerate(algos_to_run):
        if verbose:
            print(f"\nProcessing Algorithm {algo_idx+1}/{len(algos_to_run)}: {algo}")

        algo_dir = os.path.join(exp_root, algo)
        os.makedirs(algo_dir, exist_ok=True)

        # Load outputs
        all_outs = {}
        missing = []

        for mix_idx, mix in enumerate(mixture_entries):
            for e in mix:
                assigned_path = e.get("outs", {}).get(algo)
                if assigned_path is None:
                    missing.append((e["mixture"], e["id"]))
                    continue

                wav, _ = librosa.load(str(assigned_path), sr=SR)
                all_outs[e["id"]] = torch.from_numpy(
                    loudness_normalize(wav)
                ).pin_memory()

        if missing:
            msg = f"[{algo}] missing outputs for {len(missing)} speaker(s)"
            if on_missing == "error":
                raise FileNotFoundError(msg)
            else:
                if verbose:
                    warnings.warn(msg + " Skipping those speakers.")

        if not all_outs:
            if verbose:
                warnings.warn(f"[{algo}] No outputs provided. Skipping algorithm.")
            continue

        # Accumulators
        ps_ts = {m: {s: [] for s in ordered_speakers} for m in models}
        pm_ts = {m: {s: [] for s in ordered_speakers} for m in models}
        ps_bias_ts = {m: {s: [] for s in ordered_speakers} for m in models}
        ps_prob_ts = {m: {s: [] for s in ordered_speakers} for m in models}
        pm_bias_ts = {m: {s: [] for s in ordered_speakers} for m in models}
        pm_prob_ts = {m: {s: [] for s in ordered_speakers} for m in models}

        # Process models
        for model_idx, mname in enumerate(models):
            if verbose:
                print(f"  Processing Model {model_idx+1}/{len(models)}: {mname}")

            # Build separate batches for PS (normal distortions) and PM (advanced distortions)
            for metric_type in ["PS", "PM"]:
                clear_gpu_memory()
                model_wrapper, layer_eff = load_model(mname, layer, max_gpus)
                get_gpu_memory_info(verbose)

                embs_by_mix = {}
                labels_by_mix = {}

                # Process mixtures
                for k, mix in enumerate(mixture_entries):
                    speakers_this_mix = [e for e in mix if e["id"] in all_outs]
                    if not speakers_this_mix:
                        continue

                    if verbose:
                        print(
                            f"Processing mixture {k+1}/{len(mixture_entries)} for {metric_type}"
                        )

                    all_signals_mix = []
                    all_masks_mix = []
                    all_labels_mix = []

                    for e in speakers_this_mix:
                        s = e["id"]

                        # Build appropriate batch for PS or PM
                        if metric_type == "PS":
                            # PS uses normal distortions
                            dists = [
                                loudness_normalize(d)
                                for d in apply_distortions(all_refs[s].numpy(), "all")
                            ]
                        else:
                            # PM uses advanced distortions
                            dists = [
                                loudness_normalize(d)
                                for d in apply_adv_distortions(
                                    all_refs[s].numpy(), "all"
                                )
                            ]

                        sigs = [all_refs[s].numpy(), all_outs[s].numpy()] + dists
                        lbls = ["ref", "out"] + [f"d{i}" for i in range(len(dists))]

                        masks = [voiced_mask_mix[k]] * len(sigs)
                        all_signals_mix.extend(sigs)
                        all_masks_mix.extend(masks)
                        all_labels_mix.extend([f"{s}-{l}" for l in lbls])

                    try:
                        embeddings = embed_batch(
                            all_signals_mix,
                            all_masks_mix,
                            model_wrapper,
                            layer_eff,
                            use_mlm=False,
                        )
                        if embeddings.numel() > 0:
                            embs_by_mix[k] = embeddings
                            labels_by_mix[k] = all_labels_mix
                    except Exception as ex:
                        if verbose:
                            print(f"      ERROR processing mixture {k+1}: {ex}")
                        continue

                    del all_signals_mix, all_masks_mix, all_labels_mix
                    clear_gpu_memory()

                # Compute metrics
                if verbose:
                    print(f"    Computing {metric_type} scores for {mname}...")

                with ThreadPoolExecutor(
                    max_workers=min(4, ngpu * 2 if ngpu > 0 else 2)
                ) as executor:
                    for k in range(len(mixture_entries)):
                        if k not in embs_by_mix:
                            continue

                        E, L, D = embs_by_mix[k].shape

                        def process_frame(f):
                            try:
                                if add_ci:
                                    coords_d, coords_c, eigvals, k_sub_gauss = (
                                        gpu_distributor.execute_on_gpu(
                                            diffusion_map_torch,
                                            embs_by_mix[k][:, f, :].numpy(),
                                            labels_by_mix[k],
                                            alpha=1.0,
                                            eig_solver="full",
                                            return_eigs=True,
                                            return_complement=True,
                                            return_cval=add_ci,
                                        )
                                    )
                                else:
                                    coords_d = gpu_distributor.execute_on_gpu(
                                        diffusion_map_torch,
                                        embs_by_mix[k][:, f, :].numpy(),
                                        labels_by_mix[k],
                                        alpha=1.0,
                                        eig_solver="full",
                                        return_eigs=False,
                                        return_complement=False,
                                        return_cval=False,
                                    )
                                    coords_c = None
                                    eigvals = None
                                    k_sub_gauss = 1

                                if metric_type == "PS":
                                    score = compute_ps(
                                        coords_d, labels_by_mix[k], max_gpus
                                    )
                                    bias = prob = None
                                    if add_ci:
                                        bias, prob = ps_ci_components_full(
                                            coords_d,
                                            coords_c,
                                            eigvals,
                                            labels_by_mix[k],
                                            delta=DEFAULT_DELTA_CI,
                                        )
                                    return f, "PS", score, bias, prob
                                else:
                                    score = compute_pm(
                                        coords_d, labels_by_mix[k], "gamma", max_gpus
                                    )
                                    bias = prob = None
                                    if add_ci:
                                        bias, prob = pm_ci_components_full(
                                            coords_d,
                                            coords_c,
                                            eigvals,
                                            labels_by_mix[k],
                                            delta=DEFAULT_DELTA_CI,
                                            K=k_sub_gauss,
                                        )
                                    return f, "PM", score, bias, prob

                            except Exception as ex:
                                if verbose:
                                    print(f"        ERROR frame {f+1}: {ex}")
                                return None

                        futures = [executor.submit(process_frame, f) for f in range(L)]
                        for fut in futures:
                            result = fut.result()
                            if result is None:
                                continue

                            f, metric, score, bias, prob = result

                            if metric == "PS":
                                for sp in score:
                                    ps_ts[mname][sp].append(score[sp])
                                    if add_ci and bias is not None:
                                        ps_bias_ts[mname][sp].append(bias[sp])
                                        ps_prob_ts[mname][sp].append(prob[sp])
                            else:
                                for sp in score:
                                    pm_ts[mname][sp].append(score[sp])
                                    if add_ci and bias is not None:
                                        pm_bias_ts[mname][sp].append(bias[sp])
                                        pm_prob_ts[mname][sp].append(prob[sp])

                del model_wrapper
                clear_gpu_memory()

        # Save results
        if verbose:
            print(f"  Saving results for {algo}...")

        for m in models:

            def _pad(vec, n):
                return vec + [np.nan] * (n - len(vec))

            max_len = 0
            for s in ordered_speakers:
                max_len = max(max_len, len(ps_ts[m][s]), len(pm_ts[m][s]))

            pd.DataFrame(
                {s: _pad(ps_ts[m][s], max_len) for s in ordered_speakers}
            ).to_csv(os.path.join(algo_dir, f"ps_scores_{m}.csv"), index=False)

            pd.DataFrame(
                {s: _pad(pm_ts[m][s], max_len) for s in ordered_speakers}
            ).to_csv(os.path.join(algo_dir, f"pm_scores_{m}.csv"), index=False)

            if add_ci:
                ci_cols = {}
                for s in ordered_speakers:
                    ci_cols[f"{s}_ps_bias"] = _pad(ps_bias_ts[m][s], max_len)
                    ci_cols[f"{s}_ps_prob"] = _pad(ps_prob_ts[m][s], max_len)
                    ci_cols[f"{s}_pm_bias"] = _pad(pm_bias_ts[m][s], max_len)
                    ci_cols[f"{s}_pm_prob"] = _pad(pm_prob_ts[m][s], max_len)
                pd.DataFrame(ci_cols).to_csv(
                    os.path.join(algo_dir, f"ci_{m}.csv"), index=False
                )

        del all_outs
        clear_gpu_memory()

    print(f"\nEXPERIMENT COMPLETED")
    print(f"Results saved to: {exp_root}")

    del all_refs, voiced_mask_mix
    clear_gpu_memory()
    get_gpu_memory_info(verbose)

    return exp_root
