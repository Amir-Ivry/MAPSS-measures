import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import librosa
import pandas as pd
from audio import (
    loudness_normalize,
    compute_speaker_activity_masks,
)
from config import *
from distortions import apply_pm_distortions, apply_ps_distortions
from metrics import (
    compute_pm,
    compute_ps,
    diffusion_map_torch,
    pm_ci_components_full,
    ps_ci_components_full,
)
from models import embed_batch, load_model
from utils import *


def compute_mapss_measures(
        models,
        mixtures,
        *,
        systems=None,
        algos=None,
        experiment_id=None,
        layer=DEFAULT_LAYER,
        add_ci=DEFAULT_ADD_CI,
        alpha=DEFAULT_ALPHA,
        seed=42,
        on_missing="skip",
        verbose=False,
        max_gpus=None,
):
    """
    Compute MAPSS measures (PM, PS, and their errors). Data is saved to csv files.

    :param models: backbone self-supervised models.
    :param mixtures: data to process from _read_manifest
    :param systems: specific systems (algos and data)
    :param algos: specific algorithms to use
    :param experiment_id: user-specified name for experiment
    :param layer: transformer layer of model to consider
    :param add_ci: True will compute error radius and tail bounds. False will not.
    :param alpha: normalization factor of the diffusion maps. Lives in [0, 1].
    :param seed: random seed number.
    :param on_missing: "skip" when missing values or throw an "error".
    :param verbose: True will print process info to console during runtime. False will minimize it.
    :param max_gpus: maximal amount of GPUs the program tries to utilize in parallel.

    """
    gpu_distributor = GPUWorkDistributor(max_gpus)
    ngpu = get_gpu_count(max_gpus)

    if on_missing not in {"skip", "error"}:
        raise ValueError("on_missing must be 'skip' or 'error'.")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    canon_mix = canonicalize_mixtures(mixtures, systems=systems)

    mixture_entries = []
    for m in canon_mix:
        entries = []
        for i, refp in enumerate(m.refs):
            sid = m.speaker_ids[i]
            entries.append(
                {"id": sid, "ref": Path(refp), "mixture": m.mixture_id, "outs": {}}
            )
        mixture_entries.append(entries)

    for m, mix_entries in zip(canon_mix, mixture_entries):
        for algo, out_list in (m.systems or {}).items():
            if len(out_list) != len(mix_entries):
                msg = f"[{algo}] Number of outputs ({len(out_list)}) does not match number of references ({len(mix_entries)}) for mixture {m.mixture_id}"
                if on_missing == "error":
                    raise ValueError(msg)
                else:
                    if verbose:
                        warnings.warn(msg + " Skipping this algorithm.")
                    continue

            for idx, e in enumerate(mix_entries):
                e["outs"][algo] = out_list[idx]

    if algos is None:
        algos_to_run = sorted(
            {algo for algo in canon_mix[0].systems.keys()} if canon_mix and canon_mix[0].systems else []
        )
    else:
        algos_to_run = list(algos)

    exp_id = experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(RESULTS_ROOT, f"experiment_{exp_id}")
    os.makedirs(exp_root, exist_ok=True)

    params = {
        "models": models,
        "layer": layer,
        "add_ci": add_ci,
        "alpha": alpha,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "ngpu": ngpu,
        "max_gpus": max_gpus,
    }

    with open(os.path.join(exp_root, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

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
    print("NOTE: Output files must be provided in the same order as reference files.")

    clear_gpu_memory()
    get_gpu_memory_info(verbose)

    flat_entries = [e for mix in mixture_entries for e in mix]
    all_refs = {}

    if verbose:
        print("Loading reference signals...")
    for e in flat_entries:
        wav, _ = librosa.load(str(e["ref"]), sr=SR)
        all_refs[e["id"]] = torch.from_numpy(loudness_normalize(wav))

    if verbose:
        print("Computing speaker activity masks...")

    win = int(ENERGY_WIN_MS * SR / 1000)
    hop = int(ENERGY_HOP_MS * SR / 1000)
    multi_speaker_masks_mix = []
    individual_speaker_masks_mix = []
    total_frames_per_mix = []

    for i, mix in enumerate(mixture_entries):
        if verbose:
            print(f"  Computing masks for mixture {i + 1}/{len(mixture_entries)}")

        if ngpu > 0:
            with torch.cuda.device(0):
                refs_for_mix = [all_refs[e["id"]].cuda() for e in mix]
                multi_mask, individual_masks = compute_speaker_activity_masks(refs_for_mix, win, hop)
                multi_speaker_masks_mix.append(multi_mask.cpu())
                individual_speaker_masks_mix.append([m.cpu() for m in individual_masks])
                total_frames_per_mix.append(multi_mask.shape[0])
                for ref in refs_for_mix:
                    del ref
                torch.cuda.empty_cache()
        else:
            refs_for_mix = [all_refs[e["id"]].cpu() for e in mix]
            multi_mask, individual_masks = compute_speaker_activity_masks(refs_for_mix, win, hop)
            multi_speaker_masks_mix.append(multi_mask.cpu())
            individual_speaker_masks_mix.append([m.cpu() for m in individual_masks])
            total_frames_per_mix.append(multi_mask.shape[0])

    ordered_speakers = [e["id"] for e in flat_entries]
    all_mixture_results = {}
    for mix_idx, (mix_canon, mix_entries) in enumerate(zip(canon_mix, mixture_entries)):
        mixture_id = mix_canon.mixture_id
        all_mixture_results[mixture_id] = {}
        total_frames = total_frames_per_mix[mix_idx]
        mixture_speakers = [e["id"] for e in mix_entries]

        for algo_idx, algo in enumerate(algos_to_run):
            if verbose:
                print(f"\nProcessing Mixture {mixture_id}, Algorithm {algo_idx + 1}/{len(algos_to_run)}: {algo}")
            all_outs = {}
            missing = []
            for e in mix_entries:
                assigned_path = e.get("outs", {}).get(algo)
                if assigned_path is None:
                    missing.append((e["mixture"], e["id"]))
                    continue
                wav, _ = librosa.load(str(assigned_path), sr=SR)
                all_outs[e["id"]] = torch.from_numpy(loudness_normalize(wav))

            if missing:
                msg = f"[{algo}] missing outputs for {len(missing)} speaker(s) in mixture {mixture_id}"
                if on_missing == "error":
                    raise FileNotFoundError(msg)
                else:
                    if verbose:
                        warnings.warn(msg + " Skipping those speakers.")

            if not all_outs:
                if verbose:
                    warnings.warn(f"[{algo}] No outputs for mixture {mixture_id}. Skipping.")
                continue

            if algo not in all_mixture_results[mixture_id]:
                all_mixture_results[mixture_id][algo] = {}

            ps_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}
            pm_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}
            ps_bias_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}
            ps_prob_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}
            pm_bias_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}
            pm_prob_frames = {m: {s: [np.nan] * total_frames for s in mixture_speakers} for m in models}

            for model_idx, mname in enumerate(models):
                if verbose:
                    print(f"  Processing Model {model_idx + 1}/{len(models)}: {mname}")

                for metric_type in ["PS", "PM"]:
                    clear_gpu_memory()
                    gc.collect()

                    model_wrapper, layer_eff = load_model(mname, layer, max_gpus)
                    get_gpu_memory_info(verbose)

                    speakers_this_mix = [e for e in mix_entries if e["id"] in all_outs]
                    if not speakers_this_mix:
                        continue

                    if verbose:
                        print(f"    Processing {metric_type} for mixture {mixture_id}")

                    multi_speaker_mask = multi_speaker_masks_mix[mix_idx]
                    individual_masks = individual_speaker_masks_mix[mix_idx]
                    valid_frame_indices = torch.where(multi_speaker_mask)[0].tolist()

                    speaker_signals = {}
                    speaker_labels = {}

                    for speaker_idx, e in enumerate(speakers_this_mix):
                        s = e["id"]

                        if metric_type == "PS":
                            dists = [
                                loudness_normalize(d)
                                for d in apply_ps_distortions(all_refs[s].numpy(), "all")
                            ]
                        else:
                            dists = [
                                loudness_normalize(d)
                                for d in apply_pm_distortions(
                                    all_refs[s].numpy(), "all"
                                )
                            ]

                        sigs = [all_refs[s].numpy(), all_outs[s].numpy()] + dists
                        lbls = ["ref", "out"] + [f"d{i}" for i in range(len(dists))]

                        speaker_signals[s] = sigs
                        speaker_labels[s] = [f"{s}-{l}" for l in lbls]

                    all_embeddings = {}
                    for s in speaker_signals:
                        sigs = speaker_signals[s]
                        masks = [multi_speaker_mask] * len(sigs)

                        batch_size = min(2, BATCH_SIZE)
                        embeddings_list = []

                        for i in range(0, len(sigs), batch_size):
                            batch_sigs = sigs[i:i + batch_size]
                            batch_masks = masks[i:i + batch_size]

                            batch_embs = embed_batch(
                                batch_sigs,
                                batch_masks,
                                model_wrapper,
                                layer_eff,
                                use_mlm=False,
                            )

                            if batch_embs.numel() > 0:
                                embeddings_list.append(batch_embs.cpu())

                            torch.cuda.empty_cache()

                        if embeddings_list:
                            all_embeddings[s] = torch.cat(embeddings_list, dim=0)
                        else:
                            all_embeddings[s] = torch.empty(0, 0, 0)

                    if not all_embeddings or all(e.numel() == 0 for e in all_embeddings.values()):
                        if verbose:
                            print(f"WARNING: mixture {mixture_id} produced 0 frames after masking; skipping.")
                        continue

                    L = next(iter(all_embeddings.values())).shape[1] if all_embeddings else 0

                    if L == 0:
                        if verbose:
                            print(f"WARNING: mixture {mixture_id} produced 0 frames after masking; skipping.")
                        continue

                    if verbose:
                        print(f"Computing {metric_type} scores for {mname}...")

                    with ThreadPoolExecutor(
                            max_workers=min(2, ngpu if ngpu > 0 else 1)
                    ) as executor:

                        def process_frame(f, frame_idx, all_embeddings_dict, speaker_labels_dict, individual_masks_list,
                                          speaker_indices):
                            try:
                                active_speakers = []
                                for spk_idx, spk_id in enumerate(speaker_indices):
                                    if individual_masks_list[spk_idx][frame_idx]:
                                        active_speakers.append(spk_id)

                                if len(active_speakers) < 2:
                                    return frame_idx, metric_type, {}, None, None

                                frame_embeddings = []
                                frame_labels = []
                                for spk_id in active_speakers:
                                    spk_embs = all_embeddings_dict[spk_id][:, f, :]
                                    frame_embeddings.append(spk_embs)
                                    frame_labels.extend(speaker_labels_dict[spk_id])

                                frame_emb = torch.cat(frame_embeddings, dim=0).detach().cpu().numpy()

                                if add_ci:
                                    coords_d, coords_c, eigvals, k_sub_gauss = (
                                        gpu_distributor.execute_on_gpu(
                                            diffusion_map_torch,
                                            frame_emb,
                                            frame_labels,
                                            alpha=alpha,
                                            eig_solver="full",
                                            return_eigs=True,
                                            return_complement=True,
                                            return_cval=add_ci,
                                        )
                                    )
                                else:
                                    coords_d = gpu_distributor.execute_on_gpu(
                                        diffusion_map_torch,
                                        frame_emb,
                                        frame_labels,
                                        alpha=alpha,
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
                                        coords_d, frame_labels, max_gpus
                                    )
                                    bias = prob = None
                                    if add_ci:
                                        bias, prob = ps_ci_components_full(
                                            coords_d,
                                            coords_c,
                                            eigvals,
                                            frame_labels,
                                            delta=DEFAULT_DELTA_CI,
                                        )
                                    return frame_idx, "PS", score, bias, prob
                                else:
                                    score = compute_pm(
                                        coords_d, frame_labels, "gamma", max_gpus
                                    )
                                    bias = prob = None
                                    if add_ci:
                                        bias, prob = pm_ci_components_full(
                                            coords_d,
                                            coords_c,
                                            eigvals,
                                            frame_labels,
                                            delta=DEFAULT_DELTA_CI,
                                            K=k_sub_gauss,
                                        )
                                    return frame_idx, "PM", score, bias, prob

                            except Exception as ex:
                                if verbose:
                                    print(f"ERROR frame {frame_idx}: {ex}")
                                return None

                        speaker_ids = [e["id"] for e in speakers_this_mix]

                        futures = [
                            executor.submit(
                                process_frame,
                                f,
                                valid_frame_indices[f],
                                all_embeddings,
                                speaker_labels,
                                individual_masks,
                                speaker_ids
                            )
                            for f in range(L)
                        ]

                        for fut in futures:
                            result = fut.result()
                            if result is None:
                                continue

                            frame_idx, metric, score, bias, prob = result

                            if metric == "PS":
                                for sp in mixture_speakers:
                                    if sp in score:
                                        ps_frames[mname][sp][frame_idx] = score[sp]
                                        if add_ci and bias is not None and sp in bias:
                                            ps_bias_frames[mname][sp][frame_idx] = bias[sp]
                                            ps_prob_frames[mname][sp][frame_idx] = prob[sp]
                            else:
                                for sp in mixture_speakers:
                                    if sp in score:
                                        pm_frames[mname][sp][frame_idx] = score[sp]
                                        if add_ci and bias is not None and sp in bias:
                                            pm_bias_frames[mname][sp][frame_idx] = bias[sp]
                                            pm_prob_frames[mname][sp][frame_idx] = prob[sp]

                    clear_gpu_memory()
                    gc.collect()

                    del model_wrapper
                    clear_gpu_memory()
                    gc.collect()

            all_mixture_results[mixture_id][algo][mname] = {
                'ps_frames': ps_frames[mname],
                'pm_frames': pm_frames[mname],
                'ps_bias_frames': ps_bias_frames[mname] if add_ci else None,
                'ps_prob_frames': ps_prob_frames[mname] if add_ci else None,
                'pm_bias_frames': pm_bias_frames[mname] if add_ci else None,
                'pm_prob_frames': pm_prob_frames[mname] if add_ci else None,
                'total_frames': total_frames
            }

        if verbose:
            print(f"Saving results for mixture {mixture_id}...")

        timestamps_ms = [i * hop * 1000 / SR for i in range(total_frames)]

        for model in models:
            ps_data = {'timestamp_ms': timestamps_ms}
            pm_data = {'timestamp_ms': timestamps_ms}
            ci_data = {'timestamp_ms': timestamps_ms} if add_ci else None

            for algo in algos_to_run:
                if algo not in all_mixture_results[mixture_id]:
                    continue
                if model not in all_mixture_results[mixture_id][algo]:
                    continue

                model_data = all_mixture_results[mixture_id][algo][model]

                for speaker in mixture_speakers:
                    col_name = f"{algo}_{speaker}"
                    ps_data[col_name] = model_data['ps_frames'][speaker]
                    pm_data[col_name] = model_data['pm_frames'][speaker]

                    if add_ci and ci_data is not None:
                        ci_data[f"{algo}_{speaker}_ps_bias"] = model_data['ps_bias_frames'][speaker]
                        ci_data[f"{algo}_{speaker}_ps_prob"] = model_data['ps_prob_frames'][speaker]
                        ci_data[f"{algo}_{speaker}_pm_bias"] = model_data['pm_bias_frames'][speaker]
                        ci_data[f"{algo}_{speaker}_pm_prob"] = model_data['pm_prob_frames'][speaker]

            mixture_dir = os.path.join(exp_root, mixture_id)
            os.makedirs(mixture_dir, exist_ok=True)

            pd.DataFrame(ps_data).to_csv(
                os.path.join(mixture_dir, f"ps_scores_{model}.csv"),
                index=False
            )

            pd.DataFrame(pm_data).to_csv(
                os.path.join(mixture_dir, f"pm_scores_{model}.csv"),
                index=False
            )

            if add_ci and ci_data is not None:
                pd.DataFrame(ci_data).to_csv(
                    os.path.join(mixture_dir, f"ci_{model}.csv"),
                    index=False
                )

        del all_outs
        clear_gpu_memory()
        gc.collect()

    print(f"\nEXPERIMENT COMPLETED")
    print(f"Results saved to: {exp_root}")

    del all_refs, multi_speaker_masks_mix, individual_speaker_masks_mix

    from models import cleanup_all_models
    cleanup_all_models()

    clear_gpu_memory()
    get_gpu_memory_info(verbose)
    gc.collect()

    return exp_root