import queue
import threading
import gc

import torch
import torch.nn.functional as F
from transformers import (
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    WavLMModel,
)

from config import BATCH_SIZE, ENERGY_HOP_MS, ENERGY_WIN_MS, SR
from utils import get_gpu_count


class BalancedMultiGPUModel:
    """
    Distributes model inference workload across GPUs.
    """
    def __init__(self, model_name, layer, max_gpus=None):
        self.layer = layer
        self.models = []
        self.extractors = []
        self.devices = []
        ngpu = get_gpu_count(max_gpus)

        for gpu_id in range(ngpu):
            device = f"cuda:{gpu_id}"
            self.devices.append(device)
            ckpt, cls, _ = get_model_config(layer)[model_name]
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)

            attn_impl = "eager" if cls is WavLMModel else "sdpa"
            model = cls.from_pretrained(
                ckpt,
                output_hidden_states=True,
                use_safetensors=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl
            )
            model.eval()
            model = model.to(device)

            for param in model.parameters():
                param.requires_grad = False

            self.extractors.append(extractor)
            self.models.append(model)

        self.gpu_queues = [queue.Queue() for _ in range(len(self.devices))]
        self.result_queue = queue.Queue()
        self.workers = []
        for i in range(len(self.devices)):
            worker = threading.Thread(target=self._gpu_worker, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _gpu_worker(self, gpu_id):
        device = self.devices[gpu_id]
        model = self.models[gpu_id]
        extractor = self.extractors[gpu_id]
        while True:
            task = self.gpu_queues[gpu_id].get()
            if task is None:
                break
            signals, masks, use_mlm, task_id = task
            try:
                inputs = extractor(
                    signals, sampling_rate=SR, return_tensors="pt", padding=True
                )
                input_values = inputs.input_values.to(device, non_blocking=True)

                torch.cuda.empty_cache()

                orig_mode = model.training
                model.train() if use_mlm else model.eval()
                with torch.no_grad():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        hs = model(
                            input_values, output_hidden_states=True
                        ).hidden_states[self.layer]
                model.train(orig_mode)

                B, T, D = hs.shape
                keep = []
                for b in range(B):
                    mask_b = masks[b].float().unsqueeze(0).unsqueeze(0).to(device)
                    mask_t = F.interpolate(mask_b, size=T, mode="nearest")[0, 0].bool()
                    keep.append(hs[b][mask_t].cpu())

                del hs, input_values, inputs
                torch.cuda.empty_cache()

                if keep:
                    L_max = max(x.shape[0] for x in keep)
                    keep_padded = [
                        F.pad(x, (0, 0, 0, L_max - x.shape[0])) for x in keep
                    ]
                    result = torch.stack(keep_padded, dim=0)
                else:
                    result = torch.empty(0, 0, 0)
                self.result_queue.put((task_id, result))
            except Exception as e:
                self.result_queue.put((task_id, e))
            finally:
                torch.cuda.empty_cache()

    def process_batch(self, signals, masks, use_mlm=False):
        if not signals:
            return torch.empty(0, 0, 0)
        batch_size = len(signals)
        split = (batch_size + len(self.devices) - 1) // len(self.devices)
        results = {}
        task_id = 0
        for i in range(0, batch_size, split):
            end = min(i + split, batch_size)
            gpu_id = (i // split) % len(self.devices)
            self.gpu_queues[gpu_id].put(
                (signals[i:end], masks[i:end], use_mlm, task_id)
            )
            task_id += 1
        for _ in range(task_id):
            tid, result = self.result_queue.get()
            if isinstance(result, Exception):
                raise result
            results[tid] = result
        parts = [results[i] for i in range(task_id) if results[i].numel() > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 0, 0)

    def cleanup(self):
        """Explicit cleanup method"""
        for q in self.gpu_queues:
            q.put(None)
        for w in self.workers:
            w.join(timeout=5.0)
        for model in self.models:
            del model
        for extractor in self.extractors:
            del extractor
        self.models.clear()
        self.extractors.clear()
        torch.cuda.empty_cache()
        gc.collect()

    def __del__(self):
        self.cleanup()


def get_model_config(layer):
    """
    Get self-supervised model configuration.
    :param layer: specific transformer layer to choose.
    :return: Configuration.
    """
    return {
        "raw": (None, None, None),
        "wavlm": ("microsoft/wavlm-large", WavLMModel, layer),
        "wav2vec2": ("facebook/wav2vec2-large-lv60", Wav2Vec2Model, layer),
        "hubert": ("facebook/hubert-large-ll60k", HubertModel, layer),
        "wavlm_base": ("microsoft/wavlm-base", WavLMModel, layer),
        "wav2vec2_base": ("facebook/wav2vec2-base", Wav2Vec2Model, layer),
        "hubert_base": ("facebook/hubert-base-ls960", HubertModel, layer),
        "wav2vec2_xlsr": ("facebook/wav2vec2-large-xlsr-53", Wav2Vec2Model, layer),
    }


_loaded_models = {}


def load_model(name, layer, max_gpus=None):
    """
    Load the chosen self-supervised model.
    :param name: name of model.
    :param layer: chosen layer.
    :param max_gpus: maximal gpus to use.
    :return: extractor, model, and layer.
    """
    global _loaded_models

    if _loaded_models:
        for key, model_data in _loaded_models.items():
            if isinstance(model_data, tuple) and len(model_data) == 2:
                if isinstance(model_data[0], BalancedMultiGPUModel):
                    model_data[0].cleanup()
                elif isinstance(model_data[0], tuple):
                    _, model = model_data[0]
                    del model
        _loaded_models.clear()
        torch.cuda.empty_cache()
        gc.collect()

    if name.lower() in {"raw", "waveform"}:
        return "raw", layer

    ngpu = get_gpu_count(max_gpus)
    if ngpu > 1:
        model = BalancedMultiGPUModel(name, layer, max_gpus)
        _loaded_models[name] = (model, layer)
        return model, layer
    else:
        ckpt, cls, layer_eff = get_model_config(layer)[name]
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        attn_impl = "eager" if cls is WavLMModel else "sdpa"
        model = cls.from_pretrained(
            ckpt,
            output_hidden_states=True,
            use_safetensors=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl
        )
        model.eval()
        model = model.to(device)

        for param in model.parameters():
            param.requires_grad = False

        model_tuple = ((extractor, model), layer_eff)
        _loaded_models[name] = model_tuple
        return (extractor, model), layer_eff


def cleanup_all_models():
    """
    Call this at the end of each experiment to ensure complete cleanup
    """
    global _loaded_models
    if _loaded_models:
        for key, model_data in _loaded_models.items():
            if isinstance(model_data, tuple) and len(model_data) == 2:
                if isinstance(model_data[0], BalancedMultiGPUModel):
                    model_data[0].cleanup()
                elif isinstance(model_data[0], tuple):
                    _, model = model_data[0]
                    del model
        _loaded_models.clear()
    torch.cuda.empty_cache()
    gc.collect()


def embed_batch_raw(signals, masks_audio):
    """
    Waveform encoding in case it was chosen to skip self-supervised encording and push waveform directly to diffusion maps
    :param signals: waveform signals.
    :param masks_audio: voice activity masks of sources.
    :return:
    """
    win = int(ENERGY_WIN_MS * SR / 1000)
    hop = int(ENERGY_HOP_MS * SR / 1000)
    reps, L_max = [], 0
    for sig_np, mask_np in zip(signals, masks_audio):
        x = torch.as_tensor(sig_np[:-1], dtype=torch.float32)
        frames = x.unfold(0, win, hop)
        mask = torch.as_tensor(mask_np[: len(frames)], dtype=torch.bool)
        keep = frames[mask] if mask.any() else frames[:1]
        reps.append(keep)
        L_max = max(L_max, keep.size(0))
    reps = [F.pad(r, (0, 0, 0, L_max - r.size(0))) for r in reps]
    return torch.stack(reps, dim=0)


def embed_batch_single_gpu(
        signals, masks_audio, extractor, model, layer, use_mlm=False
):
    """
    See embed_batch.
    """
    if not signals:
        return torch.empty(0, 0, 0)
    device = next(model.parameters()).device

    max_batch = 2
    all_keeps = []

    for i in range(0, len(signals), max_batch):
        batch_signals = signals[i:i + max_batch]
        batch_masks = masks_audio[i:i + max_batch]

        inputs = extractor(batch_signals, sampling_rate=SR, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device, non_blocking=True)

        orig_mode = model.training
        model.train() if use_mlm else model.eval()

        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                hs = model(input_values, output_hidden_states=True).hidden_states[layer]
        model.train(orig_mode)

        B, T, D = hs.shape
        for b in range(B):
            mask_b = batch_masks[b].float().unsqueeze(0).unsqueeze(0).to(device)
            mask_t = F.interpolate(mask_b, size=T, mode="nearest")[0, 0].bool()
            all_keeps.append(hs[b][mask_t].cpu())

        del hs, input_values, inputs
        torch.cuda.empty_cache()

    if all_keeps:
        L_max = max(x.shape[0] for x in all_keeps)
        keep_padded = [F.pad(x, (0, 0, 0, L_max - x.shape[0])) for x in all_keeps]
        result = torch.stack(keep_padded, dim=0)
        del all_keeps, keep_padded
        return result
    else:
        return torch.empty(0, 0, 0)


def embed_batch(signals, masks_audio, model_wrapper, layer, use_mlm=False):
    """
    Encode a batch of signals using the self-supervised model chosen.

    :param signals: waveform signals to encode.
    :param masks_audio: voice activity masks of sources.
    :param model_wrapper: chosen model's wrapper.
    :param layer: transformer layer.
    :param use_mlm: deprecated.
    :return: embedded signal representations by the model's layer.
    """
    if model_wrapper == "raw":
        return embed_batch_raw(signals, masks_audio)
    if isinstance(model_wrapper, BalancedMultiGPUModel):
        all_embeddings = []
        batch_size = min(BATCH_SIZE, 2)
        for i in range(0, len(signals), batch_size):
            batch_emb = model_wrapper.process_batch(
                signals[i: i + batch_size], masks_audio[i: i + batch_size], use_mlm
            )
            if batch_emb.numel() > 0:
                all_embeddings.append(batch_emb)
            torch.cuda.empty_cache()

        if all_embeddings:
            result = torch.cat(all_embeddings, dim=0)
            del all_embeddings
            return result
        else:
            return torch.empty(0, 0, 0)
    else:
        extractor, model = model_wrapper
        return embed_batch_single_gpu(
            signals, masks_audio, extractor, model, layer, use_mlm=use_mlm
        )