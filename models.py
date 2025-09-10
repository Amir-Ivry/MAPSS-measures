"""Model loading and embedding - preserves ALL original functionality."""
import queue
import threading
from functools import lru_cache
import torch
import torch.nn.functional as F
from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
    WavLMModel,
    Wav2Vec2Model
)
from config import SR, BATCH_SIZE, ENERGY_WIN_MS, ENERGY_HOP_MS
from utils import get_gpu_count


class BalancedDualGPUModel:
    """Load one model per GPU and process in balanced chunks - EXACT from original."""

    def __init__(self, model_name, layer, max_gpus=None):
        self.layer = layer
        self.models = []
        self.extractors = []
        self.devices = []
        ngpu = get_gpu_count(max_gpus)

        for gpu_id in range(min(ngpu, 2)):
            device = f'cuda:{gpu_id}'
            self.devices.append(device)
            ckpt, cls, _ = get_model_config(layer)[model_name]
            extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
            model = cls.from_pretrained(
                ckpt,
                output_hidden_states=True,
                use_safetensors=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).eval().to(device)
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
                inputs = extractor(signals, sampling_rate=SR, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device, non_blocking=True)
                orig_mode = model.training
                model.train() if use_mlm else model.eval()
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        hs = model(input_values.half(), output_hidden_states=True).hidden_states[self.layer]
                model.train(orig_mode)
                B, T, D = hs.shape
                keep = []
                for b in range(B):
                    mask_b = masks[b].float().unsqueeze(0).unsqueeze(0).to(device)
                    mask_t = F.interpolate(mask_b, size=T, mode="nearest")[0, 0].bool()
                    keep.append(hs[b][mask_t])
                if keep:
                    L_max = max(x.shape[0] for x in keep)
                    keep_padded = [F.pad(x, (0, 0, 0, L_max - x.shape[0])) for x in keep]
                    result = torch.stack(keep_padded, dim=0).cpu()
                else:
                    result = torch.empty(0, 0, 0)
                self.result_queue.put((task_id, result))
            except Exception as e:
                self.result_queue.put((task_id, e))

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
            self.gpu_queues[gpu_id].put((signals[i:end], masks[i:end], use_mlm, task_id))
            task_id += 1
        for _ in range(task_id):
            tid, result = self.result_queue.get()
            if isinstance(result, Exception):
                raise result
            results[tid] = result
        parts = [results[i] for i in range(task_id) if results[i].numel() > 0]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, 0, 0)

    def __del__(self):
        for q in self.gpu_queues:
            q.put(None)
        for w in self.workers:
            w.join(timeout=5.0)
        for model in self.models:
            del model
        for extractor in self.extractors:
            del extractor
        torch.cuda.empty_cache()


@lru_cache(maxsize=4)
def get_model_config(layer):
    """Get model configuration exactly as original."""
    return {
        "raw": (None, None, None),
        "wavlm": ("microsoft/wavlm-large", WavLMModel, layer),
        "wav2vec2": ("facebook/wav2vec2-large-lv60", Wav2Vec2Model, layer),
        "hubert": ("facebook/hubert-large-ll60k", HubertModel, layer),
        "wav2vec2_xlsr": ("facebook/wav2vec2-large-xlsr-53", Wav2Vec2Model, layer),
        "wavlm_base": ("microsoft/wavlm-base", WavLMModel, layer),
        "wav2vec2_base": ("facebook/wav2vec2-base", Wav2Vec2Model, layer),
        "hubert_base": ("facebook/hubert-base-ls960", HubertModel, layer),
    }


@lru_cache(maxsize=4)
def load_model(name, layer, max_gpus=None):
    """Load model exactly as original."""
    if name.lower() in {"raw", "waveform"}:
        return "raw", layer

    ngpu = get_gpu_count(max_gpus)
    if ngpu > 1:
        return BalancedDualGPUModel(name, layer, max_gpus), layer
    else:
        ckpt, cls, layer_eff = get_model_config(layer)[name]
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
        model = cls.from_pretrained(
            ckpt,
            output_hidden_states=True,
            use_safetensors=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).eval().to('cuda:0' if torch.cuda.is_available() else 'cpu')
        return (extractor, model), layer_eff


def embed_batch_raw(signals, masks_audio):
    """Raw embedding exactly as original."""
    win = int(ENERGY_WIN_MS * SR / 1000)
    hop = int(ENERGY_HOP_MS * SR / 1000)
    reps, L_max = [], 0
    for sig_np, mask_np in zip(signals, masks_audio):
        x = torch.as_tensor(sig_np[:-1], dtype=torch.float32)
        frames = x.unfold(0, win, hop)
        mask = torch.as_tensor(mask_np[:len(frames)], dtype=torch.bool)
        keep = frames[mask] if mask.any() else frames[:1]
        reps.append(keep)
        L_max = max(L_max, keep.size(0))
    reps = [F.pad(r, (0, 0, 0, L_max - r.size(0))) for r in reps]
    return torch.stack(reps, dim=0)


def embed_batch_single_gpu(signals, masks_audio, extractor, model, layer, use_mlm=False):
    """Single GPU embedding exactly as original."""
    if not signals:
        return torch.empty(0, 0, 0)
    device = next(model.parameters()).device
    inputs = extractor(signals, sampling_rate=SR, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device, non_blocking=True)
    orig_mode = model.training
    model.train() if use_mlm else model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            hs = model(input_values, output_hidden_states=True).hidden_states[layer]
    model.train(orig_mode)
    B, T, D = hs.shape
    keep = []
    for b in range(B):
        mask_b = masks_audio[b].float().unsqueeze(0).unsqueeze(0).to(device)
        mask_t = F.interpolate(mask_b, size=T, mode="nearest")[0, 0].bool()
        keep.append(hs[b][mask_t])
    if keep:
        L_max = max(x.shape[0] for x in keep)
        keep_padded = [F.pad(x, (0, 0, 0, L_max - x.shape[0])) for x in keep]
        return torch.stack(keep_padded, dim=0)
    else:
        return torch.empty(0, 0, 0)


def embed_batch(signals, masks_audio, model_wrapper, layer, use_mlm=False):
    """Embed batch exactly as original."""
    if model_wrapper == "raw":
        return embed_batch_raw(signals, masks_audio)
    if isinstance(model_wrapper, BalancedDualGPUModel):
        all_embeddings = []
        batch_size = BATCH_SIZE
        for i in range(0, len(signals), batch_size):
            batch_emb = model_wrapper.process_batch(
                signals[i:i + batch_size],
                masks_audio[i:i + batch_size],
                use_mlm
            )
            if batch_emb.numel() > 0:
                all_embeddings.append(batch_emb)
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, 0, 0)
    else:
        extractor, model = model_wrapper
        return embed_batch_single_gpu(signals, masks_audio, extractor, model, layer, use_mlm=use_mlm)