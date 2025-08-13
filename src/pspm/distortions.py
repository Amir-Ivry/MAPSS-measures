from __future__ import annotations

import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import butter, filtfilt, lfilter
import librosa


def _signal_stats(x: np.ndarray, eps: float = 1e-6) -> tuple[float, float, float]:
    A_pk = max(np.max(np.abs(x)), eps)
    A_rms = max(np.sqrt(np.mean(x ** 2)), eps)
    A_95 = max(np.percentile(np.abs(x), 95), eps)
    return A_pk, A_rms, A_95


def apply_distortions(ref: np.ndarray, distortion_keys: str, sr: int) -> list[np.ndarray]:
    """Simple whole-signal distortions (deterministic)."""
    distortions: dict[str, np.ndarray] = {}
    X = rfft(ref)
    freqs = rfftfreq(len(ref), 1 / sr)
    t = np.arange(len(ref)) / sr

    if ("notch" in distortion_keys) or distortion_keys == "all":
        for c in [500, 1000, 2000, 4000, 8000]:
            Y = X.copy()
            Y[(freqs > c - 50) & (freqs < c + 50)] = 0
            distortions[f"Notch_{c}Hz"] = irfft(Y, n=len(ref))

    if ("comb" in distortion_keys) or distortion_keys == "all":
        for d_ms, decay in zip([2.5, 5, 7.5, 10, 12.5, 15], [0.4, 0.5, 0.6, 0.7, 0.9]):
            D = int(sr * d_ms / 1000)
            if D < len(ref):
                out = ref.copy()
                out[:-D] += decay * ref[D:]
                distortions[f"Comb_{int(d_ms)}ms"] = out

    if ("tremolo" in distortion_keys) or distortion_keys == "all":
        for r, depth in zip([1, 2, 4, 6], [0.3, 0.5, 0.8, 1.0]):
            mod = (1 - depth) + depth * 0.5 * (1 + np.sin(2 * np.pi * r * t))
            distortions[f"Tremolo_{r}Hz"] = ref * mod

    if ("noise" in distortion_keys) or distortion_keys == "all":
        def add_noise(signal, snr_db, color):
            rms = np.sqrt(np.mean(signal ** 2))
            nl = 10 ** (snr_db / 10)
            noise_rms = rms / np.sqrt(nl)
            n = np.random.randn(len(signal))
            if color == "pink":
                n = np.cumsum(n)
                n /= np.max(np.abs(n)) + 1e-9
            elif color == "brown":
                n = np.cumsum(np.cumsum(n))
                n /= np.max(np.abs(n)) + 1e-9
            return signal + noise_rms * n

        for snr in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
            for clr in ["white", "pink", "brown"]:
                if snr in [-15, -10, -5] and clr == "white":
                    continue
                distortions[f"{clr.capitalize()}Noise_{snr}dB"] = add_noise(ref, snr, clr)

    if ("harmonic" in distortion_keys) or distortion_keys == "all":
        for f_h, amp in zip([100, 500, 1000, 4000], [0.02, 0.03, 0.05, 0.08]):
            tone = amp * np.sin(2 * np.pi * f_h * t)
            distortions[f"Harmonic_{f_h}Hz"] = ref + tone

    if ("reverb" in distortion_keys) or distortion_keys == "all":
        for tail_ms, decay in zip([5, 10, 15, 20], [0.3, 0.5, 0.7, 0.9]):
            L = int(sr * tail_ms / 1000)
            if L < len(ref):
                irv = np.exp(-np.linspace(0, 3, L)) * decay
                reverbed = np.convolve(ref, irv)[: len(ref)]
                distortions[f"Reverb_{tail_ms}ms"] = reverbed

    if ("noisegate" in distortion_keys) or distortion_keys == "all":
        for thr in [0.005, 0.01, 0.02, 0.04]:
            g = ref.copy()
            g[np.abs(g) < thr] = 0
            distortions[f"NoiseGate_{thr}"] = g

    if ("pitch_shift" in distortion_keys) or distortion_keys == "all":
        n_fft = min(2048, len(ref) // 2)
        for shift in [-4, -2, 2, 4]:
            shifted = librosa.effects.pitch_shift(y=ref, sr=sr, n_steps=shift, n_fft=n_fft)
            distortions[f"PitchShift_{shift}st"] = shifted[: len(ref)]

    if ("lowpass" in distortion_keys) or distortion_keys == "all":
        for freq in [2000, 3000, 4000, 6000]:
            if freq < sr / 2:
                b, a = butter(4, freq / (sr / 2), "low")
                distortions[f"Lowpass_{freq}Hz"] = filtfilt(b, a, ref)

    if ("highpass" in distortion_keys) or distortion_keys == "all":
        for freq in [100, 300, 500, 800]:
            if freq < sr / 2:
                b, a = butter(4, freq / (sr / 2), "high")
                distortions[f"Highpass_{freq}Hz"] = filtfilt(b, a, ref)

    if ("echo" in distortion_keys) or distortion_keys == "all":
        for delay_ms, amp in zip([5, 10, 15, 20], [0.3, 0.5, 0.7]):
            D = int(sr * delay_ms / 1000)
            if D < len(ref):
                echo = np.pad(ref, (D, 0), "constant")[:-D] * amp
                distortions[f"Echo_{delay_ms}ms"] = ref + echo

    if ("clipping" in distortion_keys) or distortion_keys == "all":
        for thr in [0.3, 0.5, 0.7]:
            distortions[f"Clipping_{thr}"] = np.clip(ref, -thr, thr)

    if ("vibrato" in distortion_keys) or distortion_keys == "all":
        for rate, depth in zip([3, 5, 7], [0.001, 0.002, 0.003]):
            vibrato = np.sin(2 * np.pi * rate * t) * depth
            vibrato_signal = librosa.effects.time_stretch(ref, rate=1 + vibrato.mean(), n_fft=min(2048, len(ref) // 2))
            distortions[f"Vibrato_{rate}Hz"] = librosa.util.fix_length(vibrato_signal, size=len(ref))

    return list(distortions.values())


def apply_adv_distortions(ref: np.ndarray, distortion_keys: str, sr: int, frame_ms: int) -> list[np.ndarray]:
    """Content-aware framewise distortions (more diverse)."""
    distortions: dict[str, np.ndarray] = {}
    frame_len = int(frame_ms * sr / 1000)
    n_frames = int(np.ceil(len(ref) / frame_len))

    pad_len = n_frames * frame_len - len(ref)
    ref_padded = np.concatenate([ref, np.zeros(pad_len, dtype=ref.dtype)]) if pad_len else ref

    X_full = rfft(ref_padded)
    freqs_f = rfftfreq(len(ref_padded), 1 / sr)
    mag_full = np.abs(X_full)

    valid = (freqs_f > 80) & (freqs_f < 0.45 * sr)
    cand_indices = np.argsort(mag_full[valid])[-60:]
    cand_freqs = freqs_f[valid][cand_indices]
    cand_freqs = cand_freqs[np.argsort(mag_full[valid][cand_indices])[::-1]]

    selected_notch_freqs: list[float] = []
    for f0 in cand_freqs:
        if all(abs(f0 - f_sel) > 300 for f_sel in selected_notch_freqs):
            selected_notch_freqs.append(float(f0))
        if len(selected_notch_freqs) >= 20:
            break

    mag2 = np.abs(X_full) ** 2
    total_p = mag2.sum()
    cum_low = np.cumsum(mag2)
    q_low = [0.50, 0.70, 0.85, 0.95]
    lowpass_cutoffs = []
    for q in q_low:
        idx = np.searchsorted(cum_low, q * total_p)
        lowpass_cutoffs.append(round(float(freqs_f[idx]) / 100.0) * 100)

    cum_high = np.cumsum(mag2[::-1])
    q_high = [0.05, 0.15, 0.30, 0.50]
    highpass_cutoffs = []
    for q in q_high:
        idx = np.searchsorted(cum_high, q * total_p)
        highpass_cutoffs.append(round(float(freqs_f[-1 - idx]) / 100.0) * 100)

    lowpass_cutoffs = sorted(set(lowpass_cutoffs))
    highpass_cutoffs = sorted(set(highpass_cutoffs))

    for f in range(n_frames):
        start, end = f * frame_len, (f + 1) * frame_len
        frame = ref_padded[start:end]
        # re-use the simple distortions with frame-local parameters
        for lbl, sig in _frame_distortions(
            frame, sr, distortion_keys, selected_notch_freqs, lowpass_cutoffs, highpass_cutoffs, frame_start=start
        ).items():
            if lbl not in distortions:
                distortions[lbl] = np.zeros_like(ref_padded)
            distortions[lbl][start:end] = sig

    return list(distortions.values())


def _frame_distortions(
    frame: np.ndarray,
    sr: int,
    distortion_keys: str,
    notch_freqs: list[float],
    low_cutoffs: list[float],
    high_cutoffs: list[float],
    frame_start: int = 0,
) -> dict[str, np.ndarray]:
    A_pk, A_rms, A_95 = _signal_stats(frame)
    frame_len = len(frame)
    X = rfft(frame)
    freqs = rfftfreq(frame_len, 1 / sr)
    t = np.arange(frame_len) / sr

    out: dict[str, np.ndarray] = {}

    if ("notch" in distortion_keys) or distortion_keys == "all":
        bw = 60.0
        for f0 in notch_freqs:
            Y = X.copy()
            band = (freqs > f0 - bw) & (freqs < f0 + bw)
            Y[band] = 0
            out[f"Notch_{int(round(f0))}Hz"] = irfft(Y, n=len(frame))

    if ("comb" in distortion_keys) or distortion_keys == "all":
        for d_ms, decay in zip([2.5, 5, 7.5, 10, 12.5, 15], [0.4, 0.5, 0.6, 0.7, 0.9]):
            D = int(sr * d_ms / 1000)
            if D < frame_len:
                o = frame.copy()
                o[:-D] += decay * frame[D:]
                out[f"Comb_{int(d_ms)}ms"] = o

    if ("tremolo" in distortion_keys) or distortion_keys == "all":
        depth = 1.0
        t_centre = (frame_start + 0.5 * len(frame)) / sr
        for r_hz in [1, 2, 4, 6]:
            mod = (1 - depth) + depth * 0.5 * (1 + np.sin(2 * np.pi * r_hz * t_centre))
            out[f"Tremolo_{r_hz}Hz"] = frame * mod

    if ("noise" in distortion_keys) or distortion_keys == "all":
        nyq = sr / 2
        low_norm = 20 / nyq
        high_freq = min(20_000, 0.45 * sr)
        high_norm = min(high_freq / nyq, 0.99)
        b, a = butter(5, [low_norm, high_norm], btype="band")

        def _add_noise(sig, snr_db, color="white"):
            nl_target = 10 ** (snr_db / 10)
            n = np.random.randn(len(sig))
            if color == "pink":
                n = np.cumsum(n); n /= np.max(np.abs(n)) + 1e-9
            elif color == "brown":
                n = np.cumsum(np.cumsum(n)); n /= np.max(np.abs(n)) + 1e-9
            n = lfilter(b, a, n)
            rms_sig = np.sqrt(np.mean(sig ** 2))
            rms_n = np.sqrt(np.mean(n ** 2))
            noise_rms = max(rms_sig / np.sqrt(nl_target), rms_sig / np.sqrt(10 ** (15 / 10)))
            n *= noise_rms / (rms_n + 1e-12)
            return sig + n

        for snr in [-15, -10, -5, 0, 5, 10, 15, 20, 25]:
            for clr in ["white", "pink", "brown"]:
                if (snr in [-15, -10, -5]) and (clr == "white"):
                    continue
                out[f"{clr.capitalize()}Noise_{snr}dB"] = _add_noise(frame, snr, clr)

    if ("harmonic" in distortion_keys) or distortion_keys == "all":
        for f_h, rel_amp in zip([100, 500, 1000, 4000], [0.4, 0.6, 0.8, 1.0]):
            tone = (rel_amp * A_rms) * np.sin(2 * np.pi * f_h * t)
            out[f"Harmonic_{f_h}Hz"] = frame + tone

    if ("reverb" in distortion_keys) or distortion_keys == "all":
        for tail_ms, decay in zip([50, 100, 200, 400], [0.3, 0.5, 0.7, 0.9]):
            L = int(sr * tail_ms / 1000)
            if L < frame_len:
                irv = np.exp(-np.linspace(0, 6, L)) * decay
                out[f"Reverb_{tail_ms}ms"] = np.convolve(frame, irv)[: frame_len]

    if ("noisegate" in distortion_keys) or distortion_keys == "all":
        for pct in [0.05, 0.10, 0.20, 0.40]:
            thr = pct * A_95
            g = frame.copy()
            g[np.abs(g) < thr] = 0
            out[f"NoiseGate_{int(pct*100)}pct"] = g

    if ("pitch_shift" in distortion_keys) or distortion_keys == "all":
        n_fft = min(2048, frame_len // 2)
        for shift in [-4, -2, 2, 4]:
            y = librosa.effects.pitch_shift(frame, sr=sr, n_steps=shift, n_fft=n_fft)
            out[f"PitchShift_{shift}st"] = y[: frame_len]

    if ("lowpass" in distortion_keys) or distortion_keys == "all":
        for fc in low_cutoffs:
            if fc < sr / 2 * 0.99:
                b, a = butter(6, fc / (sr / 2), btype="low")
                out[f"Lowpass_{fc}Hz"] = filtfilt(b, a, frame)

    if ("highpass" in distortion_keys) or distortion_keys == "all":
        for fc in high_cutoffs:
            if fc > 20:
                b, a = butter(6, fc / (sr / 2), btype="high")
                out[f"Highpass_{fc}Hz"] = filtfilt(b, a, frame)

    if ("echo" in distortion_keys) or distortion_keys == "all":
        for delay_ms, amp in zip([50, 100, 150], [0.4, 0.5, 0.7]):
            D = int(sr * delay_ms / 1000)
            if D < frame_len:
                echo = np.pad(frame, (D, 0), "constant")[:-D] * amp
                out[f"Echo_{delay_ms}ms"] = frame + echo

    if ("clipping" in distortion_keys) or distortion_keys == "all":
        for frac in [0.70, 0.50, 0.30]:
            thr = frac * A_95
            out[f"Clipping_{frac:.2f}p95"] = np.clip(frame, -thr, thr)

    if ("vibrato" in distortion_keys) or distortion_keys == "all":
        n_fft = min(2048, frame_len // 2)
        base_depth = 0.03 * (A_rms / A_pk)
        for rate_hz, scale in zip([3, 5, 7], [1.0, 1.3, 1.6]):
            depth = np.clip(base_depth * scale, 0.01, 0.05)
            y = librosa.effects.time_stretch(frame, rate=1 + depth, n_fft=n_fft)
            out[f"Vibrato_{rate_hz}Hz"] = librosa.util.fix_length(y, size=frame_len)

    return out
