#!/usr/bin/env python
"""
Inference script for frame‑level VAD model
=========================================
Features
--------
* Loads a `VADLightning` checkpoint (.ckpt) produced by *train.py*.
* Runs frame‑level inference on one or more WAV/FLAC files.
* Converts per‑frame probabilities into speech regions (timestamps).
* Saves the regions to a CSV file **<audio_basename>_speech_segments.csv**.
* Optionally plots the log‑Mel spectrogram with speech regions highlighted
  and stores the figure as **<audio_basename>_speech.png**.

Usage
-----
```bash
python inference.py \
  --checkpoint lightning_logs/version_2/checkpoints/06-0.1672-0.9295.ckpt \
  --audio my_clip.wav another_clip.flac \
  --threshold 0.5 --device cuda
```
"""
from __future__ import annotations
import argparse, csv, pathlib, sys, logging
from typing import List, Tuple

import numpy as np
import torch
import librosa
import librosa.display as libdisplay
import matplotlib.pyplot as plt

from config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_MELS,
)
from models import VADLightning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_model(ckpt_path: pathlib.Path, device: torch.device) -> VADLightning:
    """Load model from Lightning checkpoint handling the hp namespace bug."""
    try:
        model = VADLightning.load_from_checkpoint(str(ckpt_path), map_location=device)
    except (AttributeError, TypeError):  # Add TypeError to catch missing hp argument
        # Fallback – replicate logic from evaluation.py
        ckpt = torch.load(ckpt_path, map_location=device)
        hp = argparse.Namespace(**ckpt["hyper_parameters"])
        model = VADLightning(hp)
        model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval().to(device)
    logger.info("Loaded model on %s", device)
    return model

def mel_spectrogram(
    wav: np.ndarray,
    sr: int = DEFAULT_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop: int = DEFAULT_HOP_LENGTH,
    n_mels: int = DEFAULT_N_MELS,
) -> np.ndarray:
    """Compute log‑Mel spectrogram in dB (T, n_mels)."""
    S = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=1.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T  # (T, n_mels)

def extract_regions(
    probs: np.ndarray,
    frame_shift: float,
    thr: float = 0.5,
    min_speech: float = 0.2,
    min_gap: float = 0.1,
) -> List[Tuple[float, float]]:
    """Convert probability sequence into merged speech regions."""
    speech = probs >= thr
    if not speech.any():
        return []
    indices = np.where(speech)[0]
    segments = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:
            end = indices[i - 1]
            segments.append((start, end))
            start = indices[i]
    segments.append((start, indices[-1]))

    # convert to seconds, merge / filter
    merged = []
    for s, e in segments:
        t0, t1 = s * frame_shift, (e + 1) * frame_shift
        if merged and t0 - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], t1)
        else:
            merged.append((t0, t1))
    # filter by min_speech length
    merged = [(a, b) for a, b in merged if b - a >= min_speech]
    return merged

def save_csv(csv_path: pathlib.Path, regions: List[Tuple[float, float]]):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["start_sec", "end_sec"])
        writer.writerows(regions)
    logger.info("Saved %d regions → %s", len(regions), csv_path)

def process_audio(
    audio_path: pathlib.Path,
    model: VADLightning,
    device: torch.device,
    threshold: float,
):
    wav, sr = librosa.load(str(audio_path), sr=DEFAULT_SAMPLE_RATE)
    mel = mel_spectrogram(wav, sr)
    with torch.no_grad():
        inp = torch.tensor(mel).unsqueeze(0).float().to(device)  # (1, T, n_mels)
        probs = torch.sigmoid(model(inp)).cpu().numpy()[0]  # (T,)
    frame_shift = DEFAULT_HOP_LENGTH / DEFAULT_SAMPLE_RATE
    regions = extract_regions(probs, frame_shift, thr=threshold)

    # Save outputs - fix path construction
    stem = audio_path.stem  # Get filename without extension
    csv_path = f"./audio/{stem}_speech_segments.csv"
    save_csv(csv_path, regions)
    png_path = f"./audio/{stem}_speech.png"
    plot_regions(wav, sr, mel, regions, png_path)

def plot_regions(
    wav: np.ndarray,
    sr: int,
    mel: np.ndarray,
    regions: List[Tuple[float, float]],
    png_path: pathlib.Path,
):
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot waveform in top subplot
    time = np.arange(len(wav)) / sr
    ax1.plot(time, wav, color='gray', alpha=0.8)
    ax1.set_title("Raw Audio Waveform with Speech Regions")
    ax1.set_ylabel("Amplitude")
    # Highlight speech regions in waveform
    for s, e in regions:
        ax1.axvspan(s, e, color="green", alpha=0.3)
    
    # Plot spectrogram in bottom subplot
    libdisplay.specshow(
        mel.T,
        sr=sr,
        hop_length=DEFAULT_HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax2,
    )
    # Highlight speech regions in spectrogram
    for s, e in regions:
        ax2.axvspan(s, e, color="cyan", alpha=0.4)
    ax2.set_title("Mel Spectrogram")
    
    # Adjust layout
    plt.tight_layout()
    fig.savefig(png_path, dpi=300)  # Higher DPI for better quality
    plt.close(fig)
    logger.info("Saved waveform and spectrogram plot → %s", png_path)


def main():
    p = argparse.ArgumentParser("Frame‑level VAD inference")
    p.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    p.add_argument("--audio", nargs="+", required=True, help="Audio files (wav/flac)")
    p.add_argument("--threshold", type=float, default=0.5, help="Speech prob threshold")
    p.add_argument("--device", default="cpu", help="cuda, cuda:0, cpu…")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = pathlib.Path(args.checkpoint)
    model = load_model(ckpt, device)

    for a in args.audio:
        audio_path = pathlib.Path(a)
        if not audio_path.exists():
            logger.error("File not found: %s", a)
            continue
        logger.info("Processing %s", audio_path.name)
        process_audio(audio_path, model, device, args.threshold)

if __name__ == "__main__":
    main()
