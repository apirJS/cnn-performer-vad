"""inference.py
────────────────────────────────────────────────────────────────────────────
Frame‑ & clip‑level inference script for the VAD project.

It is intended to be called from *cli.py* via the `inference` sub‑command:

➜  python cli.py inference --input_files sample.wav --model_path vad_model.pt \
       --output_dir results

Key features
------------
* Accept TorchScript `.pt` **or** Lightning `.ckpt` models automatically.
* Processes a single file or an entire directory.
* Audios longer than 30 s are processed in 30 s chunks to keep memory usage stable.
* Generates **frame‑level** probabilities ➜ aggregated to a **clip‑level** score.
* Saves one CSV with all clip scores **and** optional per‑frame `.npy` dumps.
* Saves a PNG visualisation for every clip: log‑Mel spectrogram with speech
  regions highlighted.

The public entry‑point is `main(args)` so that `cli.py` can forward the parsed
`argparse.Namespace`. When run directly (`python inference.py ...`) we parse the
same arguments for convenience.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
import librosa.display as lbd
from tqdm import tqdm

from config import (
    DEFAULT_SAMPLE_RATE,
    DEFAULT_N_MELS,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_WIN_LENGTH,
)
from models import VADLightning  # Only needed when loading .ckpt

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CHUNK_SECONDS = 30  # max chunk length
FRAME_THRESHOLD = 0.5  # binary speech / non‑speech decision


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_path: Path, device: torch.device):
    """Load TorchScript **or** Lightning checkpoint transparently."""
    suffix = model_path.suffix.lower()
    if suffix == ".pt":
        logger.info(f"Loading TorchScript model from {model_path}")
        model = torch.jit.load(model_path, map_location=device).eval()
        is_scripted = True
    elif suffix == ".ckpt":
        logger.info(f"Loading Lightning checkpoint from {model_path}")
        # Load with strict=False to allow architectural differences
        lit_model = VADLightning.load_from_checkpoint(
            model_path, map_location=device, strict=False  # Allow missing or extra keys
        )
        lit_model.eval()
        model = lit_model
        is_scripted = False
    else:
        raise ValueError("Model must be a .pt or .ckpt file")

    model.to(device)
    return model, is_scripted


def _prepare_transforms(sample_rate: int, n_fft: int, hop: int, win: int, n_mels: int):
    """Return (mel_transform, db_transform) configured for training parameters."""
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mels=n_mels,
        power=1.0,
    )
    db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    return mel, db


def _chunk_indices(num_samples: int, sample_rate: int) -> List[Tuple[int, int]]:
    """Return (start, end) sample indices for ≤30 s chunks covering the audio."""
    chunk_len = CHUNK_SECONDS * sample_rate
    indices = [
        (s, min(num_samples, s + chunk_len)) for s in range(0, num_samples, chunk_len)
    ]
    return indices


def _run_model(mel_tensor: torch.Tensor, model, device: torch.device) -> np.ndarray:
    """Run the model on a (T, n_mels) tensor and return frame probabilities."""
    with torch.no_grad():
        logits = model(mel_tensor.unsqueeze(0).to(device))  # (1,T)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # (T,)
    return probs


def _visualise(
    path: Path,
    mel_db: np.ndarray,
    frame_probs: np.ndarray,
    out_file: Path,
    sample_rate: int,
    hop: int,
):
    """Save spectrogram visualisation with speech regions highlighted."""
    times = np.arange(len(frame_probs)) * hop / sample_rate
    speech_mask = frame_probs > FRAME_THRESHOLD

    # Build contiguous speech segments for faster plotting
    segments: List[Tuple[int, int]] = []
    in_seg = False
    seg_start = 0
    for i, v in enumerate(speech_mask):
        if v and not in_seg:
            in_seg = True
            seg_start = i
        elif not v and in_seg:
            segments.append((seg_start, i))
            in_seg = False
    if in_seg:
        segments.append((seg_start, len(speech_mask)))

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    img = lbd.specshow(
        mel_db.T,
        x_axis="time",
        y_axis="mel",
        sr=sample_rate,
        hop_length=hop,
        cmap="magma",
        ax=ax,
    )
    plt.colorbar(img, ax=ax, format="%.0f dB")

    # Overlay semi‑transparent rectangles for speech regions
    for s, e in segments:
        t0, t1 = times[s], times[e - 1]
        ax.axvspan(t0, t1, color="lime", alpha=0.25)

    ax.set_title(f"VAD prediction: {path.name}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel‑bin")
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Core processing
# ─────────────────────────────────────────────────────────────────────────────


def _process_audio(
    audio_path: Path,
    model,
    device: torch.device,
    mel_t,
    db_t,
    sample_rate: int,
    n_fft: int,
    hop: int,
    win: int,
    output_dir: Path,
    save_frames: bool = True,
):
    """Return clip_score, frame_probs and save artefacts (npy & PNG)."""
    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    wav = wav.mean(0, keepdim=True)  # mono

    # Move the audio tensor to the same device as the transforms
    wav = wav.to(device)

    total_samples = wav.shape[1]
    indices = _chunk_indices(total_samples, sample_rate)

    frame_probs_all = []
    mel_db_all = []  # for visualisation

    for s, e in indices:
        chunk = wav[:, s:e]
        mel = db_t(mel_t(chunk)).squeeze(0).transpose(0, 1)  # (T,n_mels)
        frame_probs = _run_model(mel, model, device)

        frame_probs_all.append(frame_probs)
        mel_db_all.append(mel.cpu().numpy())

    frame_probs_cat = np.concatenate(frame_probs_all, axis=0)
    mel_db_cat = np.concatenate(mel_db_all, axis=0)  # (T,n_mels)

    clip_score = float(frame_probs_cat.mean())

    # Save per‑frame probabilities for optional post‑analysis
    if save_frames:
        np.save(output_dir / f"{audio_path.stem}_frame_probs.npy", frame_probs_cat)

    # Save visualisation
    _visualise(
        audio_path,
        mel_db_cat,
        frame_probs_cat,
        output_dir / f"{audio_path.stem}_vad.png",
        sample_rate,
        hop,
    )

    return clip_score, frame_probs_cat


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace):
    """Entry‑point expected by *cli.py* (`args` already parsed)."""
    # Ensure output directory
    out_dir = Path(args.output_dir or "inference_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {out_dir}")

    device = torch.device(args.device)
    model, _ = _load_model(Path(args.model_path), device)

    mel_t, db_t = _prepare_transforms(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop=args.hop_length,
        win=args.win_length,
        n_mels=args.n_mels,
    )
    mel_t.to(device)
    db_t.to(device)

    # Resolve list of audio files
    if args.input_dir:
        audio_files = sorted(Path(args.input_dir).rglob("*.wav"))
    else:
        audio_files = [Path(p) for p in args.input_files]

    if not audio_files:
        logger.error("No audio files found for inference")
        return 1

    csv_path = out_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "clip_score"])

        for wav_path in tqdm(audio_files, desc="Inferencing"):
            clip_score, _ = _process_audio(
                wav_path,
                model,
                device,
                mel_t,
                db_t,
                args.sample_rate,
                args.n_fft,
                args.hop_length,
                args.win_length,
                out_dir,
            )
            writer.writerow([wav_path, f"{clip_score:.6f}"])

    logger.info("✅ Inference completed.")
    logger.info(f"CSV saved at {csv_path}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("VAD Inference")
    # Re‑use the same arguments as defined in cli.py
    input_grp = p.add_mutually_exclusive_group(required=True)
    input_grp.add_argument("--input_dir")
    input_grp.add_argument("--input_files", nargs="+")

    p.add_argument("--model_path", required=True)
    p.add_argument("--output_dir", default="inference_results")

    p.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE)
    p.add_argument("--n_fft", type=int, default=DEFAULT_N_FFT)
    p.add_argument("--hop_length", type=int, default=DEFAULT_HOP_LENGTH)
    p.add_argument("--win_length", type=int, default=DEFAULT_WIN_LENGTH)
    p.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS)

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args_ = p.parse_args()
    sys.exit(main(args_))
