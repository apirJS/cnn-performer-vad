"""
vad_utils.py – Inference + visualization helpers for Mel‑Performer VAD
---------------------------------------------------------------------
Sasha © 2025
"""

import pathlib, math
import torch, torchaudio
import matplotlib.pyplot as plt
import numpy as np
import librosa.display
import logging

# Configure logging
logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────
#  0.  MODEL LOADER
# ───────────────────────────────────────────────────────────────
def load_vad_checkpoint(ckpt_path: str, hp) -> torch.nn.Module:
  """
  Return the *bare* MelPerformer (nn.Module) loaded with trained weights.

  Args
  ----
  ckpt_path : str | Path   – Lightning checkpoint (*.ckpt or *.pt)
  hp        : argparse.Namespace or dict – needs the SAME model h‑params
          (n_mels, dim, n_layers, n_heads) used at train‑time.
  """
  logger.info(f"Loading VAD checkpoint from: {ckpt_path}")
  logger.debug(f"Model hyperparameters: n_mels={hp.n_mels}, dim={hp.dim}, n_layers={hp.n_layers}, n_heads={hp.n_heads}")
  
  from train import VADLightning  # re‑use lightning wrapper

  # ① Build lightning module stub ➜ ② load state dict ➜ ③ return .net
  logger.debug("Initializing VADLightning module")
  lit = VADLightning(hp)
  
  logger.debug("Loading state dictionary from checkpoint")
  sd  = torch.load(ckpt_path, map_location="cpu")
  if isinstance(sd, dict) and "state_dict" in sd:
    logger.debug("Found Lightning-style state dictionary")
    sd = sd["state_dict"]                     # lightning wrapper style
  
  logger.debug("Loading state dictionary into model")
  lit.load_state_dict(sd, strict=False)
  lit.eval()
  
  logger.info("VAD model loaded successfully")
  return lit.net                                # raw MelPerformer


# ───────────────────────────────────────────────────────────────
#  1.  FRAME‑LEVEL INFERENCE
# ───────────────────────────────────────────────────────────────
def predict_windows(
  wav: torch.Tensor,
  sr: int,
  model: torch.nn.Module,
  win_ms: int = 1000,
  hop_ms: int = 500,
  n_mels: int = 64,
  n_fft: int = 400,
  hop_len: int = 160,
):
  """
  Slide a window over `wav` and spit out per‑window speech probabilities.

  Intuitive 🎈 : like sliding a magnifying glass along a tape‑recorder
          and asking "talking or not?" every 0.5 s.
  Formal 📐    : For each window ➜ MelSpec ∈ ℝ^{T×n_mels}
           ➜ model(x.unsqueeze(0)) ➜ σ(logit) ∈ (0,1).

  Returns
  -------
  probs : 1‑D numpy array of shape (n_windows,)
  times : 1‑D array – centre‑time of each window in seconds
  """
  logger.info(f"Starting prediction with window size: {win_ms}ms, hop size: {hop_ms}ms")
  logger.debug(f"Audio shape: {wav.shape}, sample rate: {sr}Hz")
  
  device = next(model.parameters()).device
  logger.debug(f"Using device: {device}")
  
  win_s  = int(sr * win_ms / 1000)
  hop_s  = int(sr * hop_ms / 1000)
  n_win  = max(1, math.floor((wav.numel() - win_s) / hop_s) + 1)
  logger.debug(f"Window size: {win_s} samples, hop size: {hop_s} samples, total windows: {n_win}")

  logger.debug("Initializing MelSpectrogram transform")
  mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=n_fft, hop_length=hop_len,
    win_length=n_fft, n_mels=n_mels, power=1.0).to(device)
  to_db  = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)

  probs, centres = [], []
  logger.info(f"Processing {n_win} windows...")
  for i in range(n_win):
    s, e = i * hop_s, i * hop_s + win_s
    chunk = wav[..., s:e].clamp(-1, 1).to(device)                # (1,T)
    if chunk.numel() < win_s:                       # pad tail quietly
      logger.debug(f"Padding window {i} ({chunk.numel()} < {win_s})")
      chunk = torch.nn.functional.pad(chunk, (0, win_s - chunk.numel()))
    mel  = to_db(mel_tf(chunk)).squeeze(0).T        # (T,n_mels)
    with torch.no_grad():
      p = torch.sigmoid(model(mel.unsqueeze(0))).item()
    probs.append(p)
    centres.append((s + e) / 2 / sr)
    
    if i % 10 == 0:  # Log progress periodically
      logger.debug(f"Processed window {i}/{n_win}")

  logger.info(f"Prediction complete, obtained {len(probs)} probability values")
  return np.array(probs), np.array(centres)


# ───────────────────────────────────────────────────────────────
#  2.  VISUALISER
# ───────────────────────────────────────────────────────────────
def plot_predictions(audio_path: str | pathlib.Path,
           model: torch.nn.Module,
           n_mels: int = 64,
           figsize=(12, 8)):
  """
  Plot waveform, log‑mel spectrogram, and VAD probs underneath.

  Parameters
  ----------
  audio_path : str | Path – input WAV/FLAC/OGG…
  model      : nn.Module  – MelPerformer returned by `load_vad_checkpoint`
  n_mels     : int        – must match training setting
  figsize    : tuple      – (w, h) inches
  """
  logger.info(f"Plotting predictions for audio file: {audio_path}")
  
  logger.debug("Loading audio file")
  wav, sr = torchaudio.load(audio_path)
  wav = wav.mean(0, keepdim=True)   # mono
  logger.debug(f"Loaded audio: {wav.shape}, sr={sr}Hz, duration={wav.numel()/sr:.2f}s")

  # —–– compute frame‑wise probabilities
  logger.info("Computing frame-wise speech probabilities")
  probs, times = predict_windows(wav, sr, model, n_mels=n_mels)
  logger.debug(f"Generated {len(probs)} probability values over {times[-1]:.2f} seconds")

  # —–– prepare mel spectrogram for pretty colours
  logger.debug("Computing mel spectrogram for visualization")
  mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr, n_fft=400, hop_length=160,
    win_length=400, n_mels=n_mels, power=1.0)
  mel  = mel_tf(wav).squeeze(0)
  mel_db = torchaudio.transforms.AmplitudeToDB(top_db=80)(mel)

  # —–– draw ✨
  logger.info("Creating visualization with matplotlib")
  fig, ax = plt.subplots(3, 1, figsize=figsize,
               gridspec_kw={"height_ratios": [1, 2, 1]},
               sharex=True)

  # ① Waveform
  logger.debug("Plotting waveform")
  ax[0].plot(np.linspace(0, wav.numel()/sr, wav.numel()),
         wav.squeeze().numpy(), linewidth=0.6)
  ax[0].set(title="Waveform", ylabel="Amplitude")

  # ② Spectrogram
  logger.debug("Plotting spectrogram")
  img = librosa.display.specshow(mel_db.numpy(),
                   sr=sr, hop_length=160,
                   x_axis="time", y_axis="mel",
                   fmax=sr/2, ax=ax[1])
  ax[1].set(title="Log‑Mel Spectrogram")
  fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

  # ③ VAD probability track
  logger.debug("Plotting VAD probability track")
  ax[2].plot(times, probs, marker="o", linewidth=1)
  ax[2].set(title="Speech probability", xlabel="Time [s]",
        ylim=(-0.05, 1.05))
  ax[2].fill_between(times, 0, probs, alpha=0.3)

  plt.tight_layout()
  logger.info("Displaying visualization")
  plt.show()


# ───────────────────────────────────────────────────────────────
#  3.  QUICK DEMO
# ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
  import argparse, textwrap

  logger.info("Starting VAD utility demo")
  
  p = argparse.ArgumentParser(
    description="Tiny demo – pass --ckpt ckpt_path --wav some.wav")
  p.add_argument("--ckpt", required=True)
  p.add_argument("--wav",  required=True)
  args = p.parse_args()
  
  logger.info(f"Arguments: checkpoint={args.ckpt}, audio={args.wav}")

  # Minimal hyper‑param bundle matching DEFAULTS in train.py
  hp = argparse.Namespace(n_mels=64, dim=256, n_layers=4, n_heads=4)
  logger.debug(f"Using default hyperparameters: {vars(hp)}")

  logger.info("Loading model checkpoint")
  net = load_vad_checkpoint(args.ckpt, hp)   # 🚀
  
  logger.info("Generating and displaying predictions")
  plot_predictions(args.wav, net)
  
  logger.info("Demo completed successfully")
