#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  data.py - Dataset and data handling for VAD
# ───────────────────────────────────────────────────────────────────────
import csv
import json
import logging
import pathlib
import random
from typing import Tuple, Optional, List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from config import (
    DEFAULT_N_MELS,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_WIN_LENGTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_TIME_MASK_MAX,
    DEFAULT_FREQ_MASK_MAX,
    DEFAULT_MAX_FRAMES
)

# Configure module logger
logger = logging.getLogger(__name__)


class CSVMelDataset(Dataset):
    """
    On‑the‑fly converts WAV → log‑Mel spectrogram with frame-level VAD labels.
      • returns tensor (T, n_mels) and frame labels (T)
      • Variable length audio results in variable time frames
      • Optional caching for faster repeated access
    """

    def __init__(
        self,
        manifest: str,
        n_mels: int = DEFAULT_N_MELS,
        n_fft: int = DEFAULT_N_FFT,
        hop: int = DEFAULT_HOP_LENGTH,
        win: int = DEFAULT_WIN_LENGTH,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        cache_dir: Optional[str] = None,
        time_mask_max: int = DEFAULT_TIME_MASK_MAX,
        freq_mask_max: int = DEFAULT_FREQ_MASK_MAX,
    ):
        logger.info(f"Initializing frame-level VAD dataset from manifest: {manifest}")
        # Now reading in three columns: audio path, clip label, frame label path
        self.params = {
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop": hop,
            "win": win,
            "sample_rate": sample_rate,
        }

        self.manifest = manifest
        self.items = []

        for r in csv.DictReader(open(manifest)):
            if len(r) >= 3:  # Check if frame_labels column exists
                self.items.append(
                    (r["path"], int(r["label"]), r.get("frame_labels", ""))
                )
            else:
                self.items.append((r["path"], int(r["label"]), ""))

        logger.info(f"Loaded {len(self.items)} items from manifest")

        logger.info(
            f"Configuring mel spectrogram with n_mels={n_mels}, n_fft={n_fft}, hop={hop}, win={win}"
        )
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            n_mels=n_mels,
            power=1.0,
        )
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.sample_rate = sample_rate
        self.cache_dir = self._setup_cache_dir(cache_dir)
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

    def _setup_cache_dir(self, cache_dir: Optional[str]) -> Optional[pathlib.Path]:
        """Set up and validate the cache directory if provided."""
        if not cache_dir:
            return None
            
        cache_path = pathlib.Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        params_file = cache_path / "params.json"

        # Check if params file exists and matches current params
        if params_file.exists():
            try:
                with open(params_file, "r") as f:
                    stored_params = json.load(f)
                if stored_params != self.params:
                    logger.warning(f"Cache parameters mismatch! Clearing cache.")
                    import shutil

                    # Backup existing cache directory
                    backup_dir = (
                        cache_path.parent / f"{cache_path.name}_backup"
                    )
                    if backup_dir.exists():
                        shutil.rmtree(backup_dir)
                    cache_path.rename(backup_dir)
                    cache_path.mkdir(parents=True)
                    # Save new params
                    with open(params_file, "w") as f:
                        json.dump(self.params, f)
            except Exception as e:
                logger.error(f"Error validating cache parameters: {e}")
                # Create new params file
                with open(params_file, "w") as f:
                    json.dump(self.params, f)
        else:
            # Save params if file doesn't exist
            with open(params_file, "w") as f:
                json.dump(self.params, f)

        logger.info(f"Caching mel spectrograms to {cache_path}")
        return cache_path

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                path, clip_label, label_path = self.items[idx]
                path = pathlib.Path(path)
                label_path = pathlib.Path(label_path) if label_path else None

                # Try loading from cache first if enabled
                if self.cache_dir:
                    cache_path = self.cache_dir / f"{path.stem}.pt"
                    label_cache_path = self.cache_dir / f"{path.stem}_labels.pt"

                    if cache_path.exists() and (
                        not label_path or label_cache_path.exists()
                    ):
                        logger.debug(f"Loading cached data for {path.stem}")
                        mel = torch.load(cache_path)

                        # Load frame labels if available, otherwise use clip label for all frames
                        if label_path and label_cache_path.exists():
                            frame_labels = torch.load(label_cache_path)
                        else:
                            frame_labels = torch.full(
                                (mel.shape[0],), clip_label, dtype=torch.float32
                            )

                        return mel, frame_labels

                # Load and process audio file
                mel, frame_labels = self._process_audio_file(path, clip_label, label_path)
                
                # Cache result if enabled
                if self.cache_dir:
                    logger.debug(f"Caching data for {path.stem}")
                    torch.save(mel, self.cache_dir / f"{path.stem}.pt")
                    torch.save(frame_labels, self.cache_dir / f"{path.stem}_labels.pt")

                return mel, frame_labels

            except Exception as e:
                logger.warning(
                    f"Error loading {path} (attempt {attempt+1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    idx = random.randint(0, len(self.items) - 1)
                else:
                    logger.error(
                        f"Failed to load after {max_retries} attempts, returning zeros"
                    )
                    # Return dummy frame-level data with special flag for error detection
                    dummy_frames = 100
                    dummy_mel = torch.zeros(dummy_frames, self.mel.n_mels)
                    dummy_labels = torch.zeros(dummy_frames)
                    # Set first value to -1 to flag this as an error sample that should be handled specially
                    dummy_mel[0, 0] = -1
                    return dummy_mel, dummy_labels

    def _process_audio_file(self, path, clip_label, label_path):
        """Process an audio file to generate mel spectrogram and labels."""
        # Load and process audio
        logger.debug(f"Processing audio file: {path}")
        wav, sr = torchaudio.load(path)  # (1, T)
        if sr != self.sample_rate:
            logger.debug(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.clamp(-1, 1)
        mel = self.db(self.mel(wav)).squeeze(0).transpose(0, 1)  # (T,n_mels)

        if (
            "train" in str(self.manifest) and random.random() > 0.3
        ):  # 70% chance for training
            mel = self.spec_augment(mel)

        # Load frame-level labels if available, otherwise use clip label for all frames
        if label_path and label_path.exists():
            frame_labels = torch.from_numpy(np.load(label_path)).float()
            # Ensure length matches mel spectrogram frames
            if len(frame_labels) > mel.shape[0]:
                frame_labels = frame_labels[: mel.shape[0]]
            elif len(frame_labels) < mel.shape[0]:
                padding = torch.zeros(
                    mel.shape[0] - len(frame_labels), dtype=torch.float32
                )
                frame_labels = torch.cat([frame_labels, padding])
        else:
            # If no frame labels, use clip label for all frames
            frame_labels = torch.full(
                (mel.shape[0],), clip_label, dtype=torch.float32
            )
            
        return mel, frame_labels

    def spec_augment(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to mel spectrogram for data augmentation."""
        # Make a copy to avoid modifying the original
        mel = mel_spectrogram.clone()

        # Time masking
        for i in range(random.randint(1, 2)):
            t = random.randint(0, min(self.time_mask_max, mel.shape[0] // 10))
            t0 = random.randint(0, mel.shape[0] - t)
            mel[t0 : t0 + t, :] = 0

        # Frequency masking
        for i in range(random.randint(1, 2)):
            f = random.randint(0, min(self.freq_mask_max, mel.shape[1] // 4))
            f0 = random.randint(0, mel.shape[1] - f)
            mel[:, f0 : f0 + f] = 0

        return mel


def collate_pad(batch, max_frames=DEFAULT_MAX_FRAMES):
    """Pad variable-length Mel sequences and their frame labels on time axis."""
    # Filter out error samples (those with first value set to -1)
    filtered_batch = []
    for x, y in batch:
        # Check if this is an error sample
        if x.shape[0] > 0 and x[0, 0] == -1:
            logger.warning("Filtered out error sample in collate_fn")
            continue
        filtered_batch.append((x, y))
    
    # If all samples were filtered out, return a minimal dummy batch
    if len(filtered_batch) == 0:
        logger.error("All samples in batch were invalid!")
        # Return minimal batch with clear indication it's invalid
        # Using 1 frame and n_mels from original batch
        n_mels = batch[0][0].shape[1] if batch else DEFAULT_N_MELS
        dummy_x = torch.zeros(1, 1, n_mels)  # Batch size 1, 1 frame
        dummy_y = torch.zeros(1, 1)
        dummy_mask = torch.zeros(1, 1, dtype=torch.bool)
        return dummy_x, dummy_y, dummy_mask
    
    # Process the filtered batch
    xs, ys = zip(*filtered_batch)
    n_mels = xs[0].shape[1]
    longest = max(x.shape[0] for x in xs)
    T = min(longest, max_frames)  # Limit max sequence length

    logger.debug(
        f"Collating batch: longest seq={longest}, using T={T}, n_mels={n_mels}"
    )

    # Padded mel spectrograms
    out_x = torch.zeros(len(xs), T, n_mels)
    # Padded frame labels
    out_y = torch.zeros(len(ys), T)
    # Mask to identify valid (non-padded) frames for each sample
    mask = torch.zeros(len(xs), T, dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        frames = min(x.shape[0], T)
        out_x[i, :frames] = x[:frames]
        out_y[i, :frames] = y[:frames]
        mask[i, :frames] = True

    return out_x, out_y, mask


class VADDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for VAD task."""
    
    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        max_frames: int = DEFAULT_MAX_FRAMES,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_frames = max_frames
        logger.info(
            f"Initialized VADDataModule with batch_size={batch_size}, num_workers={num_workers}"
        )

    def collate_fn(self, batch):
        """Pad variable-length Mel sequences and frame labels."""
        return collate_pad(batch, self.max_frames)

    def train_dataloader(self):
        logger.info("Creating training dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        logger.info("Creating validation dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )