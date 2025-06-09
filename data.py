#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  data.py - Dataset and data handling for VAD
# ───────────────────────────────────────────────────────────────────────
import csv
import json
import logging
import pathlib
import random
from typing import Tuple, Optional

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
    DEFAULT_MAX_FRAMES,
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
                    backup_dir = cache_path.parent / f"{cache_path.name}_backup"
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
                mel, frame_labels = self._process_audio_file(
                    path, clip_label, label_path
                )

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
        try:
            wav, sr = torchaudio.load(path)
        except Exception as e:
            # Fallback to librosa if torchaudio fails
            logger.warning(f"Torchaudio failed, trying librosa: {e}")
            try:
                import librosa

                y, sr = librosa.load(str(path), sr=self.sample_rate)
                wav = torch.from_numpy(y).unsqueeze(0)
            except Exception as e2:
                raise RuntimeError(f"Both loading methods failed: {e2}")

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
            frame_labels = torch.full((mel.shape[0],), clip_label, dtype=torch.float32)

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
        # Return dummy batch with the original batch size and correct dimensions
        batch_size = len(batch)
        n_mels = batch[0][0].shape[1] if batch else DEFAULT_N_MELS
        dummy_x = torch.zeros(batch_size, max_frames, n_mels)
        dummy_y = torch.zeros(batch_size, max_frames)
        dummy_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
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


class BoundaryEnhancedDataset(CSVMelDataset):
    """Dataset with boundary-focused augmentation."""
    
    def __init__(self, *args, boundary_focus_prob=0.7, max_boundary_shift=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.boundary_focus_prob = boundary_focus_prob
        self.max_boundary_shift = max_boundary_shift
        logger.info(f"Initialized BoundaryEnhancedDataset with boundary_focus_prob={boundary_focus_prob}")
        
    def __getitem__(self, idx):
        path, clip_label, label_path = self.items[idx]
        
        # Process audio normally first
        mel, frame_labels = super()._process_audio_file(
            pathlib.Path(path), clip_label, pathlib.Path(label_path) if label_path else None
        )
        
        # Apply boundary-focused augmentation with high probability during training
        if "train" in str(self.manifest) and random.random() < self.boundary_focus_prob:
            mel, frame_labels = self.boundary_augment(mel, frame_labels)
            
        return mel, frame_labels
        
    def boundary_augment(self, mel, frame_labels):
        """Apply augmentation focused on speech boundaries."""
        # Find transitions in labels (speech boundaries)
        boundaries = []
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != frame_labels[i-1]:
                boundaries.append(i)
                
        if not boundaries:  # No boundaries found
            return mel, frame_labels
            
        # Apply several types of boundary-focused augmentations
        augment_type = random.randint(0, 3)
        
        if augment_type == 0:
            # 1. Boundary shifting: Slightly move boundaries to create variations
            mel, frame_labels = self.shift_boundaries(mel, frame_labels, boundaries)
        elif augment_type == 1:
            # 2. Boundary emphasis: Emphasize mel features at boundaries
            mel = self.emphasize_boundaries(mel, boundaries)
        elif augment_type == 2:
            # 3. Boundary noise: Add noise specifically around boundaries
            mel = self.add_boundary_noise(mel, boundaries)
        else:
            # 4. Create synthetic fast transitions by splicing segments
            if len(boundaries) >= 2 and random.random() < 0.3:
                mel, frame_labels = self.create_synthetic_transition(mel, frame_labels, boundaries)
                
        return mel, frame_labels
        
    def shift_boundaries(self, mel, labels, boundaries):
        """Slightly shift speech boundaries to create robust detection."""
        shifted_labels = labels.clone()
        
        for boundary in boundaries:
            # Skip boundaries too close to edges
            if boundary < self.max_boundary_shift or boundary >= len(labels) - self.max_boundary_shift:
                continue
                
            # Randomly shift by -2 to +2 frames
            shift = random.randint(-self.max_boundary_shift, self.max_boundary_shift)
            if shift == 0:
                continue
                
            new_boundary = boundary + shift
            # Update labels to reflect shifted boundary
            if labels[boundary-1] == 1:  # Speech to non-speech transition
                shifted_labels[boundary:new_boundary] = 0 if shift > 0 else 1
            else:  # Non-speech to speech transition
                shifted_labels[boundary:new_boundary] = 1 if shift > 0 else 0
                
        return mel, shifted_labels
        
    def emphasize_boundaries(self, mel, boundaries):
        """Emphasize mel features at boundaries to make them more distinct."""
        enhanced_mel = mel.clone()
        
        for boundary in boundaries:
            if boundary < 1 or boundary >= len(mel) - 1:
                continue
                
            # Enhance the contrast at boundaries (3 frames around boundary)
            start_idx = max(0, boundary - 1)
            end_idx = min(len(mel), boundary + 2)
            
            # Increase feature values slightly at boundaries
            boost_factor = random.uniform(1.05, 1.15)
            enhanced_mel[start_idx:end_idx, :] *= boost_factor
            
        return enhanced_mel
        
    def add_boundary_noise(self, mel, boundaries):
        """Add noise specifically around boundaries to improve robustness."""
        noisy_mel = mel.clone()
        
        for boundary in boundaries:
            if boundary < 2 or boundary >= len(mel) - 2:
                continue
                
            # Add noise in a small window around the boundary
            window_size = random.randint(2, 4)
            start_idx = max(0, boundary - window_size // 2)
            end_idx = min(len(mel), boundary + window_size // 2 + 1)
            
            # Add mild noise
            noise_level = random.uniform(0.05, 0.15)
            noise = torch.randn(end_idx - start_idx, mel.shape[1]) * noise_level
            noisy_mel[start_idx:end_idx, :] += noise
            
        return noisy_mel
        
    def create_synthetic_transition(self, mel, labels, boundaries):
        """Create synthetic fast transitions by splicing segments together."""
        if len(boundaries) < 2:
            return mel, labels
            
        # Choose two boundaries
        b1_idx = random.randint(0, len(boundaries)-2)
        b1 = boundaries[b1_idx]
        b2 = boundaries[b1_idx+1]
        
        # Ensure they're far enough apart
        if b2 - b1 < 10:
            return mel, labels
            
        # Create a new synthetic boundary by splicing
        splice_point = random.randint(b1 + 3, b2 - 3)
        
        # Create a rapid transition by duplicating frames from other boundary
        transition_length = random.randint(2, 4)
        
        # Copy frames from around b1 to splice_point
        for i in range(transition_length):
            if splice_point + i < len(mel) and b1 - transition_length + i >= 0:
                mel[splice_point + i] = mel[b1 - transition_length + i]
                
        # Update labels - create a more abrupt transition
        new_labels = labels.clone()
        # Assuming b1 transitions from 1→0 and b2 from 0→1
        if labels[b1 - 1] == 1:  # If b1 is speech→non-speech
            new_labels[splice_point:splice_point+transition_length] = 0  # Make it non-speech
        else:
            new_labels[splice_point:splice_point+transition_length] = 1  # Make it speech
            
        return mel, new_labels

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
