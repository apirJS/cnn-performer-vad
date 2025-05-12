#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  train.py All‑in‑one data‑prep + MEL‑spectrogram VAD training
# ───────────────────────────────────────────────────────────────────────
import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
from typing import List, Tuple, Optional

import numpy as np
import torch, torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa, soundfile as sf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional import auroc, f1_score
from torchmetrics import F1Score, AUROC
from pytorch_lightning.callbacks import EarlyStopping

import os
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Install dependencies if missing
try:
    from performer_pytorch import Performer

    logger.info("Performer library already installed")
except ImportError:
    logger.warning("Performer library not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "performer-pytorch"])
    from performer_pytorch import Performer

    logger.info("Successfully installed performer-pytorch")


# ───────────────────────────────────────────────────────────────────────
# SECTION 0 : HELPER UTILITIES
# ───────────────────────────────────────────────────────────────────────
def run_cmd(cmd: str):
    """Run shell command and exit on error (simple wrapper)."""
    logger.info(f"Executing command: {cmd}")
    print(f"➜  {cmd}")
    if subprocess.call(cmd, shell=True) != 0:
        logger.error(f"Command failed: {cmd}")
        sys.exit(f"❌ Command failed: {cmd}")
    logger.info("Command completed successfully")


def seed_everything(seed: int = 42):
    logger.info(f"Setting random seed to {seed} for reproducibility")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_best_precision():
    """Determine the best precision format for the available hardware."""
    logger.info("Using default 32-bit precision for stability")
    return 32  # Always use full precision on GTX 1650


# ───────────────────────────────────────────────────────────────────────
# SECTION 1 : DATA PREPARATION (with variable-length clips and augmentations)
# ───────────────────────────────────────────────────────────────────────
LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
]
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def download_file(url, dest_path):
    """
    Cross-platform file download function that works on Windows and Unix systems.
    """
    import os
    import requests
    from tqdm import tqdm

    logger.info(f"Downloading {url} to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as file, tqdm(
            desc=os.path.basename(url),
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        logger.info(f"Successfully downloaded {url}")
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_and_extract(url: str, dest: pathlib.Path):
    fname = dest / url.split("/")[-1]
    if not fname.exists():
        success = download_file(url, fname)
        if not success:
            logger.error(f"Failed to download {url}")
            sys.exit(f"❌ Failed to download {url}")
    else:
        logger.info(f"File already exists: {fname}, skipping download")

    mark = dest / f".extracted_{fname.name}"
    if not mark.exists():
        logger.info(f"Extracting archive: {fname.name}")
        print(f"📦 Extracting {fname.name}")
        import tarfile

        with tarfile.open(fname) as tar:
            tar.extractall(path=dest)
        mark.touch()
        logger.info(f"Extraction completed and marked with {mark}")
    else:
        logger.info(f"Archive already extracted (marker file exists): {mark}")


def make_pos_neg(
    libri_root: pathlib.Path,
    musan_root: pathlib.Path,
    out_pos: pathlib.Path,
    out_neg: pathlib.Path,
    n_pos: int,
    n_neg: int,
    snr_range=(0, 15),
    duration_range=(5, 10),
    sample_rate=16000,
    split_name="train"
):
    """Generate variable-length positives (speech+noise) & negatives (noise/music) with frame-level labels."""

    logger.info(
        f"Generating {n_pos} positive and {n_neg} negative samples for {split_name} split"
    )
    logger.info(f"Duration range: {duration_range}s, SNR range: {snr_range}dB")

    libri = sorted(libri_root.rglob("*.flac"))
    musan_noise = list(musan_root.joinpath("noise").rglob("*.wav"))
    musan_music = list(musan_root.joinpath("music").rglob("*.wav"))

    logger.info(
        f"Found {len(libri)} speech files, {len(musan_noise)} noise files, {len(musan_music)} music files"
    )
    assert libri and musan_noise and musan_music, "Missing source audio!"

    # Create directories for audio and frame-level labels
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)
    out_pos_labels = out_pos.parent / f"{split_name}_pos_labels"  # Updated path
    out_neg_labels = out_neg.parent / f"{split_name}_neg_labels"  # Updated path
    out_pos_labels.mkdir(parents=True, exist_ok=True)
    out_neg_labels.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Output directories created: {out_pos}, {out_neg}, {out_pos_labels}, {out_neg_labels}"
    )

    # Constants for frame-level calculations
    hop_length = 160  # Match your spectrogram hop_length
    win_length = 400  # Match your spectrogram win_length

    # → POSITIVE SAMPLES (speech + noise)
    logger.info(
        "Generating positive samples (speech + noise) with frame-level labels..."
    )
    start_time = time.time()
    for idx, sp_path in enumerate(libri[:n_pos]):
        if idx % 100 == 0:
            logger.info(f"Generating positive sample {idx}/{n_pos}")

        # Random duration between duration_range
        duration = random.uniform(*duration_range)

        # Load original speech and keep a copy for VAD labels
        original_speech, sr = librosa.load(sp_path, sr=sample_rate)
        original_speech = librosa.util.fix_length(
            original_speech, size=int(duration * sr)
        )

        # Store a copy for VAD label generation (before any augmentation)
        clean_speech = original_speech.copy()

        # Apply augmentation to the speech signal (but not to the reference used for labels)
        speech = original_speech.copy()
        if random.random() > 0.5:
            speech = librosa.effects.time_stretch(speech, rate=random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            speech = librosa.effects.pitch_shift(
                speech, sr=sr, n_steps=random.uniform(-2, 2)
            )
        speech = librosa.util.fix_length(speech, size=int(duration * sr))

        # Create the noisy mixture
        noise_path = random.choice(musan_noise)
        noise, _ = librosa.load(noise_path, sr=sr, mono=True, duration=duration)
        noise = librosa.util.fix_length(noise, size=len(speech))
        snr_db = random.uniform(*snr_range)
        alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(noise) + 1e-7)
        mix = speech + alpha * noise

        # Generate frame-level VAD labels from the original clean speech
        # Number of frames when using hop_length for striding
        n_frames = 1 + (len(clean_speech) - win_length) // hop_length

        # Calculate energy of each frame
        frame_energies = librosa.feature.rms(
            y=clean_speech, frame_length=win_length, hop_length=hop_length
        )[0]

        # Sort energies and use a better threshold approach
        sorted_energies = np.sort(frame_energies)
        # 30th percentile works better than 40th
        energy_threshold = sorted_energies[int(len(sorted_energies) * 0.3)]
        vad_labels = (frame_energies > energy_threshold).astype(np.float32)

        # Smooth labels with longer minimum segments
        min_speech_frames = 3  # Increased from 2 (about 50ms at 16kHz)
        for i in range(len(vad_labels) - min_speech_frames + 1):
            if 0 < sum(vad_labels[i : i + min_speech_frames]) < min_speech_frames:
                vad_labels[i : i + min_speech_frames] = 0

        # Save audio and frame-level labels
        sf.write(out_pos / f"pos_{idx:06}.wav", mix, sr)
        np.save(out_pos_labels / f"pos_{idx:06}_labels.npy", vad_labels)

    pos_time = time.time() - start_time
    logger.info(
        f"Completed {n_pos} positive samples in {pos_time:.2f}s ({pos_time/n_pos:.3f}s per sample)"
    )

    # → NEGATIVE SAMPLES (noise + music)
    logger.info(
        "Generating negative samples (noise + music) with frame-level labels..."
    )
    start_time = time.time()
    pool = musan_noise + musan_music
    for idx in range(n_neg):
        if idx % 100 == 0:
            logger.info(f"Generating negative sample {idx}/{n_neg}")

        duration = random.uniform(*duration_range)
        n1, n2 = random.sample(pool, 2)
        a, sr = librosa.load(n1, sr=sample_rate, mono=True, duration=duration)
        b, _ = librosa.load(n2, sr=sr, mono=True, duration=duration)
        mix = librosa.util.fix_length(
            a, size=int(duration * sr)
        ) + 0.5 * librosa.util.fix_length(b, size=int(duration * sr))

        # For negative samples, all frames are non-speech (0)
        n_frames = 1 + (len(mix) - win_length) // hop_length
        vad_labels = np.zeros(n_frames, dtype=np.float32)

        sf.write(out_neg / f"neg_{idx:06}.wav", mix, sr)
        np.save(out_neg_labels / f"neg_{idx:06}_labels.npy", vad_labels)

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {n_neg} negative samples in {neg_time:.2f}s ({neg_time/n_neg:.3f}s per sample)"
    )


def write_manifests(prep_dir: pathlib.Path, split=0.9):
    """Create train/val manifests with paths to audio and frame-level labels."""
    logger.info(f"Creating manifest files with train/val split of {split}")

    pos_clips = list(prep_dir.joinpath("pos").glob("*.wav"))
    neg_clips = list(prep_dir.joinpath("neg").glob("*.wav"))

    logger.info(
        f"Found {len(pos_clips)} positive and {len(neg_clips)} negative samples"
    )

    random.shuffle(pos_clips)
    random.shuffle(neg_clips)

    train_pos = pos_clips[: int(len(pos_clips) * split)]
    val_pos = pos_clips[int(len(pos_clips) * split) :]
    train_neg = neg_clips[: int(len(neg_clips) * split)]
    val_neg = neg_clips[int(len(neg_clips) * split) :]

    train, val = train_pos + train_neg, val_pos + val_neg
    random.shuffle(train)
    random.shuffle(val)

    logger.info(
        f"Train: {len(train)} samples ({len(train_pos)} positive, {len(train_neg)} negative)"
    )
    logger.info(
        f"Val: {len(val)} samples ({len(val_pos)} positive, {len(val_neg)} negative)"
    )

    for sub, name in [(train, "manifest_train.csv"), (val, "manifest_val.csv")]:
        manifest_path = prep_dir.parent / name
        logger.info(f"Writing manifest: {manifest_path}")
        with open(manifest_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "frame_labels"])
            for p in sub:
                is_speech = 1 if "pos_" in p.name else 0
                # Path to corresponding frame labels
                label_dir = "pos_labels" if is_speech else "neg_labels"
                frame_label_path = prep_dir.parent / label_dir / f"{p.stem}_labels.npy"
                w.writerow([p, is_speech, frame_label_path])
        logger.info(f"Created manifest {name} with {len(sub)} entries")

def create_manifest(prep_dir: pathlib.Path, manifest_path: pathlib.Path, split_name: str):
    """Create manifest with paths to audio and frame-level labels for specific split."""
    logger.info(f"Creating {split_name} manifest file")

    pos_clips = list(prep_dir.joinpath("pos").glob("*.wav"))
    neg_clips = list(prep_dir.joinpath("neg").glob("*.wav"))

    logger.info(
        f"Found {len(pos_clips)} positive and {len(neg_clips)} negative samples for {split_name}"
    )

    # Combine and shuffle samples
    all_clips = pos_clips + neg_clips
    random.shuffle(all_clips)

    logger.info(f"Writing manifest: {manifest_path}")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "frame_labels"])
        for p in all_clips:
            is_speech = 1 if "pos_" in p.name else 0
            # Path to corresponding frame labels
            label_dir = f"{split_name}_pos_labels" if is_speech else f"{split_name}_neg_labels"
            frame_label_path = prep_dir.parent / label_dir / f"{p.stem}_labels.npy"
            w.writerow([p, is_speech, frame_label_path])

    logger.info(f"Created {split_name} manifest with {len(all_clips)} entries")
    return manifest_path

def prepare_data(args):
    """Download, extract, and prepare audio data for VAD training with state tracking."""
    logger.info("Starting data preparation process")
    root = pathlib.Path(args.vad_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using root directory: {root}")

    # Track state in a JSON file
    state_file = root / "preparation_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            logger.info(f"Found existing preparation state: {state}")
        except Exception as e:
            logger.error(f"Error loading state file: {e}")

    # Early check if data directories already exist
    libri_clean_dir = root / "LibriSpeech" / "train-clean-100"
    musan_dir = root / "musan"

    if libri_clean_dir.exists() and musan_dir.exists():
        logger.info("LibriSpeech and MUSAN directories already exist")
        # Ensure state reflects that downloads are complete
        if not state.get("downloads_complete", False):
            logger.info("Updating state file to reflect existing data")
            state["downloads_complete"] = True
            with open(state_file, "w") as f:
                json.dump(state, f)
        logger.info("Skipping download step")
    # Download data if not already done
    elif not state.get("downloads_complete", False):
        logger.info("Beginning download of LibriSpeech and MUSAN datasets")
        for url in LIBRISPEECH_URLS:
            logger.info(f"Processing LibriSpeech URL: {url}")
            download_and_extract(url, root)
        logger.info(f"Processing MUSAN URL: {MUSAN_URL}")
        download_and_extract(MUSAN_URL, root)
        state["downloads_complete"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("All downloads completed successfully")
    else:
        # Verify files actually exist even if state claims downloads are complete
        if not libri_clean_dir.exists() or not musan_dir.exists():
            logger.warning(
                "State claims downloads complete, but directories are missing!"
            )
            logger.info("Resetting state and starting downloads again")
            state["downloads_complete"] = False

            # Download missing directories
            if not libri_clean_dir.exists():
                logger.info("LibriSpeech train-clean-100 missing, downloading...")
                for url in LIBRISPEECH_URLS:
                    if "train-clean-100" in url:
                        download_and_extract(url, root)

            if not musan_dir.exists():
                logger.info("MUSAN directory missing, downloading...")
                download_and_extract(MUSAN_URL, root)

            state["downloads_complete"] = True
            with open(state_file, "w") as f:
                json.dump(state, f)
            logger.info("Missing files downloaded successfully")
        else:
            logger.info("Downloads already completed and files exist, skipping")

    # Generate positive/negative samples if not done
    if not state.get("samples_generated", False):
        logger.info("Generating positive and negative audio samples")
        libri_root = root / "LibriSpeech" / "train-clean-100"
        musan_root = root / "musan"
        prep = root / "prepared"
        
        # Create train/val splits - updated paths
        train_pos, train_neg = prep / "train" / "pos", prep / "train" / "neg"
        val_pos, val_neg = prep / "val" / "pos", prep / "val" / "neg"
        
        # Calculate split sizes (90% train, 10% val)
        train_pos_count = int(args.n_pos * 0.9)
        train_neg_count = int(args.n_neg * 0.9)
        val_pos_count = args.n_pos - train_pos_count
        val_neg_count = args.n_neg - train_neg_count
        
        logger.info(f"Creating training set with {train_pos_count} positive and {train_neg_count} negative samples")
        make_pos_neg(
            libri_root,
            musan_root,
            train_pos,
            train_neg,
            train_pos_count,
            train_neg_count,
            duration_range=args.duration_range,
            sample_rate=args.sample_rate,
            split_name="train",  # Add split name
        )
        
        logger.info(f"Creating validation set with {val_pos_count} positive and {val_neg_count} negative samples")
        make_pos_neg(
            libri_root,
            musan_root,
            val_pos,
            val_neg,
            val_pos_count,
            val_neg_count,
            duration_range=args.duration_range,
            sample_rate=args.sample_rate,
            split_name="val",  # Add split name
        )
        
        state["samples_generated"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("Sample generation completed")
    else:
        logger.info("Samples already generated, skipping")

    # Create manifests directly rather than using write_manifests
    if not state.get("manifests_created", False):
        logger.info("Creating train/val manifests")
        create_manifest(root / "prepared" / "train", root / "manifest_train.csv", "train")
        create_manifest(root / "prepared" / "val", root / "manifest_val.csv", "val")
        state["manifests_created"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("Manifest creation completed")
    else:
        logger.info("Manifests already created, skipping")

    logger.info("✅ Data preparation completed successfully")
    print("✅ Data ready")


# ───────────────────────────────────────────────────────────────────────
# SECTION 2 : DATASET (→ MEL‑SPECTROGRAM) & DATALOADER
# ───────────────────────────────────────────────────────────────────────
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
        n_mels=64,
        n_fft=400,
        hop=160,
        win=400,
        sample_rate=16000,
        cache_dir=None,
        time_mask_max=20,
        freq_mask_max=10,
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
        self.cache_dir = None

        # Create params file in cache directory for validation
        if cache_dir:
            self.cache_dir = pathlib.Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            params_file = self.cache_dir / "params.json"

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
                            self.cache_dir.parent / f"{self.cache_dir.name}_backup"
                        )
                        if backup_dir.exists():
                            shutil.rmtree(backup_dir)
                        self.cache_dir.rename(backup_dir)
                        self.cache_dir.mkdir(parents=True)
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

            logger.info(f"Caching mel spectrograms to {self.cache_dir}")

        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
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

                # Cache result if enabled
                if self.cache_dir:
                    logger.debug(f"Caching data for {path.stem}")
                    torch.save(mel, cache_path)
                    torch.save(frame_labels, label_cache_path)

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

    # Update spec_augment method:
    def spec_augment(self, mel_spectrogram):
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


def collate_pad(batch, max_frames=3000):
    """Pad variable-length Mel sequences and their frame labels on time axis."""
    xs, ys = zip(*batch)
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


# ───────────────────────────────────────────────────────────────────────
# SECTION 3 :  MODEL (Performer over Mel frames)
# ───────────────────────────────────────────────────────────────────────
class MelPerformer(nn.Module):
    """
    Model that performs frame-level VAD using Performer attention over mel spectrograms.
    """

    def __init__(self, n_mels=64, dim=256, layers=4, heads=4, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads

        logger.info(
            f"Initializing frame-level MelPerformer with n_mels={n_mels}, dim={dim}, "
            f"layers={layers}, heads={heads}, dim_head={dim_head}"
        )

        # Add convolutional layers to capture local patterns
        # Enhanced frontend with BatchNorm
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2),  # Add BatchNorm for stability
            nn.ReLU(),
            nn.Conv1d(dim // 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),  # Add BatchNorm
            nn.ReLU(),
        )

        # Initialize all weights properly
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.proj = nn.Linear(dim, dim)  # Changed input dimension to match conv output
        self.norm = nn.LayerNorm(dim)

        self.transformer = Performer(
            dim=dim,
            depth=layers,
            heads=heads,
            dim_head=dim // heads,
            causal=False,
            nb_features=512,  # Increased from 256
            feature_redraw_interval=500,
            generalized_attention=False,
            reversible=True,
            ff_chunks=1,
            use_scalenorm=False,
            use_rezero=False,
            ff_glu=True,
            ff_dropout=0.2,  # Increased dropout
            attn_dropout=0.2,  # Increased dropout
        )

        # Change classifier to frame-level prediction (no pooling)
        # More robust classifier
        self.clf = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
        )

        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def forward(self, x):  # x (B,T,n_mels)
        logger.debug(f"Forward pass with input shape: {x.shape}")
        # Process with convolutional layers (transpose for conv1d)
        x = x.transpose(1, 2)  # (B,n_mels,T)
        x = self.conv_layers(x)  # (B,dim,T)
        x = x.transpose(1, 2)  # (B,T,dim)

        x = self.proj(x)  # (B,T,dim)
        x = self.norm(x)
        x = self.transformer(x)  # (B,T,dim)
        # Apply classifier to each time step
        return self.clf(x).squeeze(-1)  # (B,T) - one prediction per frame


class VADLightning(pl.LightningModule):
    """
    Lightning wrapper for frame-level VAD model with BCE loss & comprehensive metrics logging.
    """

    def __init__(self, hp):
        super().__init__()
        logger.info(f"Initializing frame-level VADLightning with hyperparameters: {hp}")
        self.save_hyperparameters(hp)
        self.net = MelPerformer(hp.n_mels, hp.dim, hp.n_layers, hp.n_heads)

        # Use weighted BCE loss
        pos_weight = torch.tensor([hp.pos_weight])
        self.loss = FocalLoss(alpha=0.25, gamma=2.0)

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        logger.info("Frame-level VADLightning model initialized")

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, tag):
        x, y, mask = batch  # Now includes mask for valid frames
        logger.debug(
            f"{tag} step with batch shapes: x={x.shape}, y={y.shape}, mask={mask.shape}"
        )

        logits = self(x)  # (B,T) - per frame logits

        # Apply loss only on valid (non-padded) frames using the mask
        valid_logits = logits[mask]
        valid_targets = y[mask]

        loss = self.loss(valid_logits, valid_targets)
        preds = torch.sigmoid(valid_logits)
        acc = ((preds > 0.5) == valid_targets.bool()).float().mean()

        # Log metrics
        self.log(f"{tag}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True)

        # Use stateful metrics with valid frames only
        if tag == "train":
            self.train_f1(preds > 0.5, valid_targets.int())
            self.train_auroc(preds, valid_targets.int())
            self.log("train_f1", self.train_f1, on_epoch=True)
            self.log("train_auroc", self.train_auroc, on_epoch=True)
        else:  # validation
            self.val_f1(preds > 0.5, valid_targets.int())
            self.val_auroc(preds, valid_targets.int())
            self.log("val_f1", self.val_f1, on_epoch=True)
            self.log("val_auroc", self.val_auroc, on_epoch=True)

        logger.debug(f"{tag} metrics: loss={loss:.4f}, acc={acc:.4f}")
        return loss

    def training_step(self, b, _):
        logger.debug("Executing training step")
        return self._step(b, "train")

    def validation_step(self, b, _):
        logger.debug("Executing validation step")
        self._step(b, "val")

    def configure_optimizers(self):
        logger.info(f"Configuring optimizer with lr={self.hparams.lr}")
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )

        # OneCycle scheduler - better convergence
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,  # First 10% for warmup
                div_factor=25,  # Start with lr/25
                final_div_factor=1000,
                anneal_strategy="cos",
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": opt, "lr_scheduler": scheduler}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Apply focal loss formula
        pt = torch.exp(-BCE_loss)  # Probability of being correct
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


# ───────────────────────────────────────────────────────────────
# Custom DataModule to handle collate_fn properly
# ───────────────────────────────────────────────────────────────
class VADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        batch_size=4,
        num_workers=4,
        max_frames=2000,
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
        xs, ys = zip(*batch)
        n_mels = xs[0].shape[1]
        longest = max(x.shape[0] for x in xs)
        T = min(longest, self.max_frames)  # Limit max sequence length

        # Padded mel spectrograms
        out_x = torch.zeros(len(xs), T, n_mels)
        # Padded frame labels
        out_y = torch.zeros(len(ys), T)
        # Mask to identify valid (non-padded) frames
        mask = torch.zeros(len(xs), T, dtype=torch.bool)

        for i, (x, y) in enumerate(zip(xs, ys)):
            frames = min(x.shape[0], T)
            out_x[i, :frames] = x[:frames]
            out_y[i, :frames] = y[:frames]
            mask[i, :frames] = True

        return out_x, out_y, mask

    def train_dataloader(self):
        logger.info("Creating training dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,  # Use the method directly
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
            collate_fn=self.collate_fn,  # Use the method directly
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# ───────────────────────────────────────────────────────────────────────
# SECTION 4 : ARGPARSE & MAIN
# ───────────────────────────────────────────────────────────────────────
def build_parser():
    logger.info("Building argument parser")
    p = argparse.ArgumentParser("Mel‑spectrogram Performer VAD")

    # workspace / data prep
    p.add_argument(
        "--vad_root",
        default="datasets",
        help="Root directory for all datasets and outputs",
    )
    p.add_argument(
        "--prepare_data", action="store_true", help="Download and prepare training data"
    )
    p.add_argument(
        "--n_pos",
        type=int,
        default=20_000,
        help="Number of positive samples to generate",
    )
    p.add_argument(
        "--n_neg",
        type=int,
        default=20_000,
        help="Number of negative samples to generate",
    )
    p.add_argument(
        "--duration_range",
        type=float,
        nargs=2,
        default=[5, 10],
        help="Min and max duration in seconds for audio clips",
    )
    p.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio processing",
    )
    p.add_argument(
        "--time_mask_max",
        type=int,
        default=20,
        help="Maximum time mask length for SpecAugment",
    )
    p.add_argument(
        "--freq_mask_max",
        type=int,
        default=10,
        help="Maximum frequency mask length for SpecAugment",
    )

    # manifests
    p.add_argument("--train_manifest", help="Path to training manifest CSV")
    p.add_argument("--val_manifest", help="Path to validation manifest CSV")
    p.add_argument("--test_manifest", help="Path to test manifest for final evaluation")
    p.add_argument(
        "--test_after_training",
        action="store_true",
        help="Run evaluation on test set after training",
    )

    # mel params
    p.add_argument(
        "--n_mels", type=int, default=64, help="Number of mel bands in spectrogram"
    )
    p.add_argument("--n_fft", type=int, default=400, help="FFT size for spectrogram")
    p.add_argument("--hop", type=int, default=160, help="Hop length for spectrogram")

    # caching
    p.add_argument(
        "--use_mel_cache",
        action="store_true",
        help="Cache Mel spectrograms to disk for faster training",
    )
    p.add_argument(
        "--mel_cache_dir",
        default="mel_cache",
        help="Directory to store cached Mel spectrograms",
    )

    # model
    p.add_argument(
        "--dim", type=int, default=256, help="Transformer embedding dimension"
    )
    p.add_argument(
        "--n_layers", type=int, default=4, help="Number of transformer layers"
    )
    p.add_argument("--n_heads", type=int, default=4, help="Number of attention heads")
    p.add_argument(
        "--max_frames",
        type=int,
        default=2000,
        help="Maximum number of frames to use in a sequence",
    )
    p.add_argument(
        "--export_model",
        action="store_true",
        help="Export the trained model for inference",
    )
    p.add_argument(
        "--export_path", default="vad_model.pt", help="Path to save the exported model"
    )

    # optim
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    p.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    p.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count,
        help=f"Number of data loading workers (default: {cpu_count}, based on CPU count)",
    )
    p.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping to prevent exploding gradients",
    )
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=0.5,
        help="Number of epochs for learning rate warmup",
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=2.0,
        help="Positive class weight for BCE loss to handle imbalance",
    )
    p.add_argument(
        "--auto_batch_size",
        action="store_true",
        help="Automatically determine optimal batch size based on GPU memory",
    )

    # checkpoint / misc
    p.add_argument("--ckpt_path", help="Path to checkpoint for resuming training")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before optimizer step",
    )
    p.add_argument(
        "--log_dir", default="lightning_logs", help="Directory for TensorBoard logs"
    )
    logger.info("Argument parser configured")
    return p


# Add this function before main()
def export_model(model_path, output_path):
    """Export a trained model for inference."""
    logger.info(f"Loading checkpoint from {model_path}")
    model = VADLightning.load_from_checkpoint(model_path)
    model.eval()

    # Extract the core network without the Lightning wrapper
    net = model.net

    logger.info(f"Creating TorchScript model")
    # Script the model for portable deployment
    scripted_model = torch.jit.script(net)

    logger.info(f"Saving model to {output_path}")
    torch.jit.save(scripted_model, output_path)
    logger.info(f"Model exported successfully")
    print(f"✅ Model exported to {output_path}")


# Add this function before main():
def estimate_optimal_batch_size(model, n_mels, max_frames, gpu_memory_gb=None):
    """Estimate optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        logger.info("No GPU available, using default batch size")
        return 4  # Default batch size for CPU

    # Get available GPU memory if not provided
    if gpu_memory_gb is None:
        gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory_bytes / (1024**3)

    # Estimate memory requirements
    n_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = n_params * 4 / (1024**3)  # 4 bytes per parameter

    # Memory for single sample with gradients (roughly 3x forward pass)
    sample_memory_gb = 3 * max_frames * n_mels * 4 / (1024**3)

    # Reserve memory for PyTorch's own usage and other overhead (30%)
    available_memory_gb = gpu_memory_gb * 0.7 - param_memory_gb

    # Calculate maximum batch size based on available memory
    max_batch_size = max(1, int(available_memory_gb / sample_memory_gb))

    # Cap at reasonable limits
    optimal_batch_size = min(64, max(1, max_batch_size))

    logger.info(f"Estimated GPU memory: {gpu_memory_gb:.2f} GB")
    logger.info(f"Parameter memory: {param_memory_gb:.2f} GB")
    logger.info(f"Sample memory: {sample_memory_gb:.2f} GB per sample")
    logger.info(f"Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


def main():
    logger.info("Starting VAD training script")
    args = build_parser().parse_args()
    logger.info(f"Parsed arguments: {args}")

    seed_everything(args.seed)

    # Stage A: optionally build data
    if args.prepare_data:
        logger.info("Preparing data as requested")
        prepare_data(args)
        args.train_manifest = str(pathlib.Path(args.vad_root) / "manifest_train.csv")
        args.val_manifest = str(pathlib.Path(args.vad_root) / "manifest_val.csv")
        logger.info(
            f"Set manifest paths: train={args.train_manifest}, val={args.val_manifest}"
        )

    # Verify manifests exist
    for m in [args.train_manifest, args.val_manifest]:
        if not m or not pathlib.Path(m).exists():
            logger.error(f"Manifest missing: {m}")
            sys.exit("❌ Manifest missing. Use --prepare_data first.")
    logger.info("Manifest files verified")

    # Windows compatibility: Disable multiprocessing on Windows
    if sys.platform == "win32" and args.num_workers > 0:
        logger.info(
            "Running on Windows - disabling multiprocessing (num_workers=0) to avoid pickling issues"
        )
        args.num_workers = 0
        print("⚠️ Windows detected: Disabled multiprocessing for compatibility")

    # Create collate function with max_frames limit
    logger.info(f"Configuring collate function with max_frames={args.max_frames}")

    def collate_fn(batch):
        return collate_pad(batch, max_frames=args.max_frames)

    # DataModule via from_datasets helper
    logger.info("Creating dataset and datamodule")
    cache_dir = pathlib.Path(args.mel_cache_dir) if args.use_mel_cache else None
    if cache_dir:
        logger.info(f"Using mel spectrogram caching in {cache_dir}")

    logger.info("Initializing training dataset")
    train_dataset = CSVMelDataset(
        args.train_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    )

    logger.info("Initializing validation dataset")
    val_dataset = CSVMelDataset(
        args.val_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    )

    if args.auto_batch_size and torch.cuda.is_available():
        logger.info("Estimating optimal batch size based on GPU memory")
        # Create a temporary model instance for estimation
        temp_model = VADLightning(args)
        optimal_batch_size = estimate_optimal_batch_size(
            temp_model, args.n_mels, args.max_frames
        )
        logger.info(f"Using automatically determined batch size: {optimal_batch_size}")
        args.batch_size = optimal_batch_size
        print(f"📊 Auto-determined batch size: {optimal_batch_size}")

    logger.info(
        f"Creating custom datamodule with batch_size={args.batch_size}, num_workers={args.num_workers}"
    )
    dm = VADDataModule(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_frames=args.max_frames,
    )

    logger.info("Creating VAD model")
    model = VADLightning(args)

    logger.info("Configuring checkpointing and callbacks")
    cb_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename="{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}",
        auto_insert_metric_name=False,
    )
    cb_lr = LearningRateMonitor(logging_interval="epoch")
    cb_early_stop = EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=True
    )

    # Memory estimate
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    mem_est = n_params * 4 / (1024**2)  # Approx mem in MB for parameters
    batch_mem = (
        args.batch_size * args.max_frames * args.n_mels * 4 / (1024**2)
    )  # Approx mem for batch
    logger.info(
        f"Estimated GPU memory usage: {mem_est:.1f}MB (params) + {batch_mem:.1f}MB (batch)"
    )
    print(f"Model parameters: {n_params:,}")
    print(
        f"Estimated GPU memory usage: {mem_est:.1f}MB (params) + {batch_mem:.1f}MB (batch)"
    )

    logger.info("Configuring trainer")
    precision = get_best_precision()
    accelerator = "gpu" if args.gpus else "cpu"
    logger.info(f"Using {accelerator} with precision={precision}")

    if args.gpus > 1:
        strategy = "ddp"
        logger.info(f"Using DDP strategy for {args.gpus} GPUs")
    else:
        strategy = "auto"  # Change from None to "auto"
        logger.info(f"Using single device strategy")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.gpus or 1,
        max_epochs=args.max_epochs,
        precision=precision,
        callbacks=[cb_ckpt, cb_lr, cb_early_stop],
        log_every_n_steps=25,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        logger=pl.loggers.TensorBoardLogger(save_dir=args.log_dir),
    )

    logger.info("Starting training")
    # Pass the checkpoint path to the fit method instead
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)
    logger.info("Training completed")

    if args.test_after_training and args.test_manifest:
        logger.info(f"Running final evaluation on test set: {args.test_manifest}")

        # Load test dataset
        test_dataset = CSVMelDataset(
            args.test_manifest,
            args.n_mels,
            args.n_fft,
            args.hop,
            sample_rate=args.sample_rate,
            cache_dir=cache_dir,
            time_mask_max=args.time_mask_max,
            freq_mask_max=args.freq_mask_max,
        )

        # Create test dataloader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dm.collate_fn,
            pin_memory=True,
        )

        # Use best checkpoint for testing
        if hasattr(cb_ckpt, "best_model_path") and cb_ckpt.best_model_path:
            logger.info(f"Using best checkpoint: {cb_ckpt.best_model_path}")
            best_model = VADLightning.load_from_checkpoint(cb_ckpt.best_model_path)

            # Run test
            logger.info("Testing model performance...")
            test_results = trainer.test(best_model, test_dataloader)[0]
            logger.info(f"Test results: {test_results}")
            print(f"✅ Test results: {test_results}")
        else:
            logger.warning("No best checkpoint found, skipping test evaluation")

    # Add at the end of main() function, after training
    if (
        args.export_model
        and hasattr(cb_ckpt, "best_model_path")
        and cb_ckpt.best_model_path
    ):
        logger.info(f"Exporting best model from {cb_ckpt.best_model_path}")
        export_model(cb_ckpt.best_model_path, args.export_path)


if __name__ == "__main__":
    main()
