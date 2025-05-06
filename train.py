#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  train.py All‑in‑one data‑prep + MEL‑spectrogram VAD training
# ───────────────────────────────────────────────────────────────────────
import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
from typing import List, Tuple, Optional

import numpy as np
import torch, torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

import librosa, soundfile as sf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchmetrics.functional import auroc, f1_score
from torchmetrics import F1Score, AUROC

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
    logger.info("Determining optimal precision for current hardware")
    if torch.cuda.is_bf16_supported():
        logger.info("BF16 precision supported, using bf16-mixed")
        return "bf16-mixed"
    elif torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        logger.info("FP16 precision supported, using 16-bit")
        return 16
    logger.info("Using default 32-bit precision")
    return 32  # Default to full precision


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
):
    """Generate variable-length positives (speech+noise) & negatives (noise/music)."""

    logger.info(f"Generating {n_pos} positive and {n_neg} negative samples")
    logger.info(f"Duration range: {duration_range}s, SNR range: {snr_range}dB")

    libri = sorted(libri_root.rglob("*.flac"))
    musan_noise = list(musan_root.joinpath("noise").rglob("*.wav"))
    musan_music = list(musan_root.joinpath("music").rglob("*.wav"))

    logger.info(
        f"Found {len(libri)} speech files, {len(musan_noise)} noise files, {len(musan_music)} music files"
    )
    assert libri and musan_noise and musan_music, "Missing source audio!"

    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directories created: {out_pos}, {out_neg}")

    # → POSITIVE
    logger.info("Generating positive samples (speech + noise)...")
    start_time = time.time()
    for idx, sp_path in enumerate(libri[:n_pos]):
        if idx % 100 == 0:
            logger.info(f"Generating positive sample {idx}/{n_pos}")

        # Random duration between duration_range
        duration = random.uniform(*duration_range)
        speech, sr = librosa.load(sp_path, sr=sample_rate)
        speech = librosa.util.fix_length(speech, size=int(duration * sr))

        # Add time stretching, pitch shifting
        if random.random() > 0.5:
            speech = librosa.effects.time_stretch(speech, rate=random.uniform(0.9, 1.1))
        if random.random() > 0.5:
            speech = librosa.effects.pitch_shift(
                speech, sr=sr, n_steps=random.uniform(-2, 2)
            )

        speech = librosa.util.fix_length(speech, size=int(duration * sr))

        noise_path = random.choice(musan_noise)
        noise, _ = librosa.load(noise_path, sr=sr, mono=True, duration=duration)
        noise = librosa.util.fix_length(noise, size=len(speech))
        snr_db = random.uniform(*snr_range)
        alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(noise) + 1e-7)
        mix = speech + alpha * noise
        sf.write(out_pos / f"pos_{idx:06}.wav", mix, sr)

    pos_time = time.time() - start_time
    logger.info(
        f"Completed {n_pos} positive samples in {pos_time:.2f}s ({pos_time/n_pos:.3f}s per sample)"
    )

    # → NEGATIVE
    logger.info("Generating negative samples (noise + music)...")
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
        sf.write(out_neg / f"neg_{idx:06}.wav", mix, sr)

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {n_neg} negative samples in {neg_time:.2f}s ({neg_time/n_neg:.3f}s per sample)"
    )


def write_manifests(prep_dir: pathlib.Path, split=0.9):
    """Create train/val manifests with stratified splits for class balance."""
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
            w.writerow(["path", "label"])
            for p in sub:
                w.writerow([p, 1 if "pos_" in p.name else 0])
        logger.info(f"Created manifest {name} with {len(sub)} entries")


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
        prep_pos, prep_neg = prep / "pos", prep / "neg"
        make_pos_neg(
            libri_root,
            musan_root,
            prep_pos,
            prep_neg,
            args.n_pos,
            args.n_neg,
            duration_range=args.duration_range,
            sample_rate=args.sample_rate,
        )
        state["samples_generated"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("Sample generation completed")
    else:
        logger.info("Samples already generated, skipping")

    # Create manifests if not done
    if not state.get("manifests_created", False):
        logger.info("Creating train/val manifests")
        write_manifests(root / "prepared")
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
    On‑the‑fly converts WAV → log‑Mel spectrogram.
      • returns tensor (T, n_mels)
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
    ):
        logger.info(f"Initializing dataset from manifest: {manifest}")
        self.items = [
            (r["path"], int(r["label"])) for r in csv.DictReader(open(manifest))
        ]
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
        if cache_dir:
            self.cache_dir = pathlib.Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Caching mel spectrograms to {self.cache_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                path, lab = self.items[idx]
                path = pathlib.Path(path)

                # Try loading from cache first if enabled
                if self.cache_dir:
                    cache_path = self.cache_dir / f"{path.stem}.pt"
                    if cache_path.exists():
                        logger.debug(f"Loading cached mel spectrogram for {path.stem}")
                        mel = torch.load(cache_path)
                        return mel, torch.tensor(lab, dtype=torch.float32)

                # Load and process audio
                logger.debug(f"Processing audio file: {path}")
                wav, sr = torchaudio.load(path)  # (1, T)
                if sr != self.sample_rate:
                    logger.debug(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
                wav = wav.clamp(-1, 1)
                mel = self.db(self.mel(wav)).squeeze(0).transpose(0, 1)  # (T,n_mels)

                # Cache result if enabled
                if self.cache_dir:
                    logger.debug(f"Caching mel spectrogram for {path.stem}")
                    torch.save(mel, cache_path)

                return mel, torch.tensor(lab, dtype=torch.float32)
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
                    return torch.zeros(100, self.mel.n_mels), torch.tensor(0.0)


def collate_pad(batch, max_frames=3000):
    """Pad variable‑length Mel sequences on time axis with a maximum length constraint."""
    xs, ys = zip(*batch)
    n_mels = xs[0].shape[1]
    longest = max(x.shape[0] for x in xs)
    T = min(longest, max_frames)  # Limit max sequence length
    logger.debug(
        f"Collating batch: longest seq={longest}, using T={T}, n_mels={n_mels}"
    )
    out = torch.zeros(len(xs), T, n_mels)
    for i, x in enumerate(xs):
        out[i, : min(x.shape[0], T)] = x[:T]
    return out, torch.stack(ys)


# ───────────────────────────────────────────────────────────────────────
# SECTION 3 :  MODEL (Performer over Mel frames)
# ───────────────────────────────────────────────────────────────────────
class MelPerformer(nn.Module):
    """
    Model that performs VAD using Performer attention over mel spectrograms.
    """

    def __init__(self, n_mels=64, dim=256, layers=4, heads=4, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads

        logger.info(
            f"Initializing MelPerformer with n_mels={n_mels}, dim={dim}, "
            f"layers={layers}, heads={heads}, dim_head={dim_head}"
        )
        self.proj = nn.Linear(n_mels, dim)
        self.transformer = Performer(
            dim=dim,
            depth=layers,
            heads=heads,
            dim_head=dim // heads,
            causal=False,
            nb_features=256,  # Number of random features, can improve performance
            feature_redraw_interval=1000,  # How often to redraw random features during training
            generalized_attention=False,  # Use ReLU kernel instead of softmax approximation
            # kernel_fn=torch.nn.ReLU(),  # Only used if generalized_attention=True
            reversible=False,  # Whether to use reversible layers for memory efficiency
            ff_chunks=1,  # Number of chunks for feed-forward layer computation
            use_scalenorm=False,  # Alternative to LayerNorm
            use_rezero=False,  # Use ReZero initialization
            ff_glu=True,  # Use GLU variant for feedforward
            ff_dropout=0.1,  # Feedforward dropout
            attn_dropout=0.1,  # Attention dropout
        )
        self.clf = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(dim, 1)
        )
        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def forward(self, x):  # x (B,T,n_mels)
        logger.debug(f"Forward pass with input shape: {x.shape}")
        x = self.proj(x)  # (B,T,dim)
        x = self.transformer(x)  # (B,T,dim)
        x = x.transpose(1, 2)  # (B,dim,T)
        return self.clf(x).squeeze(1)


class VADLightning(pl.LightningModule):
    """
    Lightning wrapper for VAD model with BCE loss & comprehensive metrics logging.
    """

    def __init__(self, hp):
        super().__init__()
        logger.info(f"Initializing VADLightning with hyperparameters: {hp}")
        self.save_hyperparameters(hp)
        self.net = MelPerformer(hp.n_mels, hp.dim, hp.n_layers, hp.n_heads)
        self.loss = nn.BCEWithLogitsLoss()
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary") 
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        logger.info("VADLightning model initialized")

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, tag):
        x, y = batch
        logger.debug(f"{tag} step with batch shapes: x={x.shape}, y={y.shape}")
        logit = self(x)
        loss = self.loss(logit, y)
        preds = torch.sigmoid(logit)
        acc = ((preds > 0.5) == y.bool()).float().mean()

        # Log metrics
        self.log(f"{tag}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True)

        # Use stateful metrics instead of functional API
        if tag == "train":
            self.train_f1(preds > 0.5, y.int())
            self.train_auroc(preds, y.int())
            self.log("train_f1", self.train_f1, on_epoch=True)
            self.log("train_auroc", self.train_auroc, on_epoch=True)
        else:  # validation
            self.val_f1(preds > 0.5, y.int())
            self.val_auroc(preds, y.int())
            self.log("val_f1", self.val_f1, on_epoch=True)
            self.log("val_auroc", self.val_auroc, on_epoch=True)

        logger.debug(
            f"{tag} metrics: loss={loss:.4f}, acc={acc:.4f}"
        )
        return loss

    def training_step(self, b, _):
        logger.debug("Executing training step")
        return self._step(b, "train")

    def validation_step(self, b, _):
        logger.debug("Executing validation step")
        self._step(b, "val")

    def configure_optimizers(self):
        logger.info(
            f"Configuring optimizer with lr={self.hparams.lr}, weight_decay=1e-4"
        )
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
        logger.info("Optimizer and learning rate scheduler configured")
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


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

    # Replace get_collate_fn with a direct method that can be pickled
    def collate_fn(self, batch):
        """Pad variable‑length Mel sequences on time axis with a maximum length constraint."""
        xs, ys = zip(*batch)
        n_mels = xs[0].shape[1]
        longest = max(x.shape[0] for x in xs)
        T = min(longest, self.max_frames)  # Limit max sequence length
        logger.debug(
            f"Collating batch: longest seq={longest}, using T={T}, n_mels={n_mels}"
        )
        out = torch.zeros(len(xs), T, n_mels)
        for i, x in enumerate(xs):
            out[i, : min(x.shape[0], T)] = x[:T]
        return out, torch.stack(ys)

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
        default="vad_corpus",
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
    # manifests
    p.add_argument("--train_manifest", help="Path to training manifest CSV")
    p.add_argument("--val_manifest", help="Path to validation manifest CSV")
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
    # optim
    p.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    p.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--max_epochs", type=int, default=10, help="Maximum training epochs")
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    # checkpoint / misc
    p.add_argument("--ckpt_path", help="Path to checkpoint for resuming training")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    logger.info("Argument parser configured")
    return p


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
    )

    logger.info("Initializing validation dataset")
    val_dataset = CSVMelDataset(
        args.val_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
    )

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

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.gpus or 1,
        max_epochs=args.max_epochs,
        precision=precision,
        callbacks=[cb_ckpt, cb_lr],
        log_every_n_steps=25,
        # Remove the resume_from_checkpoint parameter
    )

    logger.info("Starting training")
    # Pass the checkpoint path to the fit method instead
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)
    logger.info("Training completed")


if __name__ == "__main__":
    main()
