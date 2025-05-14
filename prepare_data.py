#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  prepare_data.py Data preparation for MEL-spectrogram VAD
# ───────────────────────────────────────────────────────────────────────
import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
from pathlib import Path
import numpy as np
import librosa
import scipy.signal as signal
from datasets import load_dataset, Audio
import soundfile as sf
from tqdm import tqdm
from torchaudio.datasets import MUSDB_HQ
import torchaudio
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from config import *
import torch.hub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# URL constants
TRAIN_LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-360.tar.gz"
VAL_LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
ESC_50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
VOCALSET_URL = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"
MUSDB18HQ_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
URBANSOUND8K_URL = "https://goo.gl/8hY5ER"

MAX_PER_LANG_TRAIN = 1000  # FLEURS gives ≈1000 train clips per language
MAX_PER_LANG_VAL = 400  # idem for dev
MAX_PER_LANG_TEST = 400


# ───────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ───────────────────────────────────────────────────────────────────────
def download_and_extract_urbansound8k(url: str, dest: pathlib.Path):
    """Download and extract the UrbanSound8K dataset if not already done."""
    fname = dest / "urbansound8k.tar.gz"
    if not fname.exists():
        success = download_file(url, fname)
        if not success:
            logger.error(f"Failed to download UrbanSound8K dataset")
            sys.exit(f"❌ Failed to download UrbanSound8K dataset")
    else:
        logger.info(f"UrbanSound8K archive already exists: {fname}, skipping download")

    mark = dest / f".extracted_urbansound8k.tar.gz"
    if not mark.exists():
        logger.info(f"Extracting UrbanSound8K archive")
        print(f"📦 Extracting urbansound8k.tar.gz")
        import tarfile

        try:
            with tarfile.open(fname) as tar:
                tar.extractall(path=dest)
            mark.touch()
            logger.info(f"UrbanSound8K extraction completed and marked with {mark}")
        except Exception as e:
            logger.error(f"Error extracting UrbanSound8K dataset: {e}")
            sys.exit(f"❌ Error extracting UrbanSound8K dataset: {e}")
    else:
        logger.info(f"UrbanSound8K archive already extracted (marker file exists): {mark}")


def process_urbansound8k(
    root_dir: pathlib.Path, 
    split_name: str = "train", 
    sample_rate: int = 16000
) -> list[pathlib.Path]:
    """
    Process UrbanSound8K dataset with train/val/test splitting based on folds.
    Excludes human-related sounds to avoid contaminating negative samples with speech.

    Args:
        root_dir: Root directory where dataset is extracted
        split_name: Which split to use ('train', 'val', or 'test')
        sample_rate: Target sample rate

    Returns:
        List of paths to noise audio files for the requested split
    """
    # Find the UrbanSound8K directory
    urbansound_dirs = list(root_dir.glob("*UrbanSound8K*"))
    if not urbansound_dirs:
        logger.warning(f"UrbanSound8K directory not found in {root_dir}")
        return []

    urbansound_dir = urbansound_dirs[0]
    logger.info(f"Found UrbanSound8K directory at {urbansound_dir}")

    # Find audio directory
    audio_dir = urbansound_dir / "audio"
    if not audio_dir.exists():
        audio_dirs = list(urbansound_dir.rglob("audio"))
        if audio_dirs:
            audio_dir = audio_dirs[0]
            logger.info(f"Found UrbanSound8K audio directory at {audio_dir}")
        else:
            logger.warning(f"UrbanSound8K audio directory not found at {urbansound_dir}")
            return []

    # Load metadata file for class information
    metadata_path = urbansound_dir / "metadata" / "UrbanSound8K.csv"
    if not metadata_path.exists():
        metadata_paths = list(urbansound_dir.rglob("*UrbanSound8K.csv"))
        if metadata_paths:
            metadata_path = metadata_paths[0]
        else:
            logger.warning(f"UrbanSound8K metadata file not found")
            return []

    # Read metadata to identify human-related sounds
    try:
        import pandas as pd
        metadata = pd.read_csv(metadata_path)
        
        # Define human-related classes to exclude (children_playing)
        human_class_ids = [2]  # class ID for "children_playing"
        
        # Map split name to fold numbers (10 folds total)
        if split_name == "train":
            target_folds = [1, 2, 3, 4, 5, 6]  # 60% for training
        elif split_name == "val":
            target_folds = [7, 8]  # 20% for validation
        else:  # test
            target_folds = [9, 10]  # 20% for testing
            
        # Filter metadata to get files for this split
        split_files = metadata[
            (metadata['fold'].isin(target_folds)) & 
            (~metadata['class_id'].isin(human_class_ids))
        ]
        
        # Build full paths to audio files
        noise_files = []
        excluded_human = 0
        excluded_fold = 0
        
        # Process each file in the metadata
        for _, row in split_files.iterrows():
            fold = row['fold']
            file_name = row['slice_file_name']
            
            # Full path to the audio file
            file_path = audio_dir / f"fold{fold}" / file_name
            
            if file_path.exists():
                noise_files.append(file_path)
            else:
                logger.warning(f"Audio file not found: {file_path}")
                
        # Count excluded files for logging
        excluded_human = len(metadata[metadata['class_id'].isin(human_class_ids)])
        excluded_fold = len(metadata[~metadata['fold'].isin(target_folds)])
        
        logger.info(f"Selected {len(noise_files)} UrbanSound8K audio files for {split_name} split")
        logger.info(f"Excluded {excluded_human} human-related sounds and {excluded_fold} files from other folds")
        
        return noise_files
        
    except Exception as e:
        logger.error(f"Error processing UrbanSound8K metadata: {e}")
        return []

def initialize_silero_vad(
    model_path: Optional[str] = None, force_reload: bool = False, device: str = None
) -> Tuple[torch.nn.Module, Dict]:
    """
    Initialize the Silero VAD model.
    """
    logger.info("Initializing Silero VAD model")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try using the silero-vad package (preferred method)
    try:
        from silero_vad import (
            load_silero_vad,
            get_speech_timestamps,
            save_audio,
            read_audio,
            VADIterator,
            collect_chunks
        )
        
        # Load model using the pip package
        logger.info("Loading Silero VAD using pip package (recommended method)")
        model = load_silero_vad(onnx=False)
        model = model.to(device)
        
        # Create utils dictionary that matches torch.hub structure
        utils = {
            "get_speech_timestamps": get_speech_timestamps,
            "save_audio": save_audio,
            "read_audio": read_audio,
            "VADIterator": VADIterator,
            "collect_chunks": collect_chunks
        }
        
    except ImportError:
        logger.info("silero-vad package not found, falling back to torch.hub")
        
        # Handle local model path if provided
        if model_path and Path(model_path).exists():
            logger.info(f"Loading Silero VAD from local path: {model_path}")
            model = torch.jit.load(model_path, map_location=device)
            # Import utils separately when loading locally
            _, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
                verbose=False,
                trust_repo=True,
            )
        else:
            logger.info("Downloading Silero VAD from torch hub")
            # Use torch hub to get the model and utils
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=force_reload,
                onnx=False,
                verbose=False,
                trust_repo=True,
            )
            model = model.to(device)

    logger.info("Silero VAD model initialized successfully")
    return model, utils


def generate_silero_vad_labels(
    audio: np.ndarray,
    sample_rate: int,
    model: torch.nn.Module,
    threshold: float = 0.5,
    window_size_samples: int = 512,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    hop_length: int = 160,
    win_length: int = 400,
    utils=None,  # Add utils parameter
) -> np.ndarray:
    """
    Generate frame-level VAD labels using Silero VAD.
    """
    # Get model device
    device = next(model.parameters()).device
    
    # Convert audio to tensor and ensure correct format
    if not torch.is_tensor(audio):
        audio_tensor = torch.FloatTensor(audio)
    else:
        audio_tensor = audio

    # Move tensor to the same device as the model
    audio_tensor = audio_tensor.to(device)
    
    # Normalize if needed
    if torch.abs(audio_tensor).max() > 1.0:
        audio_tensor = audio_tensor / torch.abs(audio_tensor).max()

    # Get speech timestamps from Silero VAD
    if utils and "get_speech_timestamps" in utils:
        # Use the utils version if available
        speech_timestamps = utils["get_speech_timestamps"](
            audio_tensor,
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            window_size_samples=window_size_samples,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
        )
    else:
        # Fallback - load the model and get the utils
        model_obj, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
            trust_repo=True
        )
        
        # Now use the loaded utils
        speech_timestamps = utils["get_speech_timestamps"](
            audio_tensor,
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            window_size_samples=window_size_samples,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            return_seconds=False,
        )

    # Calculate number of frames - add safety for short audio
    n_frames = max(1, 1 + (len(audio) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Fill in speech frames
    for speech_segment in speech_timestamps:
        start_sample = speech_segment["start"]
        end_sample = speech_segment["end"]

        # Convert sample indices to frame indices
        start_frame = max(0, (start_sample - win_length) // hop_length + 1)
        end_frame = min(n_frames, (end_sample - win_length) // hop_length + 1)

        # Set frames in speech segments to 1
        vad_labels[start_frame:end_frame] = 1.0

    return vad_labels


def create_clean_speech_sample_with_silero(
    speech_files, duration, sample_rate, win_length, hop_length, vad_model
):
    """
    Create a clean speech sample and generate labels with Silero VAD.

    Args:
        speech_files: List of speech audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        vad_model: Silero VAD model

    Returns:
        Tuple of (clean audio, VAD labels)
    """
    model, utils = vad_model
    # Select random speech file
    speech_path = random.choice(speech_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)

    # Ensure consistent length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)

    # Generate labels using Silero VAD
    vad_labels = generate_silero_vad_labels(
        speech,
        sample_rate,
        model,
        hop_length=hop_length,
        win_length=win_length,
        utils=utils,  # Pass utils as a named parameter
    )

    # Normalize but don't add any noise or effects
    speech = np.clip(speech, -1.0, 1.0)

    return speech, vad_labels


def get_balanced_esc50_files(esc50_files, n_samples_per_category=5):
    """Get a balanced selection of ESC-50 files across all categories."""
    # Group files by category
    categories = {}
    for file in esc50_files:
        category = file.stem.split("-")[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(file)

    # Sample evenly from each category
    balanced_files = []
    for category, files in categories.items():
        if files:  # Skip empty categories (like the excluded ones)
            sampled = random.sample(files, min(n_samples_per_category, len(files)))
            balanced_files.extend(sampled)

    return balanced_files


# Add after other create_*_sample functions:


def create_esc50_negative_sample(
    esc50_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
):
    """Create a negative sample using only ESC-50 environmental sounds."""
    # Select a source file
    file_path = random.choice(esc50_files)

    # Get category information from filename: {category}-{fold}-{ID}-{take}.wav
    category = int(file_path.stem.split("-")[0])

    # Load audio
    audio = load_and_process_audio(file_path, sample_rate, duration)

    # Create zero labels (non-speech)
    n_frames = max(1, 1 + (len(audio) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Normalize
    audio = np.clip(audio, -1.0, 1.0)

    return audio, vad_labels, category


def process_vocalset(
    root_dir: pathlib.Path, split_name: str = "train", sample_rate: int = 16000
) -> list[pathlib.Path]:
    """
    Process VocalSet dataset with train/val/test splitting based on defined singer lists.

    Args:
        root_dir: Root directory where dataset is extracted
        split_name: Which split to use ('train', 'val', or 'test')
        sample_rate: Target sample rate

    Returns:
        List of paths to singing/vocal audio files for the requested split
    """
    # 1. More flexible directory finding (keep as is)
    possible_paths = [
        root_dir / "FULL",
        root_dir / "full",
        root_dir / "VocalSet" / "FULL",
        root_dir / "VocalSet-master" / "FULL",
        root_dir.parent / "FULL",
        root_dir.parent / "VocalSet" / "FULL",
    ]

    glob_paths = list(root_dir.glob("**/[Ff][Uu][Ll][Ll]"))
    glob_parent_paths = list(root_dir.parent.glob("**/[Ff][Uu][Ll][Ll]"))

    all_possible_paths = possible_paths + glob_paths + glob_parent_paths

    # Find the first valid path
    vocalset_dir = None
    for path in all_possible_paths:
        if path.exists():
            vocalset_dir = path
            logger.info(f"Found VocalSet directory at: {vocalset_dir}")
            break

    if not vocalset_dir:
        logger.warning("VocalSet directory not found after exhaustive search")
        return []

    # 2. Find ALL wav files first
    all_files = list(vocalset_dir.rglob("**/*.wav"))
    logger.info(f"Found total of {len(all_files)} WAV files in VocalSet")

    # Debug info on file locations
    if len(all_files) > 0:
        sample_files = random.sample(all_files, min(5, len(all_files)))
        logger.info(f"Sample file paths: {[str(f) for f in sample_files]}")

    # 3. Define singer IDs based on the text files
    # Test singers from test_singers.rtfd/TXT.rtf
    test_singers = ["female2", "female8", "male3", "male5", "male10"]

    # Train singers from train_singers.rtfd/TXT.rtf
    train_singers_full = [
        "female1",
        "female3",
        "female4",
        "female5",
        "female6",
        "female7",
        "female9",
        "male1",
        "male2",
        "male4",
        "male6",
        "male7",
        "male8",
        "male9",
        "male11",
    ]

    # Split train singers into train and validation
    # Use female9, male8, male9 for validation (like in original code)
    val_singers = ["female9", "male8", "male9"]
    train_singers = [s for s in train_singers_full if s not in val_singers]

    # Select the appropriate singers for this split
    if split_name == "train":
        target_singers = train_singers
    elif split_name == "val":
        target_singers = val_singers
    else:  # test
        target_singers = test_singers

    logger.info(
        f"Using {len(target_singers)} singers for {split_name} split: {target_singers}"
    )

    # Build patterns for matching filenames
    all_patterns = []
    for singer in target_singers:
        # Add common pattern formats
        all_patterns.extend([f"{singer}_", f"{singer}/", f"/{singer}/"])

    # Find all WAV files that match any of the patterns
    vocal_files = []
    for file in all_files:
        file_str = str(file).lower()
        if any(pattern in file_str for pattern in all_patterns):
            vocal_files.append(file)

    # 4. If pattern matching failed, fallback to random splitting
    if len(vocal_files) < 10:  # Very few files found with pattern matching
        logger.warning(
            f"Pattern matching found only {len(vocal_files)}/{len(all_files)} files. "
            f"Falling back to random split to avoid data loss."
        )

        # Deterministic shuffling
        random.seed(42 + {"train": 0, "val": 1, "test": 2}[split_name])
        shuffled_files = list(all_files)
        random.shuffle(shuffled_files)

        # Split: 70% train, 15% val, 15% test
        n_total = len(shuffled_files)
        if split_name == "train":
            vocal_files = shuffled_files[: int(0.7 * n_total)]
        elif split_name == "val":
            vocal_files = shuffled_files[int(0.7 * n_total) : int(0.85 * n_total)]
        else:  # test
            vocal_files = shuffled_files[int(0.85 * n_total) :]

    logger.info(
        f"Selected {len(vocal_files)} VocalSet audio files for {split_name} split"
    )
    return vocal_files


def process_esc50(
    root_dir: pathlib.Path, split_name: str = "train", sample_rate: int = 16000
) -> list[pathlib.Path]:
    """
    Process ESC-50 dataset with proper train/val/test splitting based on the original folds.
    Excludes all human-related sounds to avoid contaminating the negative samples with speech.

    Args:
        root_dir: Root directory where dataset is extracted
        split_name: Which split to use ('train', 'val', or 'test')
        sample_rate: Target sample rate

    Returns:
        List of paths to noise audio files for the requested split
    """
    # Find the ESC-50 directory
    esc50_dirs = list(root_dir.glob("*ESC-50-master*"))
    if not esc50_dirs:
        logger.warning(f"ESC-50 directory not found in {root_dir}")
        return []

    esc50_dir = esc50_dirs[0]
    logger.info(f"Found ESC-50 directory at {esc50_dir}")

    # Find audio directory
    audio_dir = esc50_dir / "audio"
    if not audio_dir.exists():
        audio_dirs = list(esc50_dir.rglob("audio"))
        if audio_dirs:
            audio_dir = audio_dirs[0]
            logger.info(f"Found audio directory at {audio_dir}")
        else:
            logger.warning(f"ESC-50 audio directory not found at {esc50_dir}")
            return []

    # Get all WAV files
    all_files = list(audio_dir.glob("*.wav"))
    logger.info(f"Found {len(all_files)} ESC-50 audio files")

    # Map split name to fold numbers (ESC-50 has 5 folds, numbered 1-5 in filenames)
    if split_name == "train":
        target_folds = ["1", "2", "3"]
    elif split_name == "val":
        target_folds = ["4"]
    else:  # test
        target_folds = ["5"]

    # Define all human-related sound categories to exclude
    human_categories = [
        "20",  # crying_baby
        "21",  # sneezing
        "22",  # clapping
        "23",  # breathing
        "24",  # coughing
        "25",  # footsteps
        "26",  # laughing
        "27",  # brushing_teeth
        "28",  # snoring
        "29",  # drinking_sipping
    ]

    # Filter files by fold and exclude human sounds
    noise_files = []
    excluded_human = 0
    excluded_fold = 0

    for file in all_files:
        # ESC-50 filename format: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
        parts = file.stem.split("-")

        if len(parts) < 4:
            logger.warning(f"Unexpected filename format: {file.name}")
            continue

        fold = parts[0]
        target = parts[3]  # Category/class is in the 4th position (index 3)

        # First check if this file belongs to the right fold
        if fold not in target_folds:
            excluded_fold += 1
            continue

        # Then check if this is a human-related sound
        if target in human_categories:
            excluded_human += 1
            continue

        # If it passes both checks, include it
        noise_files.append(file)

    logger.info(
        f"Selected {len(noise_files)} ESC-50 audio files for {split_name} split"
    )
    logger.info(
        f"Excluded {excluded_human} human-related sounds and {excluded_fold} files from other folds"
    )

    return noise_files


def download_and_extract_zip(url: str, dest: pathlib.Path):
    """Download and extract a ZIP archive if not already done."""
    fname = (
        dest / url.split("/")[-1].split("?")[0]
    )  # Handle Zenodo URLs with query params
    if not fname.exists():
        success = download_file(url, fname)
        if not success:
            logger.error(f"Failed to download {url}")
            sys.exit(f"❌ Failed to download {url}")
    else:
        logger.info(f"File already exists: {fname}, skipping download")

    mark = dest / f".extracted_{fname.name}"
    if not mark.exists():
        logger.info(f"Extracting ZIP archive: {fname.name}")
        print(f"📦 Extracting {fname.name}")
        import zipfile

        with zipfile.ZipFile(fname, "r") as zip_ref:
            zip_ref.extractall(path=dest)
        mark.touch()
        logger.info(f"Extraction completed and marked with {mark}")
    else:
        logger.info(f"Archive already extracted (marker file exists): {mark}")


# Add this new function to create mixed negative samples:
# Create a new utility function for generating overlapping speech:
def create_overlapping_speech(
    speech_files, noise_files, duration, sample_rate, win_length, hop_length
):
    """
    Create an overlapping speech sample with background noise.

    Args:
        speech_files: List of speech audio files
        noise_files: List of noise audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select two different speech files
    speech1, speech2 = random.sample(speech_files, 2)

    # Load and process both speech files
    primary = load_and_process_audio(speech1, sample_rate, duration)
    secondary = load_and_process_audio(speech2, sample_rate, duration)

    # Make primary and secondary same length
    target_length = int(duration * sample_rate)
    primary = librosa.util.fix_length(primary, size=target_length)
    secondary = librosa.util.fix_length(secondary, size=target_length)

    # Make secondary speech quieter (foreground vs background speaker)
    snr_db = random.uniform(5, 15)  # Reasonable SNR for overlapping speech
    alpha = 10 ** (-snr_db / 20) * np.std(primary) / (np.std(secondary) + 1e-7)

    # Add noise too for realism
    noise_path = random.choice(noise_files)
    noise = load_and_process_audio(noise_path, sample_rate, duration)

    # Create the final mixture
    mix = primary + alpha * secondary + 0.1 * noise

    # Generate VAD labels based on energy
    frame_energies = librosa.feature.rms(
        y=mix, frame_length=win_length, hop_length=hop_length
    )[0]

    # Apply threshold
    energy_threshold = adaptive_energy_threshold(frame_energies)
    vad_labels = (frame_energies > energy_threshold).astype(np.float32)

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels


def process_musan(
    root_dir: pathlib.Path,
    split_name: str = "train",
    sample_rate: int = 16000,
) -> tuple[list[pathlib.Path], list[pathlib.Path]]:
    """
    Process MUSAN dataset with train/val/test splitting.

    Args:
        root_dir: Root directory where MUSAN is extracted
        split_name: Which split to use ('train', 'val', or 'test')
        sample_rate: Target sample rate

    Returns:
        Tuple of (speech_files, noise_files) for the requested split
    """
    musan_root = root_dir / "musan"

    if not musan_root.exists():
        logger.warning(f"MUSAN directory not found at {musan_root}")
        return [], []

    # Load all files
    all_speech_files = list(musan_root.joinpath("speech").rglob("*.wav"))
    all_noise_files = list(musan_root.joinpath("noise").rglob("*.wav"))

    logger.info(
        f"Found {len(all_speech_files)} MUSAN speech files and {len(all_noise_files)} noise files"
    )

    # Use deterministic shuffling for consistent splits
    random.seed(42)  # Fixed seed for reproducibility

    # Shuffle both speech and noise files
    shuffled_speech = all_speech_files.copy()
    random.shuffle(shuffled_speech)

    shuffled_noise = all_noise_files.copy()
    random.shuffle(shuffled_noise)

    # Split: 70% train, 15% val, 15% test
    speech_train_idx = int(0.7 * len(shuffled_speech))
    speech_val_idx = int(0.85 * len(shuffled_speech))

    noise_train_idx = int(0.7 * len(shuffled_noise))
    noise_val_idx = int(0.85 * len(shuffled_noise))

    # Get the appropriate files based on split
    if split_name == "train":
        speech_files = shuffled_speech[:speech_train_idx]
        noise_files = shuffled_noise[:noise_train_idx]
        logger.info(
            f"Using {len(speech_files)}/{len(all_speech_files)} speech files and "
            f"{len(noise_files)}/{len(all_noise_files)} noise files for training"
        )
    elif split_name == "val":
        speech_files = shuffled_speech[speech_train_idx:speech_val_idx]
        noise_files = shuffled_noise[noise_train_idx:noise_val_idx]
        logger.info(
            f"Using {len(speech_files)}/{len(all_speech_files)} speech files and "
            f"{len(noise_files)}/{len(all_noise_files)} noise files for validation"
        )
    else:  # test
        speech_files = shuffled_speech[speech_val_idx:]
        noise_files = shuffled_noise[noise_val_idx:]
        logger.info(
            f"Using {len(speech_files)}/{len(all_speech_files)} speech files and "
            f"{len(noise_files)}/{len(all_noise_files)} noise files for testing"
        )

    return speech_files, noise_files


def create_mixed_negative_sample(
    noise_files,
    music_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
    mix_ratio=0.5,
):
    """
    Create a mixed negative sample with noise and music.

    Args:
        noise_files: List of noise audio files
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        mix_ratio: Ratio for mixing music with noise

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select source files
    noise_file = random.choice(noise_files)
    music_file = random.choice(music_files)

    # Load audio
    noise = load_and_process_audio(noise_file, sample_rate, duration)
    music = load_and_process_audio(music_file, sample_rate, duration)

    # Mix with the specified ratio
    target_length = int(duration * sample_rate)
    noise = librosa.util.fix_length(noise, size=target_length)
    music = librosa.util.fix_length(music, size=target_length)
    mix = noise + mix_ratio * music

    # Create zero labels (non-speech)
    n_frames = max(1, 1 + (len(mix) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels


def create_speech_music_sample(
    speech_files, music_files, duration, sample_rate, win_length, hop_length
):
    """
    Create a sample with speech over music background.

    Args:
        speech_files: List of speech audio files
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select source files
    speech_path = random.choice(speech_files)
    music_path = random.choice(music_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)
    music = load_and_process_audio(music_path, sample_rate, duration)

    # Ensure same length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)
    music = librosa.util.fix_length(music, size=target_length)

    # Create labels from clean speech (before mixing)
    frame_energies = librosa.feature.rms(
        y=speech, frame_length=win_length, hop_length=hop_length
    )[0]

    threshold = adaptive_energy_threshold(frame_energies)
    vad_labels = (frame_energies > threshold).astype(np.float32)

    # Mix speech with music at varying SNR
    # Using lower SNR than speech+noise since music is less masking than noise
    snr_db = random.uniform(0, 25)  # Higher SNR range than with noise
    alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(music) + 1e-7)
    mix = speech + alpha * music

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels


def load_and_process_audio(file_path, sample_rate, duration=None, target_length=None):
    """
    Centralized function for loading and processing audio files.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        duration: Optional maximum duration to load
        target_length: Optional target length for padding/trimming

    Returns:
        Processed audio array
    """
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)

    if target_length and len(audio) != target_length:
        audio = librosa.util.fix_length(audio, size=target_length)

    return audio


def generate_vad_labels(
    audio, win_length, hop_length, threshold_percentile=0.3, smooth=True
):
    """
    Generate VAD labels from audio using energy-based approach.

    Args:
        audio: Input audio array
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        threshold_percentile: Percentile for energy threshold
        smooth: Whether to smooth labels with minimum segment constraint

    Returns:
        Frame-level VAD labels
    """
    # Calculate frame energies
    frame_energies = librosa.feature.rms(
        y=audio, frame_length=win_length, hop_length=hop_length
    )[0]

    # Determine threshold and create binary labels
    sorted_energies = np.sort(frame_energies)
    threshold_idx = int(len(sorted_energies) * threshold_percentile)
    energy_threshold = sorted_energies[threshold_idx]
    vad_labels = (frame_energies > energy_threshold).astype(np.float32)

    # Apply smoothing if requested
    if smooth:
        min_speech_frames = 3  # About 50ms at 16kHz with default hop_length
        for i in range(len(vad_labels) - min_speech_frames + 1):
            if 0 < sum(vad_labels[i : i + min_speech_frames]) < min_speech_frames:
                vad_labels[i : i + min_speech_frames] = 0

    return vad_labels


def create_negative_sample(
    source_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
    mix_ratio=0.5,
    category="music",
    idx=None,  # Add index parameter to ensure variety
):
    """
    Create a negative sample (noise or music).

    Args:
        source_files: List of source audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        mix_ratio: Only used for 'mixed' category (deprecated)
        category: Type of negative sample ('noise', 'music')
        idx: Sample index for deterministic but varied file selection

    Returns:
        Tuple of (audio, VAD labels)
    """
    # Select a file deterministically if idx is provided
    if idx is not None and len(source_files) > 0:
        # Use modulo to cycle through all files
        file_path = source_files[idx % len(source_files)]
    else:
        # Select a single source file randomly as fallback
        file_path = random.choice(source_files)

    # Load audio
    audio = load_and_process_audio(file_path, sample_rate, duration)

    # Ensure consistent length
    target_length = int(duration * sample_rate)
    audio = librosa.util.fix_length(audio, size=target_length)

    # Optional: apply slight random gain variation to make the dataset more robust
    gain_factor = random.uniform(0.8, 1.2)
    audio = audio * gain_factor

    # Create zero labels (non-speech)
    n_frames = max(1, 1 + (len(audio) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Normalize
    audio = np.clip(audio, -1.0, 1.0)

    return audio, vad_labels


def create_noise_noise_sample(
    noise_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
):
    """
    Create a negative sample by mixing two different noise sources.

    Args:
        noise_files: List of noise audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select two different noise files
    if len(noise_files) < 2:
        # Fallback if we don't have enough files
        noise_file = random.choice(noise_files)
        noise1 = load_and_process_audio(noise_file, sample_rate, duration)
        noise2 = noise1.copy()
    else:
        noise_file1, noise_file2 = random.sample(noise_files, 2)
        noise1 = load_and_process_audio(noise_file1, sample_rate, duration)
        noise2 = load_and_process_audio(noise_file2, sample_rate, duration)

    # Ensure consistent length
    target_length = int(duration * sample_rate)
    noise1 = librosa.util.fix_length(noise1, size=target_length)
    noise2 = librosa.util.fix_length(noise2, size=target_length)

    # Mix with random ratio
    mix_ratio = random.uniform(0.3, 0.7)
    mix = noise1 + mix_ratio * noise2

    # Create zero labels (non-speech)
    n_frames = max(1, 1 + (len(mix) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels


def create_music_music_sample(
    music_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
):
    """
    Create a negative sample by mixing two different music sources.

    Args:
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select two different music files
    if len(music_files) < 2:
        # Fallback if we don't have enough files
        music_file = random.choice(music_files)
        music1 = load_and_process_audio(music_file, sample_rate, duration)
        music2 = music1.copy()
    else:
        music_file1, music_file2 = random.sample(music_files, 2)
        music1 = load_and_process_audio(music_file1, sample_rate, duration)
        music2 = load_and_process_audio(music_file2, sample_rate, duration)

    # Ensure consistent length
    target_length = int(duration * sample_rate)
    music1 = librosa.util.fix_length(music1, size=target_length)
    music2 = librosa.util.fix_length(music2, size=target_length)

    # Apply different EQ to one of the tracks to create more variation
    if random.random() > 0.5:
        music2 = apply_eq(music2, sample_rate)

    # Mix with random ratio
    mix_ratio = random.uniform(0.3, 0.7)
    mix = music1 + mix_ratio * music2

    # Create zero labels (non-speech)
    n_frames = max(1, 1 + (len(mix) - win_length) // hop_length)
    vad_labels = np.zeros(n_frames, dtype=np.float32)

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels


def validate_audio_sample(
    audio, sr, min_rms=0.0005, max_silence_ratio=0.8, min_duration_sec=1.0
):
    """
    Validate audio sample quality before adding to dataset.

    Args:
        audio: Audio array
        sr: Sample rate
        min_rms: Minimum RMS energy required (very quiet audio)
        max_silence_ratio: Maximum ratio of silence allowed
        min_duration_sec: Minimum duration in seconds

    Returns:
        bool: True if sample passes validation
    """
    # Check minimum length
    if len(audio) < min_duration_sec * sr:
        return False

    # Check if audio is too quiet
    rms = np.sqrt(np.mean(audio**2))
    if rms < min_rms:
        return False

    # Check if audio is clipping
    if np.max(np.abs(audio)) >= 1.0:
        # Soft clipping detected
        if (
            np.sum(np.abs(audio) > 0.99) / len(audio) > 0.01
        ):  # More than 1% near clipping
            return False

    # Check for excessive silence
    frame_energies = librosa.feature.rms(
        y=audio, frame_length=int(0.025 * sr), hop_length=int(0.010 * sr)
    )[0]

    silence_threshold = np.mean(frame_energies) * 0.1
    silence_ratio = np.sum(frame_energies < silence_threshold) / len(frame_energies)

    if silence_ratio > max_silence_ratio:
        return False

    return True


def ingest_fleurs(
    lang_list: list[str],
    out_dir: pathlib.Path,
    sr: int,
    streaming: bool = False,
    split: str = "train",
    max_per_lang: int | None = None,
    shuffle_seed: int = 42,
    cache_dir: str = None,
) -> list[pathlib.Path]:
    """
    Download/stream the requested FLEURS languages, resample,
    save as WAV and return the file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_wavs = []

    # Create a marker directory to track processed languages
    markers_dir = Path(out_dir) / ".processed_markers"
    markers_dir.mkdir(exist_ok=True, parents=True)

    for lang in lang_list:
        # 'all' is a special HF config that concatenates all 102 languages
        cfg = lang.strip()

        # Check if this language has already been processed
        marker_file = markers_dir / f"{cfg}_{split}_processed.marker"
        existing_files = list(out_dir.glob(f"{cfg}_*.wav"))

        if marker_file.exists() and existing_files:
            logger.info(f"FLEURS {cfg} ({split}) already processed, skipping download")
            # Add existing wav files to our list
            all_wavs.extend(existing_files)
            continue

        print(f"-- Loading FLEURS {cfg} ({'stream' if streaming else 'download'})")

        try:
            ds = load_dataset(
                "google/fleurs",
                cfg,
                split="dev" if split == "val" else split,
                streaming=streaming,
                trust_remote_code=True,
                cache_dir=cache_dir,
            )

            # Balance: keep at most N clips for this language
            if max_per_lang:
                if streaming:
                    # In streaming mode, shuffle and limit the number of examples
                    ds = ds.shuffle(seed=shuffle_seed, buffer_size=1000).take(
                        max_per_lang
                    )
                else:
                    # In non-streaming mode, shuffle and select specific indices
                    ds = ds.shuffle(seed=shuffle_seed).select(
                        range(min(max_per_lang, len(ds)))
                    )

            # make sure every clip is decoded at the target rate
            ds = ds.cast_column("audio", Audio(sampling_rate=sr))

            lang_wavs = []
            for i, ex in enumerate(tqdm(ds, desc=f"{cfg} clips")):
                wav = ex["audio"]["array"]
                path = out_dir / f"{cfg}_{i:06d}.wav"
                sf.write(path, wav, sr)
                lang_wavs.append(path)

            # If we processed at least one file successfully, create a marker
            if lang_wavs:
                marker_file.touch()
                all_wavs.extend(lang_wavs)
                logger.info(
                    f"Successfully processed {len(lang_wavs)} clips for {cfg} ({split})"
                )

        except Exception as e:
            logger.error(f"Error processing FLEURS {cfg}: {str(e)}")
            logger.error("Continuing with other languages...")

    return all_wavs


def run_cmd(cmd: str):
    """Run shell command and exit on error (simple wrapper)."""
    logger.info(f"Executing command: {cmd}")
    print(f"➜  {cmd}")
    if subprocess.call(cmd, shell=True) != 0:
        logger.error(f"Command failed: {cmd}")
        sys.exit(f"❌ Command failed: {cmd}")
    logger.info("Command completed successfully")


def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    logger.info(f"Setting random seed to {seed} for reproducibility")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # Torch not available, but that's okay for data prep only


def download_file(url, dest_path):
    """
    Cross-platform file download function that works on Windows and Unix systems.
    Checks if file exists first before downloading.
    """
    import os
    import requests
    from tqdm import tqdm

    # Check if file already exists
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path}, skipping download")
        return True

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
    """Download and extract a tar.gz archive if not already done."""
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


def add_reverb(audio, sr, wet_level=None):
    """
    Apply simple reverb effect through convolution with an exponential decay.

    Args:
        audio: Input audio signal
        sr: Sample rate
        wet_level: Amount of reverb (0-1), random if None

    Returns:
        Processed audio with reverb
    """
    if wet_level is None:
        wet_level = random.uniform(0.1, 0.5)

    # Create a decay envelope for the reverb impulse response
    decay_len = int(sr * random.uniform(0.1, 0.7))  # 100-700ms decay
    impulse = np.exp(-np.arange(decay_len) / (sr * random.uniform(0.05, 0.2)))
    impulse = impulse / np.sum(impulse)  # Normalize impulse response

    # Apply convolution reverb
    wet = np.convolve(audio, impulse)[: len(audio)]

    # Normalize wet signal to prevent clipping
    wet = wet / (np.max(np.abs(wet)) + 1e-7)

    # Mix dry/wet signals
    return (1 - wet_level) * audio + wet_level * wet


def apply_eq(audio, sr):
    """
    Apply random equalization to simulate different microphones or channels.
    """
    # Original audio
    eq_audio = audio.copy()

    # Apply 2-3 random frequency adjustments
    for _ in range(random.randint(2, 3)):
        # Calculate Nyquist frequency
        nyquist = sr / 2
        
        # Random center frequency (ensuring it's below Nyquist)
        max_freq = min(7000, nyquist * 0.95)  # Cap at 7kHz or 95% of Nyquist
        center_freq = random.uniform(100, max_freq)
        
        # Normalize frequency to range (0, 1)
        w0 = center_freq / nyquist
        
        # Safety check to ensure w0 is strictly between 0 and 1
        w0 = max(0.001, min(0.999, w0))
        
        q = random.uniform(0.5, 5.0)
        gain_db = random.uniform(-10, 10)  # dB

        # Create and apply peaking filter
        b, a = signal.iirpeak(w0, q, gain_db)
        eq_audio = signal.lfilter(b, a, eq_audio)

    # Normalize output to prevent level blow-up
    eq_audio = eq_audio / (np.max(np.abs(eq_audio)) + 1e-7)

    return eq_audio


def create_speech_gap_speech(
    speech_files, duration, sample_rate, hop_length, win_length
):
    """
    Create a speech-silence-speech pattern with appropriate frame-level labels.

    Args:
        speech_files: List of available speech files
        duration: Total desired duration in seconds
        sample_rate: Sample rate for audio
        hop_length: Hop length for frame calculation
        win_length: Window length for frame calculation

    Returns:
        Tuple of (audio_with_gap, frame_labels)
    """
    # Choose random gap duration (0.3-1.5 seconds)
    gap_duration = random.uniform(0.3, 1.5)

    # Calculate durations for speech segments
    remaining_duration = duration - gap_duration
    first_duration = random.uniform(remaining_duration * 0.3, remaining_duration * 0.7)
    second_duration = remaining_duration - first_duration

    # Load two different speech samples
    speech1, speech2 = random.sample(speech_files, 2)
    first_speech, _ = librosa.load(speech1, sr=sample_rate, duration=first_duration)
    second_speech, _ = librosa.load(speech2, sr=sample_rate, duration=second_duration)

    # Fix lengths to exact durations
    first_speech = librosa.util.fix_length(
        first_speech, size=int(first_duration * sample_rate)
    )
    second_speech = librosa.util.fix_length(
        second_speech, size=int(second_duration * sample_rate)
    )

    # Create silence gap
    gap = np.zeros(int(gap_duration * sample_rate))

    # Combine segments
    combined_audio = np.concatenate([first_speech, gap, second_speech])

    # Calculate frame counts for each segment
    first_frames = 1 + (len(first_speech) - win_length) // hop_length
    gap_frames = 1 + (len(gap) - win_length) // hop_length
    second_frames = 1 + (len(second_speech) - win_length) // hop_length

    # Pre-allocate arrays of the correct size
    labels1 = np.zeros(first_frames, dtype=np.float32)
    labels_gap = np.zeros(gap_frames, dtype=np.float32)
    labels2 = np.zeros(second_frames, dtype=np.float32)

    # Generate labels for first speech segment
    frame_energies1 = librosa.feature.rms(
        y=first_speech, frame_length=win_length, hop_length=hop_length
    )[0]

    # Verify frame count matches
    assert (
        len(frame_energies1) == first_frames
    ), f"Frame count mismatch: {len(frame_energies1)} vs {first_frames}"

    # Calculate threshold and assign labels
    threshold1 = adaptive_energy_threshold(frame_energies1)
    labels1 = (frame_energies1 > threshold1).astype(np.float32)

    # Generate labels for second speech segment
    frame_energies2 = librosa.feature.rms(
        y=second_speech, frame_length=win_length, hop_length=hop_length
    )[0]

    # Verify frame count matches
    assert (
        len(frame_energies2) == second_frames
    ), f"Frame count mismatch: {len(frame_energies2)} vs {second_frames}"

    # Calculate threshold using adaptive method
    threshold2 = adaptive_energy_threshold(frame_energies2)
    labels2 = (frame_energies2 > threshold2).astype(np.float32)

    # Combine all labels
    combined_labels = np.concatenate([labels1, labels_gap, labels2])

    return combined_audio, combined_labels


def adaptive_energy_threshold(frame_energies, min_percentile=0.2, max_percentile=0.5):
    """
    Determine an adaptive energy threshold based on the energy distribution.

    Args:
        frame_energies: Array of frame-level energy values
        min_percentile: Minimum percentile to consider (default: 0.2)
        max_percentile: Maximum percentile to consider (default: 0.5)

    Returns:
        Adaptive energy threshold
    """
    # Create histogram of frame energies (in log scale for better separation)
    log_energies = np.log(frame_energies + 1e-6)  # Add small constant to avoid log(0)
    hist, bin_edges = np.histogram(log_energies, bins=50)

    # Find the valley in the histogram (bimodal distribution separation point)
    # Smooth histogram first for more reliable valley detection
    smoothed_hist = np.convolve(hist, np.ones(3) / 3, mode="same")
    peaks, _ = signal.find_peaks(smoothed_hist)
    valleys, _ = signal.find_peaks(-smoothed_hist)

    if len(peaks) >= 2 and len(valleys) >= 1:
        # Find the most significant valley between peaks
        significant_valleys = [v for v in valleys if v > peaks[0] and v < peaks[-1]]
        if significant_valleys:
            valley_idx = significant_valleys[0]
            threshold_value = np.exp(bin_edges[valley_idx])
            return threshold_value

    # Fallback if bimodal detection fails: use content-based adaptive percentile
    # For high-energy audio (likely more speech), use lower percentile
    # For low-energy audio (likely less speech), use higher percentile
    energy_level = np.mean(frame_energies) / np.max(frame_energies)
    adaptive_percentile = max_percentile - (
        energy_level * (max_percentile - min_percentile)
    )
    threshold_idx = int(len(frame_energies) * adaptive_percentile)
    sorted_energies = np.sort(frame_energies)
    return sorted_energies[threshold_idx]


def create_clean_speech_sample(
    speech_files, duration, sample_rate, win_length, hop_length
):
    """
    Create a clean speech sample without added noise or effects.

    Args:
        speech_files: List of speech audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (clean audio, VAD labels)
    """
    # Select random speech file
    speech_path = random.choice(speech_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)

    # Ensure consistent length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)

    # Create labels from speech
    frame_energies = librosa.feature.rms(
        y=speech, frame_length=win_length, hop_length=hop_length
    )[0]

    threshold = adaptive_energy_threshold(frame_energies)
    vad_labels = (frame_energies > threshold).astype(np.float32)

    # Normalize but don't add any noise or effects
    speech = np.clip(speech, -1.0, 1.0)

    return speech, vad_labels

# Add this function at the top of your file (near other utility functions)
def safe_augment_audio(audio_segment, sample_rate):
    """
    Safely apply augmentations with proper error handling.
    """
    augmented = audio_segment.copy()
    
    try:
        # 1. Time Stretching (90% chance for diversity)
        if random.random() < 0.9:
            stretch_factor = random.uniform(0.85, 1.15)
            augmented = librosa.effects.time_stretch(augmented, rate=stretch_factor)
        
        # 2. Pitch Shifting (80% chance)
        if random.random() < 0.8:
            pitch_steps = random.uniform(-2, 2)
            augmented = librosa.effects.pitch_shift(
                augmented, sr=sample_rate, n_steps=pitch_steps
            )
        
        # 3. Volume Scaling (90% chance)
        if random.random() < 0.9:
            volume_factor = random.uniform(0.6, 1.4)
            augmented = augmented * volume_factor
        
        # 4. Low-level noise addition (50% chance)
        if random.random() < 0.5:
            noise = np.random.randn(len(augmented))
            noise_level = random.uniform(0.001, 0.01)
            augmented = augmented + noise * noise_level
        
        # 5. Reverb (60% chance)
        if random.random() < 0.6:
            try:
                wet_level = random.uniform(0.1, 0.5)
                augmented = add_reverb(augmented, sample_rate, wet_level)
            except Exception as reverb_error:
                logger.warning(f"Reverb error (skipping this effect): {reverb_error}")
        
        # 6. EQ Filtering (70% chance) - use existing apply_eq function with exception handling
        if random.random() < 0.7:
            try:
                augmented = apply_eq(augmented, sample_rate)
            except Exception as eq_error:
                logger.warning(f"EQ error (skipping this effect): {eq_error}")
        
        # 7. Random silence insertion (30% chance)
        if random.random() < 0.3 and len(augmented) > sample_rate * 3:
            silence_duration = random.uniform(0.05, 0.3)  # 50-300ms
            silence_samples = int(silence_duration * sample_rate)
            silence_pos = random.randint(0, len(augmented) - silence_samples)
            augmented[silence_pos:silence_pos + silence_samples] = 0
        
        # 8. Shuffling chunks (15% chance - experimental)
        if random.random() < 0.15 and len(augmented) > sample_rate * 5:
            # Only for longer segments
            chunk_size = int(sample_rate * random.uniform(0.5, 1.0))
            if chunk_size < len(augmented) / 4:  # Ensure we have enough chunks
                # Split into chunks
                chunks = [augmented[i:i+chunk_size] for i in range(0, len(augmented), chunk_size)]
                # Shuffle some consecutive chunks
                shuffle_start = random.randint(0, len(chunks) - 3)
                shuffle_end = min(shuffle_start + random.randint(2, 4), len(chunks))
                random.shuffle(chunks[shuffle_start:shuffle_end])
                # Recombine
                augmented = np.concatenate(chunks)
        
        # Ensure consistent length
        augmented = librosa.util.fix_length(augmented, size=len(audio_segment))
        
        # Normalize to prevent clipping
        augmented = augmented / (np.max(np.abs(augmented)) + 1e-7)
        
        return augmented, True
        
    except Exception as e:
        logger.warning(f"Error during audio augmentation: {e}")
        return audio_segment, False  # Return original with failure flag

def process_musdb(
    root_dir: pathlib.Path,
    split_name: str = "train",
    sample_rate: int = 16000,
    cleanup_stems: bool = True,
    augment: bool = True,
    segments_per_track: int = 8,         # Extract multiple segments per track
    segment_length: int = 10,            # 10-second segments
    augmentations_per_segment: int = 5,  # Generate 5 augmented versions per segment
) -> list[pathlib.Path]:
    """
    Process MUSDB18HQ dataset with extensive augmentation to increase dataset diversity.
    Includes segment extraction, time/pitch shifting, EQ filtering, and more to
    maximize diversity from limited music tracks.

    Args:
        root_dir: Root directory where dataset should be stored
        split_name: Which split to use ('train', 'val', or 'test')
        sample_rate: Target sample rate
        cleanup_stems: Whether to delete original stems after combining (saves disk space)
        augment: Whether to apply augmentations (only for training set)
        segments_per_track: Number of segments to extract from each track
        segment_length: Duration in seconds for each segment
        augmentations_per_segment: Number of augmented versions to create per segment

    Returns:
        List of paths to instrumental audio files (original and augmented)
    """
    # Create musdb18hq directory if it doesn't exist
    musdb_root = root_dir / "musdb18hq"
    musdb_root.mkdir(parents=True, exist_ok=True)
    
    # Create directories to store processing markers and augmented files
    markers_dir = musdb_root / ".markers"
    markers_dir.mkdir(exist_ok=True, parents=True)
    
    # Directories for segments and augmentations
    segments_dir = musdb_root / "segments"
    augmented_dir = musdb_root / "augmented"
    
    if split_name == "train" and augment:
        segments_dir.mkdir(exist_ok=True, parents=True)
        augmented_dir.mkdir(exist_ok=True, parents=True)

    # First, download and extract the dataset if needed
    try:
        # Download and extract the zip file
        download_and_extract_zip(MUSDB18HQ_URL, musdb_root)
        logger.info(f"MUSDB18HQ dataset downloaded and extracted successfully")
    except Exception as e:
        logger.error(f"Error downloading/extracting MUSDB18HQ dataset: {e}")

    # Initialize the list to store file paths
    music_files = []

    # MUSDB18HQ structure is: train/song_title/ and test/song_title/
    if split_name in ["train", "val"]:
        search_dir = musdb_root / "train"
    else:  # test
        search_dir = musdb_root / "test"

    logger.info(f"Looking for MUSDB18HQ songs in {search_dir}")

    # Find all song directories
    song_dirs = [d for d in search_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(song_dirs)} song directories in {search_dir}")

    # Process each song
    for song_dir in tqdm(song_dirs, desc="Processing MUSDB songs"):
        # Create a unique marker file path for this song
        marker_file = markers_dir / f"{search_dir.name}_{song_dir.name}_processed.marker"
        
        # Define paths for all stems
        bass_path = song_dir / "bass.wav"
        drums_path = song_dir / "drums.wav"
        other_path = song_dir / "other.wav"
        vocals_path = song_dir / "vocals.wav"

        # Define output file path - maintain existing directory structure
        output_path = song_dir / "instrumental.wav"

        # Skip if the combined file already exists and has a marker
        if output_path.exists() and marker_file.exists():
            logger.info(
                f"Combined file already exists for {song_dir.name} (marked as processed), skipping"
            )
            music_files.append(output_path)
            continue
        
        # Check if all required stems exist
        if not (bass_path.exists() and drums_path.exists() and other_path.exists()):
            logger.warning(f"Missing stems for {song_dir.name}, skipping")
            continue

        # Skip if the combined file already exists (but not marked)
        if output_path.exists():
            logger.info(
                f"Combined file exists for {song_dir.name} but not marked, adding to tracking"
            )
            # Create marker file to indicate this song is processed
            marker_file.touch()
            music_files.append(output_path)
            
            # If cleanup is requested and not already done
            if cleanup_stems and all(p.exists() for p in [bass_path, drums_path, other_path]):
                logger.info(f"Cleaning up stem files for {song_dir.name}")
                # Remove stem files to save space
                stem_files = [bass_path, drums_path, other_path]
                if vocals_path.exists():
                    stem_files.append(vocals_path)
                    
                for stem in stem_files:
                    stem.unlink()
                logger.info(f"Removed {len(stem_files)} stem files from {song_dir.name}")
                
            continue

        # Load all stems
        try:
            bass, sr_bass = librosa.load(bass_path, sr=sample_rate, mono=True)
            drums, sr_drums = librosa.load(drums_path, sr=sample_rate, mono=True)
            other, sr_other = librosa.load(other_path, sr=sample_rate, mono=True)

            # Make sure all stems have the same length
            max_length = max(len(bass), len(drums), len(other))
            bass = librosa.util.fix_length(bass, size=max_length)
            drums = librosa.util.fix_length(drums, size=max_length)
            other = librosa.util.fix_length(other, size=max_length)

            # Mix stems together
            combined = bass + drums + other

            # Normalize to prevent clipping
            combined = combined / (np.max(np.abs(combined)) + 1e-7)

            # Save combined track
            sf.write(output_path, combined, sample_rate)

            # Create marker file to indicate this song is processed
            marker_file.touch()
            
            logger.info(f"Created combined instrumental track for {song_dir.name}")
            music_files.append(output_path)
            
            # Clean up stem files if requested
            if cleanup_stems:
                logger.info(f"Cleaning up stem files for {song_dir.name}")
                # Remove stem files to save space
                stem_files = [bass_path, drums_path, other_path]
                if vocals_path.exists():
                    stem_files.append(vocals_path)
                    
                for stem in stem_files:
                    stem.unlink()
                logger.info(f"Removed {len(stem_files)} stem files from {song_dir.name}")

        except Exception as e:
            logger.error(f"Error processing {song_dir.name}: {e}")

    # For train/val, split the files from the train folder
    if split_name in ["train", "val"]:
        # Use deterministic shuffling for consistent splits
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(music_files)

        # Split: 80% for train, 20% for validation
        split_idx = int(0.8 * len(music_files))

        if split_name == "train":
            music_files = music_files[:split_idx]
        else:  # val
            music_files = music_files[split_idx:]

        logger.info(f"Using {len(music_files)} original files for {split_name}")
    
    # ───────────────────────────────────────────────────────────────────────
    # SEGMENT EXTRACTION - Break long tracks into multiple segments
    # ───────────────────────────────────────────────────────────────────────
    segmented_files = []
    if split_name == "train" and augment and music_files:
        logger.info(f"Extracting {segments_per_track} segments from each of {len(music_files)} music files")
        
        for track_idx, track_path in enumerate(tqdm(music_files, desc="Extracting segments")):
            try:
                # Check if we've already processed this track
                segment_marker = markers_dir / f"{track_path.stem}_segments.marker"
                segment_files = list(segments_dir.glob(f"{track_path.stem}_seg*.wav"))
                
                if segment_marker.exists() and segment_files:
                    logger.info(f"Found {len(segment_files)} existing segments for {track_path.stem}")
                    segmented_files.extend(segment_files)
                    continue
                    
                # Load the full track
                audio, _ = librosa.load(track_path, sr=sample_rate)
                
                # Only proceed if the file is long enough
                min_duration = segment_length * sample_rate
                if len(audio) < min_duration:
                    logger.warning(f"Track {track_path.stem} too short for segmentation, skipping")
                    continue
                
                # Calculate how many complete segments we can extract
                total_segments = min(
                    segments_per_track,
                    (len(audio) // (segment_length * sample_rate))
                )
                
                # If we can extract at least one segment
                if total_segments > 0:
                    for seg_idx in range(total_segments):
                        # Determine segment boundaries with some overlap
                        if total_segments > 1:
                            # Distribute segments evenly across the track
                            start = seg_idx * (len(audio) - segment_length * sample_rate) // max(1, total_segments - 1)
                        else:
                            # Just take the beginning if we only have one segment
                            start = 0
                            
                        end = start + segment_length * sample_rate
                        
                        # Extract segment
                        segment = audio[start:end]
                        
                        # Save segment
                        segment_path = segments_dir / f"{track_path.stem}_seg{seg_idx}.wav"
                        sf.write(segment_path, segment, sample_rate)
                        segmented_files.append(segment_path)
                
                # Create marker to indicate this track has been segmented
                segment_marker.touch()
                
            except Exception as e:
                logger.error(f"Error creating segments from {track_path}: {e}")
        
        logger.info(f"Created {len(segmented_files)} segments from {len(music_files)} tracks")
        
    # ───────────────────────────────────────────────────────────────────────
    # AUGMENTATION - Apply various transformations to increase diversity
    # ───────────────────────────────────────────────────────────────────────
    augmented_files = []
    
    # Only augment for training
    if split_name == "train" and augment and (segmented_files or music_files):
        # Determine which files to augment (segmented if available, otherwise originals)
        source_files = segmented_files if segmented_files else music_files
        logger.info(f"Creating {augmentations_per_segment} augmentations for each of {len(source_files)} music files")
        
        for i, source_path in enumerate(tqdm(source_files, desc="Augmenting music files")):
            # Check if we've already augmented this file
            aug_marker = markers_dir / f"{source_path.stem}_augmented.marker"
            aug_files = list(augmented_dir.glob(f"{source_path.stem}_aug*.wav"))
            
            if aug_marker.exists() and len(aug_files) >= augmentations_per_segment:
                logger.info(f"Found {len(aug_files)} existing augmentations for {source_path.stem}")
                augmented_files.extend(aug_files)
                continue
                
            try:
                # Load the original audio
                audio, _ = librosa.load(source_path, sr=sample_rate)
                
                # Replace the augmentation loop in process_musdb with this version
                for aug_idx in range(augmentations_per_segment):
                    # Create a unique name for this augmentation
                    aug_name = f"{source_path.stem}_aug{aug_idx}.wav"
                    aug_path = augmented_dir / aug_name
                    
                    # Skip if already created
                    if aug_path.exists():
                        augmented_files.append(aug_path)
                        continue
                    
                    # Start with the original audio
                    augmented, success = safe_augment_audio(audio, sample_rate)
                    
                    # Only save if augmentation was successful
                    if success:
                        # Save the augmented sample
                        sf.write(aug_path, augmented, sample_rate)
                        augmented_files.append(aug_path)
                    else:
                        logger.warning(f"Skipping failed augmentation {aug_idx} for {source_path.stem}")
                
                # Mark that we've augmented this file
                aug_marker.touch()
                    
            except Exception as e:
                logger.error(f"Error augmenting {source_path.name}: {e}")
        
        logger.info(f"Created {len(augmented_files)} augmented music files")
    
    # ───────────────────────────────────────────────────────────────────────
    # MUSIC MIXTURES - Create combinations of different tracks
    # ───────────────────────────────────────────────────────────────────────
    mixed_files = []
    
    if split_name == "train" and augment and len(source_files) >= 2:
        # Only try to create mixes if we have at least 2 source files
        n_combinations = min(30, len(source_files) * 2)  # Limit number of combinations
        logger.info(f"Creating {n_combinations} mixed music combinations")
        
        # Check for existing mixtures
        mix_marker = markers_dir / f"music_mixes_{split_name}.marker"
        existing_mixes = list(augmented_dir.glob("mix_*.wav"))
        
        if mix_marker.exists() and existing_mixes:
            logger.info(f"Found {len(existing_mixes)} existing music mixtures")
            mixed_files.extend(existing_mixes)
        else:
            for mix_idx in range(n_combinations):
                try:
                    # Select two random source files
                    track1, track2 = random.sample(source_files, 2)
                    
                    # Create a unique name for this mixture
                    mix_name = f"mix_{track1.stem}_{track2.stem}_{mix_idx}.wav"
                    mix_path = augmented_dir / mix_name
                    
                    # Skip if already created
                    if mix_path.exists():
                        mixed_files.append(mix_path)
                        continue
                    
                    # Load audio
                    audio1, _ = librosa.load(track1, sr=sample_rate)
                    audio2, _ = librosa.load(track2, sr=sample_rate)
                    
                    # Ensure same length
                    min_len = min(len(audio1), len(audio2))
                    audio1 = audio1[:min_len]
                    audio2 = audio2[:min_len]
                    
                    # Apply random processing to one track sometimes
                    if random.random() < 0.5:
                        # Apply EQ to second track
                        audio2 = apply_eq(audio2, sample_rate)
                    
                    # Mix with random ratio
                    mix_ratio = random.uniform(0.4, 0.6)
                    mixed = audio1 * mix_ratio + audio2 * (1 - mix_ratio)
                    
                    # Normalize
                    mixed = mixed / (np.max(np.abs(mixed)) + 1e-7)
                    
                    # Save mixed track
                    sf.write(mix_path, mixed, sample_rate)
                    mixed_files.append(mix_path)
                    
                except Exception as e:
                    logger.error(f"Error creating music mixture {mix_idx}: {e}")
            
            # Mark mixtures as created
            mix_marker.touch()
            logger.info(f"Created {len(mixed_files)} mixed music files")
    
    # ───────────────────────────────────────────────────────────────────────
    # COMBINE ALL MUSIC FILES
    # ───────────────────────────────────────────────────────────────────────
    all_music = []
    all_music.extend(music_files)  # Original full tracks
    
    # For training, add all augmentations
    if split_name == "train" and augment:
        all_music.extend(segmented_files)   # Segments
        all_music.extend(augmented_files)   # Augmented versions
        all_music.extend(mixed_files)       # Mixed combinations
    
    logger.info(f"Final music dataset: {len(all_music)} files")
    logger.info(f"  - Original tracks: {len(music_files)}")
    if split_name == "train" and augment:
        logger.info(f"  - Segments: {len(segmented_files)}")
        logger.info(f"  - Augmented: {len(augmented_files)}")
        logger.info(f"  - Mixes: {len(mixed_files)}")

    # Create fallback if no files found
    if not all_music:
        logger.warning(f"No suitable MUSDB18HQ audio files found. Creating fallback audio.")
        fallback_path = musdb_root / f"fallback_audio_{split_name}_{sample_rate//1000}k.wav"

        # Create silent audio
        fallback_audio = np.zeros(sample_rate * 10)  # 10 seconds of silence
        sf.write(fallback_path, fallback_audio, sample_rate)
        all_music = [fallback_path]
        logger.info(f"Created fallback audio file at {fallback_path}")

    return all_music


def create_speech_music_sample_with_silero(
    speech_files, music_files, duration, sample_rate, win_length, hop_length, vad_model
):
    """
    Create a sample with speech over music background using Silero VAD for labels.

    Args:
        speech_files: List of speech audio files
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        vad_model: Silero VAD model

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    model, utils = vad_model
    # Select source files
    speech_path = random.choice(speech_files)
    music_path = random.choice(music_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)
    music = load_and_process_audio(music_path, sample_rate, duration)

    # Ensure same length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)
    music = librosa.util.fix_length(music, size=target_length)

    # Mix speech with music at varying SNR
    # Using lower SNR than speech+noise since music is less masking than noise
    snr_db = random.uniform(0, 25)  # Higher SNR range than with noise
    alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(music) + 1e-7)
    mix = speech + alpha * music
    mix = np.clip(mix, -1.0, 1.0)  # Normalize

    # Generate labels using Silero VAD
    vad_labels = generate_silero_vad_labels(
        mix,
        sample_rate,
        model,
        hop_length=hop_length,
        win_length=win_length,
        utils=utils,  # Pass utils as a named parameter
    )

    return mix, vad_labels


def create_overlapping_speech_with_silero(
    speech_files, noise_files, duration, sample_rate, win_length, hop_length, vad_model
):
    """
    Create an overlapping speech sample with Silero VAD labels.

    Args:
        speech_files: List of speech audio files
        noise_files: List of noise audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        vad_model: Silero VAD model

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    model, utils = vad_model
    # Select two different speech files
    speech1, speech2 = random.sample(speech_files, 2)

    # Load and process both speech files
    primary = load_and_process_audio(speech1, sample_rate, duration)
    secondary = load_and_process_audio(speech2, sample_rate, duration)

    # Make primary and secondary same length
    target_length = int(duration * sample_rate)
    primary = librosa.util.fix_length(primary, size=target_length)
    secondary = librosa.util.fix_length(secondary, size=target_length)

    # Make secondary speech quieter (foreground vs background speaker)
    snr_db = random.uniform(5, 15)  # Reasonable SNR for overlapping speech
    alpha = 10 ** (-snr_db / 20) * np.std(primary) / (np.std(secondary) + 1e-7)

    # Add noise too for realism
    noise_path = random.choice(noise_files)
    noise = load_and_process_audio(noise_path, sample_rate, duration)
    noise = librosa.util.fix_length(noise, size=target_length)

    # Create the final mixture
    mix = primary + alpha * secondary + 0.1 * noise
    mix = np.clip(mix, -1.0, 1.0)  # Normalize

    # Generate labels using Silero VAD
    vad_labels = generate_silero_vad_labels(
        mix,
        sample_rate,
        model,  # Just pass the unpacked model here
        hop_length=hop_length,
        win_length=win_length,
        utils=utils,  # Keep the utils as a named parameter
    )

    return mix, vad_labels


def create_speech_noise_music_sample_with_silero(
    speech_files,
    noise_files,
    music_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
    vad_model,
):
    """
    Create a sample with speech + background noise + music using Silero VAD for labels.

    Args:
        speech_files: List of speech audio files
        noise_files: List of noise audio files
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        vad_model: Tuple of (model, utils) for Silero VAD

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    model, utils = vad_model

    # Select source files
    speech_path = random.choice(speech_files)
    noise_path = random.choice(noise_files)
    music_path = random.choice(music_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)
    noise = load_and_process_audio(noise_path, sample_rate, duration)
    music = load_and_process_audio(music_path, sample_rate, duration)

    # Ensure same length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)
    noise = librosa.util.fix_length(noise, size=target_length)
    music = librosa.util.fix_length(music, size=target_length)

    # Mix with varying SNRs
    # Speech-to-noise ratio
    speech_noise_snr = random.uniform(-5, 20)
    noise_scaling = (
        10 ** (-speech_noise_snr / 20) * np.std(speech) / (np.std(noise) + 1e-7)
    )

    # Speech-to-music ratio (typically higher for better intelligibility)
    speech_music_snr = random.uniform(5, 25)
    music_scaling = (
        10 ** (-speech_music_snr / 20) * np.std(speech) / (np.std(music) + 1e-7)
    )

    # Create the mixture
    mix = speech + noise_scaling * noise + music_scaling * music
    mix = np.clip(mix, -1.0, 1.0)  # Normalize

    # Generate labels using Silero VAD
    vad_labels = generate_silero_vad_labels(
        mix,
        sample_rate,
        model,
        hop_length=hop_length,
        win_length=win_length,
        utils=utils,
    )

    return mix, vad_labels


def create_speech_noise_music_sample(
    speech_files,
    noise_files,
    music_files,
    duration,
    sample_rate,
    win_length,
    hop_length,
):
    """
    Create a sample with speech + background noise + music at varying SNRs.

    Args:
        speech_files: List of speech audio files
        noise_files: List of noise audio files
        music_files: List of music audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select source files
    speech_path = random.choice(speech_files)
    noise_path = random.choice(noise_files)
    music_path = random.choice(music_files)

    # Load audio
    speech = load_and_process_audio(speech_path, sample_rate, duration)
    noise = load_and_process_audio(noise_path, sample_rate, duration)
    music = load_and_process_audio(music_path, sample_rate, duration)

    # Ensure same length
    target_length = int(duration * sample_rate)
    speech = librosa.util.fix_length(speech, size=target_length)
    noise = librosa.util.fix_length(noise, size=target_length)
    music = librosa.util.fix_length(music, size=target_length)

    # Create labels from clean speech (before mixing)
    frame_energies = librosa.feature.rms(
        y=speech, frame_length=win_length, hop_length=hop_length
    )[0]

    threshold = adaptive_energy_threshold(frame_energies)
    vad_labels = (frame_energies > threshold).astype(np.float32)

    # Mix with varying SNRs
    # Speech-to-noise ratio
    speech_noise_snr = random.uniform(-5, 20)
    noise_scaling = (
        10 ** (-speech_noise_snr / 20) * np.std(speech) / (np.std(noise) + 1e-7)
    )

    # Speech-to-music ratio (typically higher for better intelligibility)
    speech_music_snr = random.uniform(5, 25)
    music_scaling = (
        10 ** (-speech_music_snr / 20) * np.std(speech) / (np.std(music) + 1e-7)
    )

    # Create the mixture
    mix = speech + noise_scaling * noise + music_scaling * music

    # Normalize
    mix = np.clip(mix, -1.0, 1.0)

    return mix, vad_labels

def stratified_librispeech_sample(libri_files, n_samples=10000):
    """Sample LibriSpeech files with stratification by speaker."""
    speakers = {}
    for libri in libri_files:
        # Extract speaker ID from path structure
        speaker = str(libri.parent.parent.name)
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(libri)
    
    # Calculate per-speaker quota
    n_speakers = len(speakers)
    per_speaker = max(1, n_samples // n_speakers)
    
    # Take samples from each speaker
    stratified_files = []
    for speaker_files in speakers.values():
        # Take up to per_speaker files from each speaker
        n_to_take = min(per_speaker, len(speaker_files))
        random.shuffle(speaker_files)  # Shuffle first
        stratified_files.extend(speaker_files[:n_to_take])
    
    # If we need more samples, take randomly from remaining files
    if len(stratified_files) < n_samples:
        remaining = [f for f in libri_files if f not in stratified_files]
        random.shuffle(remaining)
        stratified_files.extend(remaining[:n_samples-len(stratified_files)])
    
    # If we have too many, trim
    return stratified_files[:n_samples]

def stratified_fleurs_sample(fleurs_files, n_samples=10000):
    """Sample FLEURS files with stratification by language."""
    # Handle empty input case
    if not fleurs_files:
        logger.warning("Empty FLEURS file list provided to stratified sampling function")
        return []
        
    # Group files by language code
    languages = {}
    for file_path in fleurs_files:
        # Extract language code from filename (format: {lang_code}_{id}.wav)
        lang_code = file_path.stem.split('_')[0]
        if lang_code not in languages:
            languages[lang_code] = []
        languages[lang_code].append(file_path)
    
    # Calculate per-language quota
    n_languages = len(languages)
    logger.info(f"Found {n_languages} languages in FLEURS dataset")
    per_language = max(1, n_samples // n_languages)
    
    # Take samples from each language
    stratified_files = []
    for lang, files in languages.items():
        # Take up to per_language files from each language
        n_to_take = min(per_language, len(files))
        random.shuffle(files)  # Shuffle within language
        stratified_files.extend(files[:n_to_take])
        logger.info(f"Selected {n_to_take} samples from language '{lang}'")
    
    # If we need more samples to reach n_samples, add randomly
    if len(stratified_files) < n_samples:
        remaining = [f for f in fleurs_files if f not in stratified_files]
        random.shuffle(remaining)
        stratified_files.extend(remaining[:n_samples-len(stratified_files)])
    
    # If we have too many, trim
    return stratified_files[:n_samples]

def make_pos_neg(
    libri_root: pathlib.Path,
    musan_root: pathlib.Path,
    out_pos: pathlib.Path,
    out_neg: pathlib.Path,
    n_pos: int,
    n_neg: int,
    snr_range=(-5, 20),
    duration_range=(2, 15),
    sample_rate=16000,
    split_name="train",
    fleurs_langs=None,
    fleurs_streaming=False,
    speech_dir=None,
    neg_noise_ratio=0.20,
    neg_esc50_ratio=0.20,
    neg_music_ratio=0.20,
    neg_noise_noise_ratio=0.10,
    neg_music_music_ratio=0.10,
    neg_urbansound_ratio=0.20,
    vad_model=None,
):
    """
    Generate positive and negative audio samples with frame-level VAD labels.

    This function creates a complete dataset for training VAD models with:

    - Positive samples (containing speech):
      * Regular speech with background noise
      * Speech with gaps (silence between utterances)
      * Overlapping speech (multiple speakers)
      * Various augmentations (volume, reverb, EQ, SNR variations)

    - Negative samples (no speech):
      * Pure background noise
      * Music samples
      * Mixed noise and music

    All samples include frame-level VAD labels for training segment-level models.

    Args:
        libri_root: Path to LibriSpeech dataset directory
        musan_root: Path to MUSAN dataset directory
        out_pos: Output directory for positive samples
        out_neg: Output directory for negative samples
        n_pos: Target number of positive samples to generate
        n_neg: Target number of negative samples to generate
        snr_range: Range of signal-to-noise ratios (dB) for mixing
        duration_range: Min and max duration for generated samples (seconds)
        sample_rate: Audio sample rate (Hz)
        split_name: Dataset split name ('train', 'val', 'test')
        use_full_dataset: Whether to use all available source files
        fleurs_langs: Comma-separated list of FLEURS language codes
        fleurs_streaming: Whether to stream FLEURS dataset
        speech_dir: Directory for cached speech samples

    Returns:
        None: Files are saved to the specified directories
    """
    logger.info(
        f"Generating {n_pos} positive and {n_neg} negative samples for {split_name} split"
    )
    logger.info(f"Duration range: {duration_range}s, SNR range: {snr_range}dB")

    # Get speech files from both LibriSpeech and MUSAN
    libri = sorted(libri_root.rglob("*.flac"))
    logger.info(f"Found {len(libri)} LibriSpeech files, using random subset of 10,000")

    # Use deterministic shuffling
    random.seed(42)  # Keep consistent seed
    libri = stratified_librispeech_sample(libri, n_samples=10000)
    logger.info(f"Applied stratified sampling across {len(libri)} LibriSpeech files")

    musan_speech, musan_noise = process_musan(
        musan_root.parent,  # The parent of musan_root (should be datasets/)
        split_name=split_name,
        sample_rate=sample_rate,
    )
    # With this:
    # Swap MUSAN music for MUSDB HQ mixtures
    musdb_music = process_musdb(
        out_pos.parent.parent.parent,  # reuse same root
        split_name=split_name,
        sample_rate=sample_rate,
        cleanup_stems=True,                # Save disk space by removing stems
        augment=split_name=="train",       # Only augment training data
        segments_per_track=8,              # Extract 8 segments per track
        segment_length=10,                 # 10-second segments
        augmentations_per_segment=5,       # Create 5 variants per segment
    )
    logger.info(f"Added {len(musdb_music)} MUSDB HQ mixture files as music sources")

    # Add ESC-50 and VocalSet if requested
    esc50_noise = []
    vocalset_speech = []

    # Process ESC-50 for noise sources
    root_dir = out_pos.parent.parent
    esc50_noise = process_esc50(
        root_dir=root_dir.parent, split_name=split_name, sample_rate=sample_rate
    )
    logger.info(f"Added {len(esc50_noise)} ESC-50 noise files")

    # Process VocalSet for speech sources
    vocalset_speech = process_vocalset(
        root_dir=root_dir, split_name=split_name, sample_rate=sample_rate
    )
    logger.info(f"Added {len(vocalset_speech)} VocalSet speech files")

    urbansound_noise = []
    # Process UrbanSound8K for noise sources
    urbansound_noise = process_urbansound8k(
        root_dir=root_dir.parent, 
        split_name=split_name, 
        sample_rate=sample_rate
    )
    logger.info(f"Added {len(urbansound_noise)} UrbanSound8K noise files")

    # Combine noise sources
    all_noise_files = musan_noise + esc50_noise + urbansound_noise
    logger.info(f"Total noise files: {len(all_noise_files)}")

    # Add FLEURS data if provided
    fleurs_speech = []
    if fleurs_langs:
        logger.info(f"Adding FLEURS data for languages: {fleurs_langs}")
        langs = [l.strip().lower() for l in fleurs_langs.split(",")]
        speech_dir = speech_dir or out_pos.parent.parent / "fleurs_speech"

        max_per = {
            "train": MAX_PER_LANG_TRAIN,
            "val": MAX_PER_LANG_VAL,
            "test": MAX_PER_LANG_TEST,
        }[split_name]

        fleurs_speech = ingest_fleurs(
            lang_list=langs,
            out_dir=Path(speech_dir),
            sr=sample_rate,
            streaming=fleurs_streaming,
            split=split_name,
            max_per_lang=max_per,
            shuffle_seed=42,
            cache_dir=str(out_pos.parent.parent.parent / "hf_cache"),
        )
        logger.info(f"Added {len(fleurs_speech)} FLEURS speech files")
    elif speech_dir:
        # Use existing files in speech directory if specified
        speech_dir_path = Path(speech_dir)
        if speech_dir_path.exists():
            fleurs_speech = list(speech_dir_path.rglob("*.wav"))
            logger.info(
                f"Found {len(fleurs_speech)} existing speech files in {speech_dir}"
            )

    logger.info(
        f"Found {len(libri)} LibriSpeech files, {len(musan_speech)} MUSAN speech files, "
        f"{len(musan_noise)} noise files, {len(musdb_music)} music files, "
        f"{len(fleurs_speech)} FLEURS files"
    )
    assert libri and musan_noise and musdb_music, "Missing source audio!"

    # Create combined pool of speech files
    all_speech_files = libri + musan_speech + fleurs_speech + vocalset_speech
    logger.info(
        f"Total speech files available: {len(all_speech_files)} "
        f"(LibriSpeech: {len(libri)}, MUSAN: {len(musan_speech)}, "
        f"FLEURS: {len(fleurs_speech)}, VocalSet: {len(vocalset_speech)})"
    )

    # ── BALANCED SAMPLING ─────────────────────────────────────────────
    # Define the maximum possible positives based on available files
    actual_n_pos = min(
        n_pos,
        len(libri) + len(fleurs_speech) + len(musan_speech) + len(vocalset_speech),
    )

    # ------------------------------------------------------------------
    # Determine per-source quotas so they sum exactly to n_pos
    # ------------------------------------------------------------------
    source_weights = {
        "libri": 1.0,
        "fleurs": 1.0,
        "musan": 1.0,
        "vocalset": 1.0 if vocalset_speech else 0.0,
    }
    w_sum = sum(source_weights.values())
    q_libri = int(round(actual_n_pos * source_weights["libri"] / w_sum))
    q_fleurs = int(round(actual_n_pos * source_weights["fleurs"] / w_sum))
    q_musan = int(round(actual_n_pos * source_weights["musan"] / w_sum))
    q_vocalset = actual_n_pos - q_libri - q_fleurs - q_musan  # remainder guard

    # Shuffle each source before sampling
    libri_shuffled = libri.copy()
    fleurs_shuffled = stratified_fleurs_sample(fleurs_speech, n_samples=min(q_fleurs, len(fleurs_speech)))
    musan_shuffled = musan_speech.copy()
    vocalset_shuffled = vocalset_speech.copy()
    
    random.shuffle(libri_shuffled)
    random.shuffle(fleurs_shuffled)
    random.shuffle(musan_shuffled)
    random.shuffle(vocalset_shuffled)
    
    # Sample from each shuffled source based on quotas
    selected_libri = libri_shuffled[:min(q_libri, len(libri))]
    selected_fleurs = fleurs_shuffled[:min(q_fleurs, len(fleurs_speech))]
    selected_musan = musan_shuffled[:min(q_musan, len(musan_speech))]
    selected_vocalset = vocalset_shuffled[:min(q_vocalset, len(vocalset_speech))]

    # Combine all selected speech samples
    selected_speech = (
        selected_fleurs + selected_libri + selected_musan + selected_vocalset
    )
    random.shuffle(selected_speech)

    logger.info(
        f"Using balanced sample: {len(selected_fleurs)} FLEURS + {len(selected_libri)} LibriSpeech + "
        f"{len(selected_musan)} MUSAN + {len(selected_vocalset)} VocalSet = "
        f"{len(selected_speech)} total positives"
    )

    # Create directories for audio and frame-level labels
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)

    labels_root = out_pos.parent  # == datasets/prepared/<split>
    out_pos_labels = labels_root / "pos_labels"
    out_neg_labels = labels_root / "neg_labels"

    out_pos_labels.mkdir(parents=True, exist_ok=True)
    out_neg_labels.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Output directories created: {out_pos}, {out_neg}, {out_pos_labels}, {out_neg_labels}"
    )

    # Constants for frame-level calculations
    hop_length = DEFAULT_HOP_LENGTH
    win_length = DEFAULT_WIN_LENGTH

    # Start with clean speech samples as base
    logger.info(f"Generating {n_pos} clean speech samples...")
    clean_speech_start_time = time.time()
    generated_clean = 0
    clean_speech_failures = 0

    for idx in tqdm(range(n_pos), desc="Generating clean speech samples"):
        duration = random.uniform(*duration_range)

        if vad_model:
            audio, vad_labels = create_clean_speech_sample_with_silero(
                all_speech_files,
                duration,
                sample_rate,
                win_length,
                hop_length,
                vad_model,
            )
        else:
            audio, vad_labels = create_clean_speech_sample(
                all_speech_files, duration, sample_rate, win_length, hop_length
            )

        # Validate before saving
        if not validate_audio_sample(audio, sample_rate):
            clean_speech_failures += 1
            continue

        # Save the validated sample
        sf.write(out_pos / f"pos_clean_{idx:06}.wav", audio, sample_rate)
        np.save(out_pos_labels / f"pos_clean_{idx:06}_labels.npy", vad_labels)
        generated_clean += 1

        # Stop when we've reached our target
        if generated_clean >= n_pos:
            break

    clean_speech_time = time.time() - clean_speech_start_time
    logger.info(
        f"Completed {generated_clean} clean speech samples in {clean_speech_time:.2f}s "
        f"({clean_speech_failures} failed validation)"
    )

    # Calculate other sample counts as percentages of clean speech samples
    n_regular_samples = int(round(generated_clean * 0.30))  # Regular speech with noise
    n_overlap_samples = int(round(generated_clean * 0.20))  # Overlapping speech
    n_music_speech_samples = int(round(generated_clean * 0.25))  # Speech with music
    n_snm_samples = int(round(generated_clean * 0.25))  # Speech with noise and music

    # → REGULAR POSITIVE SAMPLES (speech + noise) with enhanced augmentation
    logger.info(f"Generating {n_regular_samples} regular speech+noise samples...")
    regular_start_time = time.time()
    generated_pos = 0
    failed_positives = 0

    for idx in tqdm(
        range(n_regular_samples), desc="Generating regular positive samples"
    ):
        # Random duration between duration_range
        duration = random.uniform(*duration_range)
        target_length = int(duration * sample_rate)

        # Decide pattern type: regular or speech-gap-speech (10% chance)
        use_speech_gap = random.random() < 0.1

        if use_speech_gap:
            # Create a speech-silence-speech pattern
            speech, vad_labels = create_speech_gap_speech(
                all_speech_files, duration, sample_rate, hop_length, win_length
            )
            clean_speech = speech.copy()
        else:
            # ----- STEP 1: LOAD AUDIO -----
            sp_path = random.choice(all_speech_files)
            full_speech, sr = librosa.load(sp_path, sr=sample_rate)

            # Apply random starting offset if audio is longer than needed
            if len(full_speech) > target_length:
                max_offset = len(full_speech) - target_length
                offset = random.randint(0, max_offset)
                speech = full_speech[offset : offset + target_length]
            else:
                # If too short, pad as needed
                speech = librosa.util.fix_length(full_speech, size=target_length)

            # ----- STEP 2: APPLY TIME/PITCH AUGMENTATIONS -----
            # Apply time stretching if random condition is met
            if random.random() > 0.5:
                stretch_rate = random.uniform(0.9, 1.1)
                speech = librosa.effects.time_stretch(speech, rate=stretch_rate)

            # Apply pitch shifting if random condition is met
            if random.random() > 0.5:
                speech = librosa.effects.pitch_shift(
                    speech, sr=sample_rate, n_steps=random.uniform(-3.5, 3.5)
                )

            # Fix length to match target after transformations
            speech = librosa.util.fix_length(speech, size=target_length)

            # Keep a clean copy for label generation (after transformations)
            clean_speech = speech.copy()

        # ----- STEP 3: APPLY VOLUME ADJUSTMENTS -----
        # Apply volume variations including ASMR-like quiet speech
        if random.random() < 0.2:  # 20% chance of very quiet speech
            volume_factor = random.uniform(0.05, 0.4)  # Very quiet (ASMR-like)
        else:
            volume_factor = random.uniform(0.7, 1.4)  # Normal range

        speech *= volume_factor

        # For very quiet ASMR-like audio, add small noise floor
        if volume_factor < 0.4:
            speech += 1e-4 * np.random.randn(*speech.shape)

        # Check if speech has vanished due to extreme low volume
        if np.std(speech) < 1e-4:  # speech vanished
            failed_positives += 1
            continue  # skip sample

        # ----- STEP 4: GENERATE VAD LABELS USING SILERO -----
        # Generate VAD labels from the clean speech before REVERB/EQ/NOISE
        # This is where we use Silero VAD!
        if vad_model and not use_speech_gap:  # Only use Silero if not a gap sample
            # Get utils from the model
            _, utils = vad_model  # Assuming vad_model is a tuple (model, utils)

            # Generate labels using Silero VAD
            vad_labels = generate_silero_vad_labels(
                clean_speech,
                sample_rate,
                vad_model[0],  # The actual model
                hop_length=hop_length,
                win_length=win_length,
                utils=utils,
            )
        elif not use_speech_gap:  # Fallback to energy-based if Silero not available
            # Calculate energy of each frame
            frame_energies = librosa.feature.rms(
                y=clean_speech, frame_length=win_length, hop_length=hop_length
            )[0]

            energy_threshold = adaptive_energy_threshold(frame_energies)
            vad_labels = (frame_energies > energy_threshold).astype(np.float32)

            # Smooth labels
            min_speech_frames = 3  # About 50ms at 16kHz with default hop_length
            for i in range(len(vad_labels) - min_speech_frames + 1):
                if 0 < sum(vad_labels[i : i + min_speech_frames]) < min_speech_frames:
                    vad_labels[i : i + min_speech_frames] = 0

        # ----- STEP 5: APPLY EFFECTS AND MIX WITH NOISE -----
        # Apply reverb with 30% probability
        apply_reverb_effect = random.random() < 0.3
        if apply_reverb_effect:
            speech = add_reverb(speech, sample_rate)

        # Apply EQ with 25% probability
        apply_eq_effect = random.random() < 0.25
        if apply_eq_effect:
            speech = apply_eq(speech, sample_rate)

        # Load background noise
        noise_path = random.choice(musan_noise)
        noise, _ = librosa.load(
            noise_path, sr=sample_rate, mono=True, duration=duration
        )
        noise = librosa.util.fix_length(noise, size=len(speech))

        # Apply SNR variation
        if random.random() < 0.15:  # 15% chance of extremely challenging SNR
            snr_db = random.uniform(-10, -3)  # Very challenging SNR
        else:
            snr_db = random.uniform(*snr_range)  # Normal range

        alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(noise) + 1e-7)
        mix = speech + alpha * noise

        if not validate_audio_sample(mix, sample_rate):
            failed_positives += 1
            continue

        # ----- STEP 6: SAVE THE SAMPLE -----
        # Determine source type and add descriptive tags
        source_type = "unknown"
        if str(sp_path).endswith(".flac"):
            source_type = "pos_libri"
        elif "musan/speech" in str(sp_path):
            source_type = "pos_musan"
        elif "VocalSet" in str(sp_path):
            source_type = "pos_vocalset"
        else:
            source_type = "pos_fleurs"

        # Add descriptive tags for special augmentations
        source_prefix = source_type
        if use_speech_gap:
            source_prefix += "_gap"
        if volume_factor < 0.4:
            source_prefix += "_whisper"
        if apply_reverb_effect:
            source_prefix += "_reverb"
        if snr_db < -3:
            source_prefix += "_lowsnr"

        # Save audio and frame-level labels
        mix = np.clip(mix, -1.0, 1.0)

        if not validate_audio_sample(mix, sample_rate):
            failed_positives += 1
            logger.info(f"Sample {idx} (positive) failed quality validation, skipping")
            continue

        sf.write(out_pos / f"{source_prefix}_{idx:06}.wav", mix, sample_rate)
        np.save(out_pos_labels / f"{source_prefix}_{idx:06}_labels.npy", vad_labels)
        generated_pos += 1

    # After all positive generation:
    # Measure actual positive samples generated (safer approach)
    generated_pos = len(list(out_pos.glob("pos_*.wav")))

    pos_time = time.time() - regular_start_time
    logger.info(
        f"Completed {generated_pos} positive samples in {pos_time:.2f}s ({pos_time/generated_pos:.3f}s per sample)"
    )

    # → OVERLAPPING SPEECH SAMPLES
    logger.info(f"Generating {n_overlap_samples} overlapping speech samples...")
    overlap_start_time = time.time()
    overlap_generated = 0
    overlap_failures = 0

    if n_overlap_samples > 0:
        logger.info(f"Generating {n_overlap_samples} overlapping speech samples...")
        overlap_start_time = time.time()

        for idx in tqdm(
            range(n_overlap_samples), desc="Generating overlapping speech samples"
        ):
            duration = random.uniform(*duration_range)

            # Use Silero VAD if available
            if vad_model:
                mix, vad_labels = create_overlapping_speech_with_silero(
                    all_speech_files,
                    musan_noise,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                    vad_model,
                )
            else:
                mix, vad_labels = create_overlapping_speech(
                    all_speech_files,
                    musan_noise,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                )

            # Validate before saving
            if not validate_audio_sample(mix, sample_rate):
                overlap_failures += 1
                continue

            # Save the validated sample
            sf.write(out_pos / f"pos_overlap_{idx:06}.wav", mix, sample_rate)
            np.save(out_pos_labels / f"pos_overlap_{idx:06}_labels.npy", vad_labels)
            overlap_generated += 1

        overlap_time = time.time() - overlap_start_time
        logger.info(
            f"Completed {overlap_generated} overlapping speech samples in {overlap_time:.2f}s "
            f"({overlap_failures} failed validation)"
        )

    # → SPEECH + MUSIC SAMPLES
    music_speech_generated = 0
    music_speech_failures = 0

    if n_music_speech_samples > 0:
        logger.info(f"Generating {n_music_speech_samples} speech+music samples...")
        music_speech_start_time = time.time()

        for idx in tqdm(
            range(n_music_speech_samples), desc="Generating speech+music samples"
        ):
            duration = random.uniform(*duration_range)

            # Use Silero VAD if available
            if vad_model:
                mix, vad_labels = create_speech_music_sample_with_silero(
                    all_speech_files,
                    musdb_music,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                    vad_model,
                )
            else:
                mix, vad_labels = create_speech_music_sample(
                    all_speech_files,
                    musdb_music,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                )

            # Validate before saving
            if not validate_audio_sample(mix, sample_rate):
                music_speech_failures += 1
                continue

            # Save the validated sample
            sf.write(out_pos / f"pos_music_{idx:06}.wav", mix, sample_rate)
            np.save(out_pos_labels / f"pos_music_{idx:06}_labels.npy", vad_labels)
            music_speech_generated += 1

        music_speech_time = time.time() - music_speech_start_time
        logger.info(
            f"Completed {music_speech_generated} speech+music samples in {music_speech_time:.2f}s "
            f"({music_speech_failures} failed validation)"
        )

    # → SPEECH + NOISE + MUSIC SAMPLES
    snm_generated = 0
    snm_failures = 0

    if n_snm_samples > 0:
        logger.info(f"Generating {n_snm_samples} speech+noise+music samples...")
        snm_start_time = time.time()

        for idx in tqdm(
            range(n_snm_samples), desc="Generating speech+noise+music samples"
        ):
            duration = random.uniform(*duration_range)

            # Use Silero VAD if available
            if vad_model:
                mix, vad_labels = create_speech_noise_music_sample_with_silero(
                    all_speech_files,
                    all_noise_files,
                    musdb_music,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                    vad_model,
                )
            else:
                mix, vad_labels = create_speech_noise_music_sample(
                    all_speech_files,
                    all_noise_files,
                    musdb_music,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                )

            # Validate before saving
            if not validate_audio_sample(mix, sample_rate):
                snm_failures += 1
                continue

            # Save the validated sample
            sf.write(out_pos / f"pos_snm_{idx:06}.wav", mix, sample_rate)
            np.save(out_pos_labels / f"pos_snm_{idx:06}_labels.npy", vad_labels)
            snm_generated += 1

        snm_time = time.time() - snm_start_time
        logger.info(
            f"Completed {snm_generated} speech+noise+music samples in {snm_time:.2f}s "
            f"({snm_failures} failed validation)"
        )

    # → NEGATIVE SAMPLES (noise only, music only, noise + music, music + music)
    logger.info(
        "Generating negative samples with frame-level labels (noise only, music only, noise + music)..."
    )
    start_time = time.time()

    # Use the provided n_neg or match the total positive count
    actual_n_neg = n_neg or (
        generated_clean  # Clean speech (base samples)
        + generated_pos  # Regular speech + noise
        + overlap_generated
        + music_speech_generated
        + snm_generated
    )

    # Compute per-type negative quotas once
    n_noise_only = int(round(actual_n_neg * neg_noise_ratio))
    n_esc50_only = int(round(actual_n_neg * neg_esc50_ratio))
    n_music_only = int(round(actual_n_neg * neg_music_ratio))
    n_noise_noise = int(round(actual_n_neg * neg_noise_noise_ratio))  # New quota
    n_music_music = int(round(actual_n_neg * neg_music_music_ratio))  # New quota
    n_urbansound = int(round(actual_n_neg * neg_urbansound_ratio))
    n_mixed = (
        actual_n_neg
        - n_noise_only
        - n_esc50_only
        - n_music_only
        - n_noise_noise
        - n_music_music
        - n_urbansound
    )  # Update remainder guard

    logger.info(f"Target negative samples: {actual_n_neg} total")
    logger.info(f"  - Noise-only: {n_noise_only}")
    logger.info(f"  - ESC-50: {n_esc50_only}")
    logger.info(f"  - Music-only: {n_music_only}")
    logger.info(f"  - Noise+noise: {n_noise_noise}")  # New log
    logger.info(f"  - Music+music: {n_music_music}")  # New log
    logger.info(f"  - Mixed noise+music: {n_mixed}")

    # 1. NOISE ONLY samples
    logger.info(f"Generating {n_noise_only} noise-only samples")
    noise_only_generated = 0
    noise_only_failures = 0

    for idx in tqdm(range(n_noise_only), desc="Generating noise-only samples"):
        duration = random.uniform(*duration_range)
        audio, vad_labels = create_negative_sample(
            all_noise_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            mix_ratio=0.5,
            category="noise",
        )

        # Validate before saving
        if not validate_audio_sample(audio, sample_rate):
            noise_only_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_noise_{idx:06}.wav", audio, sample_rate)
        np.save(out_neg_labels / f"neg_noise_{idx:06}_labels.npy", vad_labels)
        noise_only_generated += 1

    logger.info(
        f"Generated {noise_only_generated} noise-only samples ({noise_only_failures} failed validation)"
    )

    # 2. ESC-50 ONLY samples
    logger.info(f"Generating {n_esc50_only} ESC-50 samples")
    esc50_generated = 0
    esc50_failures = 0

    # Get balanced ESC-50 files
    balanced_esc50 = get_balanced_esc50_files(esc50_noise) if esc50_noise else []
    logger.info(f"Selected {len(balanced_esc50)} balanced ESC-50 files")

    for idx in tqdm(range(n_esc50_only), desc="Generating ESC-50 samples"):
        duration = random.uniform(*duration_range)

        # Generate sample using our dedicated function
        audio, vad_labels, category = create_esc50_negative_sample(
            balanced_esc50,
            duration,
            sample_rate,
            win_length,
            hop_length,
        )

        # Validate before saving
        if not validate_audio_sample(audio, sample_rate):
            esc50_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_esc50_{category}_{idx:06}.wav", audio, sample_rate)
        np.save(
            out_neg_labels / f"neg_esc50_{category}_{idx:06}_labels.npy", vad_labels
        )
        esc50_generated += 1

    logger.info(
        f"Generated {esc50_generated} ESC-50 samples ({esc50_failures} failed validation)"
    )

    # 3. MUSIC ONLY samples
    logger.info(f"Generating {n_music_only} music-only samples")
    music_only_generated = 0
    music_only_failures = 0
    for idx in tqdm(range(n_music_only), desc="Generating music-only samples"):
        duration = random.uniform(*duration_range)

        # Use our refactored function
        audio, vad_labels = create_negative_sample(
            musdb_music,
            duration,
            sample_rate,
            win_length,
            hop_length,
            mix_ratio=0.5,
            category="music",
        )

        # Validate before saving
        if not validate_audio_sample(audio, sample_rate):
            music_only_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_music_{idx:06}.wav", audio, sample_rate)
        np.save(out_neg_labels / f"neg_music_{idx:06}_labels.npy", vad_labels)
        music_only_generated += 1

    logger.info(
        f"Generated {music_only_generated} music-only samples ({music_only_failures} failed validation)"
    )

    # 4. NOISE + MUSIC samples
    logger.info(f"Generating {n_mixed} noise+music samples")
    mixed_generated = 0
    mixed_failures = 0
    for idx in tqdm(range(n_mixed), desc="Generating noise+music samples"):
        duration = random.uniform(*duration_range)

        # Use our new function
        mix, vad_labels = create_mixed_negative_sample(
            all_noise_files,
            musdb_music,
            duration,
            sample_rate,
            win_length,
            hop_length,
            mix_ratio=0.5,
        )

        # Validate before saving
        if not validate_audio_sample(mix, sample_rate):
            mixed_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_mixed_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_mixed_{idx:06}_labels.npy", vad_labels)
        mixed_generated += 1

    logger.info(
        f"Generated {mixed_generated} noise+music samples ({mixed_failures} failed validation)"
    )

    # 5. NOISE + NOISE samples
    logger.info(f"Generating {n_noise_noise} noise+noise samples")
    noise_noise_generated = 0
    noise_noise_failures = 0

    for idx in tqdm(range(n_noise_noise), desc="Generating noise+noise samples"):
        duration = random.uniform(*duration_range)
        mix, vad_labels = create_noise_noise_sample(
            all_noise_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
        )

        # Validate before saving
        if not validate_audio_sample(mix, sample_rate):
            noise_noise_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_noise_noise_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_noise_noise_{idx:06}_labels.npy", vad_labels)
        noise_noise_generated += 1

    logger.info(
        f"Generated {noise_noise_generated} noise+noise samples ({noise_noise_failures} failed validation)"
    )

    # 6. MUSIC + MUSIC samples
    logger.info(f"Generating {n_music_music} music+music samples")
    music_music_generated = 0
    music_music_failures = 0

    for idx in tqdm(range(n_music_music), desc="Generating music+music samples"):
        duration = random.uniform(*duration_range)
        mix, vad_labels = create_music_music_sample(
            musdb_music,
            duration,
            sample_rate,
            win_length,
            hop_length,
        )

        # Validate before saving
        if not validate_audio_sample(mix, sample_rate):
            music_music_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_music_music_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_music_music_{idx:06}_labels.npy", vad_labels)
        music_music_generated += 1

    logger.info(
        f"Generated {music_music_generated} music+music samples ({music_music_failures} failed validation)"
    )

    # 7. Generate UrbanSound8K samples
    if n_urbansound > 0:
        logger.info(f"Generating {n_urbansound} UrbanSound8K samples")
        urbansound_generated = 0
        urbansound_failures = 0

        for idx in tqdm(range(n_urbansound), desc="Generating UrbanSound8K samples"):
            duration = random.uniform(*duration_range)
            audio, vad_labels = create_negative_sample(
                urbansound_noise,
                duration,
                sample_rate,
                win_length,
                hop_length,
                category="urbansound"
            )

            # Validate before saving
            if not validate_audio_sample(audio, sample_rate):
                urbansound_failures += 1
                continue

            # Save the validated sample
            sf.write(out_neg / f"neg_urbansound_{idx:06}.wav", audio, sample_rate)
            np.save(out_neg_labels / f"neg_urbansound_{idx:06}_labels.npy", vad_labels)
            urbansound_generated += 1

        logger.info(f"Generated {urbansound_generated} UrbanSound8K samples ({urbansound_failures} failed validation)")

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {actual_n_neg} negative samples in {neg_time:.2f}s ({neg_time/actual_n_neg:.3f}s per sample)"
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────
    summary = {
        "positives_clean": generated_clean,
        "positives_regular": generated_pos,
        "positives_overlap": overlap_generated,
        "positives_music": music_speech_generated,
        "positives_snm": snm_generated,
        "neg_noise": noise_only_generated,
        "neg_esc50": esc50_generated,
        "neg_music": music_only_generated,
        "neg_noise_noise": noise_noise_generated,  # New metric
        "neg_music_music": music_music_generated,  # New metric
        "neg_mixed": mixed_generated,
    }

    total_ok = sum(summary.values())
    total_bad = (
        failed_positives
        + overlap_failures
        + music_speech_failures
        + clean_speech_failures
        + snm_failures
        + noise_only_failures
        + esc50_failures
        + music_only_failures
        + noise_noise_failures  # New failure count
        + music_music_failures  # New failure count
        + mixed_failures
    )

    logger.info("========== DATA-GEN SUMMARY ==========")
    for k, v in summary.items():
        logger.info(f"{k.replace('_', ' ').title():<23}: {v:7d}")
    logger.info("--------------------------------------")
    logger.info(f"TOTAL SAVED          : {total_ok}")
    logger.info(f"TOTAL FAILED         : {total_bad}")
    logger.info(f"SUCCESS RATE         : {100*total_ok/(total_ok+total_bad):.2f}%")


def create_manifest(
    prep_dir: pathlib.Path, manifest_path: pathlib.Path, split_name: str
):
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
            # Detect if this is a positive sample by checking for "pos_" prefix
            is_speech = "pos_" in p.name

            # now looks inside the split folder
            label_dir = "pos_labels" if is_speech else "neg_labels"
            frame_label_path = prep_dir / label_dir / f"{p.stem}_labels.npy"

            w.writerow([p, 1 if is_speech else 0, frame_label_path])

    logger.info(f"Created {split_name} manifest with {len(all_clips)} entries")
    return manifest_path


def prepare_dataset(
    root_dir: str,
    split: str = "train",
    n_pos: int = 2000,
    n_neg: int = 2000,
    duration_range: tuple = (5, 10),
    sample_rate: int = 16000,
    force_rebuild: bool = False,
    fleurs_langs: str = None,
    fleurs_streaming: bool = False,
    neg_noise_ratio=0.20,
    neg_esc50_ratio=0.20,
    neg_music_ratio=0.20,
    neg_noise_noise_ratio=0.10,
    neg_music_music_ratio=0.10,
    neg_urbansound_ratio=0.20,
    use_silero_vad: bool = True,
):
    """
    Prepare dataset for a specific split (train, val, test).

    Args:
        root_dir: Root directory for all datasets and outputs
        split: Dataset split (train, val, test)
        n_pos: Number of positive samples to generate
        n_neg: Number of negative samples to generate
        duration_range: Min and max duration in seconds for audio clips
        sample_rate: Sample rate for audio processing
        force_rebuild: Force rebuilding the dataset even if it exists

    Returns:
        Path to the created manifest file
    """
    logger.info(f"Preparing {split} dataset")
    root = pathlib.Path(root_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    # Track state in a JSON file
    state_file = root / f"preparation_state_{split}.json"
    state = {}
    if state_file.exists() and not force_rebuild:
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            logger.info(f"Found existing preparation state: {state}")
        except Exception as e:
            logger.error(f"Error loading state file: {e}")

    # Download and extract data
    if not state.get("downloads_complete", False) or force_rebuild:
        logger.info(f"Beginning download of necessary datasets for {split}")

        # Determine which datasets to download based on split
        if split == "test":
            # For test, we need test-clean
            download_and_extract(TEST_CLEAN_URL, root)
            download_and_extract(MUSAN_URL, root)
            download_and_extract_zip(ESC_50_URL, root)
            download_and_extract_zip(VOCALSET_URL, root)
            download_and_extract_urbansound8k(URBANSOUND8K_URL, root)
        elif split == "val":
            # For validation, we need dev-clean
            download_and_extract(VAL_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)
            download_and_extract_zip(ESC_50_URL, root)
            download_and_extract_zip(VOCALSET_URL, root)
            download_and_extract_urbansound8k(URBANSOUND8K_URL, root)
        else:  # train split
            # For training, we need train-clean-360
            download_and_extract(TRAIN_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)
            download_and_extract_zip(ESC_50_URL, root)
            download_and_extract_zip(VOCALSET_URL, root)
            download_and_extract_urbansound8k(URBANSOUND8K_URL, root)

        state["downloads_complete"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)

    # Generate samples if not already done
    if not state.get("samples_generated", False) or force_rebuild:
        logger.info(f"Generating positive and negative audio samples for {split}")

        if split == "test":
            libri_root = root / "LibriSpeech" / "test-clean"
        elif split == "val":
            libri_root = root / "LibriSpeech" / "dev-clean"
        else:  # train split
            libri_root = root / "LibriSpeech" / "train-clean-360"

        musan_root = root / "musan"
        prep = root / "prepared" / split

        # Create output directories
        pos_dir, neg_dir = prep / "pos", prep / "neg"

        vad_model = None
        if use_silero_vad:
            if sample_rate != 16000:
                logger.error(
                    "Silero VAD requires 16 kHz audio. Set sample_rate=16000 or disable use_silero_vad."
                )
                raise ValueError("Silero VAD requires 16 kHz audio")

            logger.info("Initializing Silero VAD model for improved label generation")
            model, utils = initialize_silero_vad()
            vad_model = (model, utils)  # Pack both model and utils

        make_pos_neg(
            libri_root,
            musan_root,
            pos_dir,
            neg_dir,
            n_pos,
            n_neg,
            duration_range=duration_range,
            sample_rate=sample_rate,
            split_name=split,
            fleurs_langs=fleurs_langs,
            fleurs_streaming=fleurs_streaming,
            speech_dir=root / "fleurs_speech",
            neg_noise_ratio=neg_noise_ratio,
            neg_esc50_ratio=neg_esc50_ratio,
            neg_music_ratio=neg_music_ratio,
            neg_noise_noise_ratio=neg_noise_noise_ratio,
            neg_music_music_ratio=neg_music_music_ratio,
            neg_urbansound_ratio=neg_urbansound_ratio,
            vad_model=vad_model,
        )

        state["samples_generated"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)

    # Create manifest if not already done
    manifest_path = root / f"manifest_{split}.csv"
    if (
        not state.get("manifest_created", False)
        or force_rebuild
        or not manifest_path.exists()
    ):
        logger.info(f"Creating {split} manifest")
        create_manifest(root / "prepared" / split, manifest_path, split)
        state["manifest_created"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)

    logger.info(f"✅ {split.capitalize()} data preparation completed")
    print(f"✅ {split.capitalize()} data ready")

    return manifest_path


def main(args=None):
    parser = argparse.ArgumentParser("Data Preparation for MEL-spectrogram VAD")

    if args is None or not hasattr(args, "root"):
        # Parse arguments when script is called directly
        parser = argparse.ArgumentParser("Data Preparation for MEL-spectrogram VAD")

        # Base arguments
        parser.add_argument(
            "--root",
            required=True,
            help="Root directory for all datasets and outputs",
            default=DEFAULT_ROOT_DIR,
        )
        parser.add_argument(
            "--splits",
            nargs="+",
            default=["train", "val", "test"],
            choices=["train", "val", "test"],
            help="Dataset splits to prepare (train, val, and/or test)",
        )
        parser.add_argument(
            "--n_pos",
            type=int,
            default=2000,
            help="Number of positive samples per split",
        )
        parser.add_argument(
            "--n_neg",
            type=int,
            default=2000,
            help="Number of negative samples per split",
        )
        parser.add_argument(
            "--duration_range",
            type=float,
            nargs=2,
            default=[1, 30],
            help="Min and max duration in seconds for audio clips",
        )
        parser.add_argument(
            "--sample_rate",
            type=int,
            default=16000,
            help="Sample rate for audio processing",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Force rebuild even if data already exists",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Random seed for reproducibility"
        )
        parser.add_argument(
            "--fleurs_langs",
            default="id_id,yue_hant_hk,ka_ge,xh_za,yo_ng,ar_eg,hi_in,ta_in,vi_vn,mi_nz",
            help=(
                "Comma‑separated list of FLEURS language configs. "
                "Valid options include: 'id_id', 'ja_jp', 'en_us', 'ar_eg', 'hi_in', etc. "
                "Use 'all' for all 102 languages."
            ),
        )
        parser.add_argument(
            "--fleurs_streaming",
            default=False,
            action="store_true",
            help="Stream FLEURS instead of downloading it (saves disk but no random access).",
        )
        parser.add_argument(
            "--neg_noise_ratio",
            type=float,
            default=0.2,
            help="Fraction of negative samples that are pure noise",
        )
        parser.add_argument(
            "--neg_esc50_ratio",
            type=float,
            default=0.2,
            help="Fraction of negative samples that are ESC-50 sounds",
        )
        parser.add_argument(
            "--neg_music_ratio",
            type=float,
            default=0.2,
            help="Fraction of negative samples that are music",
        )
        parser.add_argument(
            "--neg_noise_noise_ratio",
            type=float,
            default=0.1,
            help="Fraction of negative samples that are noise+noise combinations",
        )
        parser.add_argument(
            "--neg_music_music_ratio",
            type=float,
            default=0.1,
            help="Fraction of negative samples that are music+music combinations",
        )
        parser.add_argument(
            "--neg_urbansound_ratio",
            type=float,
            default=0.2,
            help="Fraction of negative samples that are UrbanSound8K sounds"
        )
        parser.add_argument(
            "--use_silero_vad",
            action="store_true",
            default=True,
            help="Use Silero VAD for generating frame-level labels",
        )

        args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

    # Replace the warning with this auto-adjustment code
    neg_sum = (
        args.neg_noise_ratio
        + args.neg_esc50_ratio
        + args.neg_music_ratio
        + args.neg_noise_noise_ratio
        + args.neg_music_music_ratio
        + args.neg_urbansound_ratio
    )
    if abs(neg_sum - 1.0) > 1e-6:  # If not approximately 1.0
        logger.warning(
            f"Negative sample ratios sum to {neg_sum}, which differs from 1.0. "
            f"Auto-adjusting to ensure they sum to 1.0."
        )
        # Scale each ratio to make them sum to 1.0
        scaling_factor = 1.0 / neg_sum
        args.neg_noise_ratio *= scaling_factor
        args.neg_esc50_ratio *= scaling_factor
        args.neg_music_ratio *= scaling_factor
        args.neg_noise_noise_ratio *= scaling_factor
        args.neg_music_music_ratio *= scaling_factor
        args.neg_urbansound_ratio *= scaling_factor
        logger.info(
            f"Adjusted ratios: noise={args.neg_noise_ratio:.3f}, "
            f"esc50={args.neg_esc50_ratio:.3f}, music={args.neg_music_ratio:.3f}, "
            f"noise_noise={args.neg_noise_noise_ratio:.3f}, "
            f"music_music={args.neg_music_music_ratio:.3f}"
            f"urbansound={args.neg_urbansound_ratio:.3f}"
        )

    # Process each requested split
    for split in args.splits:
        print(f"\n🔄 Preparing {split} split...")
        manifest_path = prepare_dataset(
            args.root,
            split=split,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            duration_range=args.duration_range,
            sample_rate=args.sample_rate,
            force_rebuild=args.force if hasattr(args, "force") else False,
            fleurs_langs=args.fleurs_langs if hasattr(args, "fleurs_langs") else None,
            fleurs_streaming=(
                args.fleurs_streaming if hasattr(args, "fleurs_streaming") else False
            ),
            neg_noise_ratio=(
                args.neg_noise_ratio if hasattr(args, "neg_noise_ratio") else 0.20
            ),
            neg_esc50_ratio=(
                args.neg_esc50_ratio if hasattr(args, "neg_esc50_ratio") else 0.20
            ),
            neg_music_ratio=(
                args.neg_music_ratio if hasattr(args, "neg_music_ratio") else 0.20
            ),
            neg_noise_noise_ratio=(
                args.neg_noise_noise_ratio
                if hasattr(args, "neg_noise_noise_ratio")
                else 0.10
            ),
            neg_music_music_ratio=(
                args.neg_music_music_ratio
                if hasattr(args, "neg_music_music_ratio")
                else 0.10
            ),
            neg_urbansound_ratio=(
                args.neg_urbansound_ratio
                if hasattr(args, "neg_urbansound_ratio")
                else 0.20
            ),
            use_silero_vad=(
                args.use_silero_vad if hasattr(args, "use_silero_vad") else False
            ),
        )
        print(f"✅ Created manifest: {manifest_path}")


if __name__ == "__main__":
    main()
