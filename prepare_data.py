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
TRAIN_LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
VAL_LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
ESC_50_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"
VOCALSET_URL = "https://zenodo.org/records/1193957/files/VocalSet.zip?download=1"

FRACTION_FLEURS = 0.50  # 50% of positives from FLEURS
FRACTION_LIBRI = 0.30  # 30% of positives from LibriSpeech
FRACTION_MUSAN = 0.20  # 20% of positives from MUSAN speech
MAX_PER_LANG_TRAIN = 1000  # FLEURS gives ≈1000 train clips per language
MAX_PER_LANG_VAL = 400  # idem for dev
MAX_PER_LANG_TEST = 400


# ───────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ───────────────────────────────────────────────────────────────────────


def initialize_silero_vad(
    model_path: Optional[str] = None, force_reload: bool = False, device: str = None
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Initialize the Silero VAD model.
    """
    logger.info("Initializing Silero VAD model")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    # Convert audio to tensor and ensure correct format
    if not torch.is_tensor(audio):
        audio_tensor = torch.FloatTensor(audio)
    else:
        audio_tensor = audio

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
        # Fallback to hub's function
        speech_timestamps = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            verbose=False,
            utils_only=True,
        )["get_speech_timestamps"](
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
    root_dir: pathlib.Path, sample_rate: int = 16000
) -> list[pathlib.Path]:
    """
    Process VocalSet dataset and return paths to singing audio files.

    Args:
        root_dir: Root directory where dataset is extracted
        sample_rate: Target sample rate

    Returns:
        List of paths to singing/vocal audio files
    """
    # Try to find VocalSet directory with a more flexible pattern
    vocalset_dirs = list(root_dir.glob("*FULL*"))
    if not vocalset_dirs:
        logger.warning(f"VocalSet directory not found in {root_dir}")
        return []

    vocalset_dir = vocalset_dirs[0]
    logger.info(f"Processing VocalSet dataset from {vocalset_dir}")

    # Find all WAV files
    vocal_files = list(vocalset_dir.rglob("*.wav"))
    logger.info(f"Found {len(vocal_files)} VocalSet audio files")

    return vocal_files


def process_esc50(
    root_dir: pathlib.Path, sample_rate: int = 16000
) -> list[pathlib.Path]:
    """
    Process ESC-50 dataset and return paths to usable audio files for noise sources.

    Args:
        root_dir: Root directory where dataset is extracted
        sample_rate: Target sample rate

    Returns:
        List of paths to noise audio files
    """
    # Try to find the ESC-50 directory with a more flexible pattern
    esc50_dirs = list(root_dir.glob("*ESC-50*"))
    if not esc50_dirs:
        logger.warning(f"ESC-50 directory not found in {root_dir}")
        return []

    esc50_dir = esc50_dirs[0]
    logger.info(f"Found ESC-50 directory at {esc50_dir}")

    # ESC-50 audio is in the audio/ directory
    audio_dir = esc50_dir / "audio"
    if not audio_dir.exists():
        # Try to find audio directory with a recursive search
        audio_dirs = list(esc50_dir.rglob("audio"))
        if audio_dirs:
            audio_dir = audio_dirs[0]
            logger.info(f"Found audio directory at {audio_dir}")
        else:
            logger.warning(f"ESC-50 audio directory not found at {esc50_dir}")
            return []

    logger.info(f"Processing ESC-50 dataset from {audio_dir}")

    # Get all WAV files
    all_files = list(audio_dir.glob("*.wav"))
    logger.info(f"Found {len(all_files)} ESC-50 audio files")

    # Filter out human vocal sounds to avoid false positives in VAD
    # Categories to exclude (based on ESC-50 metadata):
    # - Category 0: "human non-speech sounds" (coughing, sneezing, etc.)
    # - Category 5: "human speech"
    exclude_prefixes = ["0", "5"]

    noise_files = []
    for file in all_files:
        # ESC-50 filename format: {category}-{fold}-{ID}-{take}.wav
        category = file.stem.split("-")[0]
        if not any(category.startswith(prefix) for prefix in exclude_prefixes):
            noise_files.append(file)

    logger.info(f"Selected {len(noise_files)} ESC-50 audio files as noise sources")
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
):
    """
    Create a negative sample (noise, music, or mixed).

    Args:
        source_files: List of source audio files
        duration: Target duration in seconds
        sample_rate: Audio sample rate
        win_length: Window length for frame analysis
        hop_length: Hop length for frame analysis
        mix_ratio: Ratio for mixing second file
        category: Type of negative sample ('noise', 'music', or 'mixed')

    Returns:
        Tuple of (mixed audio, VAD labels)
    """
    # Select source files
    file1, file2 = random.sample(source_files, 2)

    # Load audio
    audio1 = load_and_process_audio(file1, sample_rate, duration)
    audio2 = load_and_process_audio(file2, sample_rate, duration)

    # Mix with the specified ratio
    target_length = int(duration * sample_rate)
    audio1 = librosa.util.fix_length(audio1, size=target_length)
    audio2 = librosa.util.fix_length(audio2, size=target_length)
    mix = audio1 + mix_ratio * audio2

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

    Args:
        audio: Input audio signal
        sr: Sample rate

    Returns:
        Equalized audio
    """
    # Original audio
    eq_audio = audio.copy()

    # Apply 2-3 random frequency adjustments
    for _ in range(random.randint(2, 3)):
        # Random center frequency
        center_freq = random.uniform(100, 7000)
        q = random.uniform(0.5, 5.0)
        gain_db = random.uniform(-10, 10)  # dB

        # Create and apply peaking filter
        b, a = signal.iirpeak(center_freq / (sr / 2), q, gain_db)
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


def process_musdb(
    root_dir: pathlib.Path,
    subset: str = "train",  # or "test"
    sample_rate: int = 16000,
) -> list[pathlib.Path]:
    """
    Download ex MUSDB HQ and return local paths to mixture WAVs
    resampled to the target sample_rate (written once, cached later).
    """
    musdb_root = root_dir / "MUSDB_HQ"
    musdb_root.mkdir(exist_ok=True, parents=True)

    ds = MUSDB_HQ(
        root=musdb_root,
        subset=subset,
        sources=["bass", "drums", "other"],
        download=True,
    )

    mixture_paths: list[pathlib.Path] = []
    for waveform, sr, _, track_name in ds:
        out = musdb_root / f"{track_name}_mixture_{sample_rate//1000}k.wav"
        if not out.exists():  # resample & cache
            if sr != sample_rate:
                import torchaudio.functional as F

                waveform = F.resample(waveform, sr, sample_rate)
            torchaudio.save(out, waveform, sample_rate)
        mixture_paths.append(out)
    return mixture_paths


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
    use_full_dataset=False,
    fleurs_langs=None,
    fleurs_streaming=False,
    speech_dir=None,
    use_additional_datasets=True,  # Add this parameter
    neg_noise_ratio=0.25,
    neg_esc50_ratio=0.25,
    neg_music_ratio=0.25,
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
    musan_speech = list(musan_root.joinpath("speech").rglob("*.wav"))
    musan_noise = list(musan_root.joinpath("noise").rglob("*.wav"))
    # With this:
    # Swap MUSAN music for MUSDB HQ mixtures
    musdb_music = process_musdb(
        out_pos.parent.parent,  # reuse same root
        subset="train" if split_name == "train" else "test",
        sample_rate=sample_rate,
    )
    logger.info(f"Added {len(musdb_music)} MUSDB HQ mixture files as music sources")

    # Add ESC-50 and VocalSet if requested
    esc50_noise = []
    vocalset_speech = []

    if use_additional_datasets:
        # Process ESC-50 for noise sources
        root_dir = out_pos.parent.parent
        esc50_noise = process_esc50(root_dir, sample_rate)
        logger.info(f"Added {len(esc50_noise)} ESC-50 noise files")

        # Process VocalSet for speech sources
        vocalset_speech = process_vocalset(root_dir, sample_rate)
        logger.info(f"Added {len(vocalset_speech)} VocalSet speech files")

    # Combine noise sources
    all_noise_files = musan_noise + esc50_noise
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
            cache_dir=str(out_pos.parent.parent / "hf_cache"),
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
    if use_full_dataset:
        logger.info("Using full dataset mode - will use all available speech files")
        # Use all available files
        selected_fleurs = fleurs_speech
        selected_libri = libri
        selected_musan = musan_speech
        selected_vocalset = vocalset_speech
    else:
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

        # Sample from each source based on quotas
        selected_libri = random.sample(libri, min(q_libri, len(libri)))
        selected_fleurs = random.sample(
            fleurs_speech, min(q_fleurs, len(fleurs_speech))
        )
        selected_musan = random.sample(musan_speech, min(q_musan, len(musan_speech)))
        selected_vocalset = random.sample(
            vocalset_speech, min(q_vocalset, len(vocalset_speech))
        )

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
    out_pos_labels = out_pos.parent.parent / f"{split_name}_pos_labels"
    out_neg_labels = out_neg.parent.parent / f"{split_name}_neg_labels"
    out_pos_labels.mkdir(parents=True, exist_ok=True)
    out_neg_labels.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Output directories created: {out_pos}, {out_neg}, {out_pos_labels}, {out_neg_labels}"
    )

    # Constants for frame-level calculations
    hop_length = DEFAULT_HOP_LENGTH
    win_length = DEFAULT_WIN_LENGTH

    # → POSITIVE SAMPLES (speech + noise) with enhanced augmentation
    logger.info(
        "Generating positive samples (speech + noise) with enhanced augmentation..."
    )

    start_time = time.time()
    generated_pos = 0
    failed_positives = 0

    for idx, sp_path in enumerate(
        tqdm(selected_speech, desc="Generating positive samples")
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

    pos_time = time.time() - start_time
    logger.info(
        f"Completed {generated_pos} positive samples in {pos_time:.2f}s ({pos_time/generated_pos:.3f}s per sample)"
    )

    # → OVERLAPPING SPEECH SAMPLES (more challenging positives)
    # Calculate sample count based on fixed percentage of actual positives
    n_overlap_samples = int(round(generated_pos * 0.20))  # 20% of actual positives

    # Replace the existing overlapping speech generation loop:
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
    n_music_speech_samples = int(round(generated_pos * 0.25))  # 25% of actual positives
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

    # → CLEAN SPEECH SAMPLES (no added noise or effects)
    n_clean_speech_samples = int(round(generated_pos * 0.15))  # 15% of actual positives
    clean_speech_generated = 0
    clean_speech_failures = 0

    if n_clean_speech_samples > 0:
        logger.info(f"Generating {n_clean_speech_samples} clean speech samples...")
        clean_speech_start_time = time.time()

        for idx in tqdm(
            range(n_clean_speech_samples), desc="Generating clean speech samples"
        ):
            duration = random.uniform(*duration_range)

            if vad_model:
                # Use Silero-based functions
                audio, vad_labels = create_clean_speech_sample_with_silero(
                    all_speech_files,
                    duration,
                    sample_rate,
                    win_length,
                    hop_length,
                    vad_model,
                )
            else:
                # Use current energy-based method
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
            clean_speech_generated += 1

        clean_speech_time = time.time() - clean_speech_start_time
        logger.info(
            f"Completed {clean_speech_generated} clean speech samples in {clean_speech_time:.2f}s "
            f"({clean_speech_failures} failed validation)"
        )

    # → NEGATIVE SAMPLES (noise only, music only, noise + music)
    logger.info(
        "Generating negative samples with frame-level labels (noise only, music only, noise + music)..."
    )
    start_time = time.time()

    # Use the provided n_neg or match the total positive count
    actual_n_neg = n_neg or (
        generated_pos
        + overlap_generated
        + music_speech_generated
        + clean_speech_generated
    )

    # Compute per-type negative quotas once
    n_noise_only = int(round(actual_n_neg * neg_noise_ratio))
    n_esc50_only = int(round(actual_n_neg * neg_esc50_ratio))
    n_music_only = int(round(actual_n_neg * neg_music_ratio))
    n_mixed = (
        actual_n_neg - n_noise_only - n_esc50_only - n_music_only
    )  # remainder guard

    logger.info(f"Target negative samples: {actual_n_neg} total")
    logger.info(f"  - Noise-only: {n_noise_only}")
    logger.info(f"  - ESC-50: {n_esc50_only}")
    logger.info(f"  - Music-only: {n_music_only}")
    logger.info(f"  - Mixed noise+music: {n_mixed}")

    # 1. NOISE ONLY samples
    logger.info(f"Generating {n_noise_only} noise-only samples")
    noise_only_generated = 0
    noise_only_failures = 0

    for idx in tqdm(range(n_noise_only), desc="Generating noise-only samples"):
        duration = random.uniform(*duration_range)
        mix, vad_labels = create_negative_sample(
            all_noise_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            mix_ratio=0.5,
            category="noise",
        )

        # Validate before saving
        if not validate_audio_sample(mix, sample_rate):
            noise_only_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_noise_{idx:06}.wav", mix, sample_rate)
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
        mix, vad_labels = create_negative_sample(
            musdb_music,
            duration,
            sample_rate,
            win_length,
            hop_length,
            mix_ratio=0.5,
            category="music",
        )

        # Validate before saving
        if not validate_audio_sample(mix, sample_rate):
            music_only_failures += 1
            continue

        # Save the validated sample
        sf.write(out_neg / f"neg_music_{idx:06}.wav", mix, sample_rate)
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

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {actual_n_neg} negative samples in {neg_time:.2f}s ({neg_time/actual_n_neg:.3f}s per sample)"
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────
    summary = {
        "positives_regular": generated_pos,
        "positives_overlap": overlap_generated,
        "positives_music": music_speech_generated,
        "positives_clean": clean_speech_generated,
        "neg_noise": noise_only_generated,
        "neg_esc50": esc50_generated,
        "neg_music": music_only_generated,
        "neg_mixed": mixed_generated,
    }

    total_ok = sum(summary.values())
    total_bad = (
        failed_positives
        + overlap_failures
        + music_speech_failures
        + clean_speech_failures
        + noise_only_failures
        + esc50_failures
        + music_only_failures
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

            # Path to corresponding frame labels
            label_dir = (
                f"{split_name}_pos_labels" if is_speech else f"{split_name}_neg_labels"
            )
            frame_label_path = prep_dir.parent / label_dir / f"{p.stem}_labels.npy"
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
    use_full_dataset: bool = False,
    fleurs_langs: str = None,
    fleurs_streaming: bool = False,
    use_additional_datasets: bool = True,  # Add this parameter
    neg_noise_ratio=0.25,
    neg_esc50_ratio=0.25,
    neg_music_ratio=0.25,
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
            # Add additional datasets for test if requested
            if use_additional_datasets:
                download_and_extract_zip(ESC_50_URL, root)
                download_and_extract_zip(VOCALSET_URL, root)
        elif split == "val":
            # For validation, we need dev-clean
            download_and_extract(VAL_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)
            # Add additional datasets for validation if requested
            if use_additional_datasets:
                download_and_extract_zip(ESC_50_URL, root)
                download_and_extract_zip(VOCALSET_URL, root)
        else:  # train split
            # For training, we need train-clean-100
            download_and_extract(TRAIN_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)
            # Add additional datasets for training if requested
            if use_additional_datasets:
                download_and_extract_zip(ESC_50_URL, root)
                download_and_extract_zip(VOCALSET_URL, root)

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
            libri_root = root / "LibriSpeech" / "train-clean-100"

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
            use_full_dataset=use_full_dataset,
            fleurs_langs=fleurs_langs,
            fleurs_streaming=fleurs_streaming,
            speech_dir=root / "fleurs_speech",
            neg_noise_ratio=neg_noise_ratio,
            neg_esc50_ratio=neg_esc50_ratio,
            neg_music_ratio=neg_music_ratio,
            use_additional_datasets=use_additional_datasets,
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
            default=["train", "val"],
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
            "--use_full_dataset",
            action="store_true",
            help="Use the entire dataset (ignores n_pos and n_neg count limits)",
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
            "--fraction_fleurs",
            type=float,
            default=0.50,
            help="Fraction of positive samples from FLEURS",
        )
        parser.add_argument(
            "--fraction_libri",
            type=float,
            default=0.30,
            help="Fraction of positive samples from LibriSpeech",
        )
        parser.add_argument(
            "--fraction_musan",
            type=float,
            default=0.20,
            help="Fraction of positive samples from MUSAN speech",
        )
        parser.add_argument(
            "--use_additional_datasets",
            action="store_true",
            default=True,
            help="Use ESC-50 (noise) and VocalSet (singing) datasets in addition to LibriSpeech and MUSAN",
        )
        parser.add_argument(
            "--neg_noise_ratio",
            type=float,
            default=0.25,
            help="Fraction of negative samples that are pure noise",
        )
        parser.add_argument(
            "--neg_esc50_ratio",
            type=float,
            default=0.25,
            help="Fraction of negative samples that are ESC-50 sounds",
        )
        parser.add_argument(
            "--neg_music_ratio",
            type=float,
            default=0.25,
            help="Fraction of negative samples that are music",
        )
        parser.add_argument(
            "--use_silero_vad",
            action="store_true",
            default=False,
            help="Use Silero VAD for generating frame-level labels",
        )

        args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

    global FRACTION_FLEURS, FRACTION_LIBRI, FRACTION_MUSAN
    FRACTION_FLEURS = args.fraction_fleurs
    FRACTION_LIBRI = args.fraction_libri
    FRACTION_MUSAN = args.fraction_musan

    # Validate that fractions sum to 1.0
    if abs(FRACTION_FLEURS + FRACTION_LIBRI + FRACTION_MUSAN - 1.0) > 1e-6:
        logger.warning(
            f"Source fractions sum to {FRACTION_FLEURS + FRACTION_LIBRI + FRACTION_MUSAN}, not 1.0"
        )

    # Replace the warning with this auto-adjustment code
    neg_sum = args.neg_noise_ratio + args.neg_esc50_ratio + args.neg_music_ratio
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
        logger.info(
            f"Adjusted ratios: noise={args.neg_noise_ratio:.3f}, "
            f"esc50={args.neg_esc50_ratio:.3f}, music={args.neg_music_ratio:.3f}"
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
            use_full_dataset=(
                args.use_full_dataset if hasattr(args, "use_full_dataset") else False
            ),
            fleurs_langs=args.fleurs_langs if hasattr(args, "fleurs_langs") else None,
            fleurs_streaming=(
                args.fleurs_streaming if hasattr(args, "fleurs_streaming") else False
            ),
            use_additional_datasets=(
                args.use_additional_datasets
                if hasattr(args, "use_additional_datasets")
                else True
            ),
            neg_noise_ratio=(
                args.neg_noise_ratio if hasattr(args, "neg_noise_ratio") else 0.25
            ),
            neg_esc50_ratio=(
                args.neg_esc50_ratio if hasattr(args, "neg_esc50_ratio") else 0.25
            ),
            neg_music_ratio=(
                args.neg_music_ratio if hasattr(args, "neg_music_ratio") else 0.25
            ),
            use_silero_vad=(
                args.use_silero_vad if hasattr(args, "use_silero_vad") else False
            ),
        )
        print(f"✅ Created manifest: {manifest_path}")


if __name__ == "__main__":
    main()
