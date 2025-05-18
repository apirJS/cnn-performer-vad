import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
import numpy as np
import librosa
import scipy.signal as signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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
        try:
            # Calculate Nyquist frequency
            nyquist = sr / 2

            # More conservative range for center frequency
            min_freq = 200  # Increased from 100
            max_freq = min(6000, nyquist * 0.9)  # More restrictive cap

            # Ensure we have a valid frequency range
            if min_freq >= max_freq:
                continue  # Skip this iteration

            center_freq = random.uniform(min_freq, max_freq)

            # Normalize frequency to range (0, 1) with stricter bounds
            w0 = center_freq / nyquist

            # Much stricter safety bounds
            w0 = max(0.01, min(0.99, w0))

            # Double-check validity before proceeding
            if w0 <= 0 or w0 >= 1:
                continue  # Skip if invalid despite our checks

            # More conservative Q and gain values
            q = random.uniform(0.8, 4.0)
            gain_db = random.uniform(-8, 8)  # Reduced from -10,10

            # Create and apply peaking filter with error handling
            b, a = signal.iirpeak(w0, q, gain_db)
            eq_audio = signal.lfilter(b, a, eq_audio)

        except Exception as e:
            # Gracefully handle any errors without stopping the whole process
            continue  # Skip this filter and continue

    # Normalize output to prevent level blow-up
    max_val = np.max(np.abs(eq_audio))
    if max_val > 0:  # Avoid division by zero
        eq_audio = eq_audio / (max_val + 1e-7)

    return eq_audio


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
        logger.info(
            f"UrbanSound8K archive already extracted (marker file exists): {mark}"
        )


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


def load_and_process_audio(
    file_path,
    sample_rate,
    duration=None,
    target_length=None,
    random_offset: bool = True,
):
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
    audio, sr = librosa.load(file_path, sr=sample_rate)  # load full track

    # --- NEW: random crop ---
    if duration is not None:
        want = int(duration * sample_rate)
        if len(audio) >= want:
            if random_offset:
                start = random.randint(0, len(audio) - want)
                audio = audio[start : start + want]
            else:  # deterministic (old behaviour)
                audio = audio[:want]
        else:  # source shorter → pad
            audio = librosa.util.fix_length(audio, size=want)

    if target_length:
        audio = librosa.util.fix_length(audio, size=target_length)

    return audio


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
        stratified_files.extend(remaining[: n_samples - len(stratified_files)])

    # If we have too many, trim
    return stratified_files[:n_samples]


def stratified_fleurs_sample(fleurs_files, n_samples=10000):
    """Sample FLEURS files with stratification by language."""
    # Handle empty input case
    if not fleurs_files:
        logger.warning(
            "Empty FLEURS file list provided to stratified sampling function"
        )
        return []

    # Group files by language code
    languages = {}
    for file_path in fleurs_files:
        # Extract language code from filename (format: {lang_code}_{id}.wav)
        lang_code = file_path.stem.split("_")[0]
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
        stratified_files.extend(remaining[: n_samples - len(stratified_files)])

    # If we have too many, trim
    return stratified_files[:n_samples]


import numpy as np


def merge_vad_labels(*label_arrays):
    """
    Logical-OR reduction of any number of 1-D VAD masks.
    All arrays must have identical length.
    """
    merged = np.zeros_like(label_arrays[0], dtype=np.float32)
    for arr in label_arrays:
        merged = np.maximum(merged, arr)
    return merged
