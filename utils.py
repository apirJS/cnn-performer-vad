import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
import numpy as np
import librosa
import scipy.signal as signal
from pathlib import Path
from typing import Optional
from scipy.ndimage import binary_closing, binary_dilation
from config import *


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def is_vocalset(path: Path) -> bool:
    tag = "vocalset_"
    return path.name.lower().startswith(tag) or any(
        part.lower() == tag for part in path.parts
    )


def get_original_path(path: Path) -> Path:
    """Remove synthetic prefix (vocalset_, fleurs_, …) and return actual disk file."""
    if not isinstance(path, Path):
        path = Path(path)
    fname = path.name
    if any(
        fname.startswith(pref) for pref in ["vocalset_", "libri_", "musan_", "fleurs_"]
    ):
        return path.parent / fname.split("_", 1)[1]
    return path


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


def download_and_extract(
    url: str, dest: pathlib.Path, custom_filename: str = None, dataset_name: str = None
):
    """
    Download and extract a tar.gz archive if not already done.

    Args:
        url: URL to download from
        dest: Destination path
        custom_filename: Override filename (otherwise derived from URL)
        dataset_name: Name of dataset for error messages (defaults to filename)
    """
    if custom_filename:
        fname = dest / custom_filename
    else:
        fname = dest / url.split("/")[-1]

    dataset_name = dataset_name or fname.name

    if not fname.exists():
        success = download_file(url, fname)
        if not success:
            logger.error(f"Failed to download {dataset_name}")
            sys.exit(f"❌ Failed to download {dataset_name}")
    else:
        logger.info(
            f"{dataset_name} archive already exists: {fname}, skipping download"
        )

    mark = dest / f".extracted_{fname.name}"
    if not mark.exists():
        logger.info(f"Extracting {dataset_name} archive")
        print(f"📦 Extracting {fname.name}")
        import tarfile

        try:
            with tarfile.open(fname) as tar:
                tar.extractall(path=dest)
            mark.touch()
            logger.info(f"{dataset_name} extraction completed and marked with {mark}")
        except Exception as e:
            logger.error(f"Error extracting {dataset_name}: {e}")
            sys.exit(f"❌ Error extracting {dataset_name}: {e}")
    else:
        logger.info(
            f"{dataset_name} archive already extracted (marker file exists): {mark}"
        )


def download_and_extract_zip(url: str, dest: pathlib.Path):
    """Download and extract a ZIP archive if not already done."""
    url_filename = url.split("/")[-1]
    if "?" in url_filename:
        url_filename = url_filename.split("?")[0]

    fname = dest / url_filename

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
        # Use get_original_path to get the actual file path
        actual_path = get_original_path(libri)
        # Extract speaker ID from path structure
        speaker = str(actual_path.parent.parent.name)
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(libri)  # Keep original path for selection

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
        # Use get_original_path to get the actual file path
        actual_path = get_original_path(file_path)
        # Extract language code from filename (format: {lang_code}_{id}.wav)
        lang_code = actual_path.stem.split("_")[0]
        if lang_code not in languages:
            languages[lang_code] = []
        languages[lang_code].append(file_path)  # Keep original path for selection

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


def merge_vad_labels(*label_arrays):
    """
    Logical-OR reduction of any number of 1-D VAD masks.
    All arrays must have identical length.
    """
    merged = np.zeros_like(label_arrays[0], dtype=np.float32)
    for arr in label_arrays:
        merged = np.maximum(merged, arr)
    return merged



def label_singing_vocals_heuristically(
    audio_path: Path,
    frame_hop_length: int = DEFAULT_HOP_LENGTH,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    # --- Tunables with improved defaults for singing -------------
    energy_rms_frame_ms: int = 25,
    energy_rms_hop_ms: int = 10,
    energy_thresh_percentile: int = 5,      # Lower to 5% to catch very quiet singing
    pitch_fmin: float = librosa.note_to_hz("A2"),
    pitch_fmax: float = librosa.note_to_hz("C6"),
    voiced_prob_threshold: float = 0.2,    # Even more permissive threshold
    min_segment_duration_ms: int = 60,      # Shorter to catch brief onsets
    expand_segment_ms: int = 100,           # Expand more to connect segments
    smoothing_filter_size: int = 7,
    apply_normalization: bool = True,
    apply_preemphasis: bool = True, 
    use_alternative_method: bool = True,
    detect_quiet_beginnings: bool = True,   # NEW: special handling for quiet starts
) -> np.ndarray:
    """Return frame‑level 0/1 labels indicating probable singing voice.

    Enhanced for singing voice detection with normalization, pre-emphasis,
    and alternative detection for subtle singing passages.
    """

    # ------------------------------------------------------------------
    # 1. Load & pre‑process audio (stereo → mono, resample)
    # ------------------------------------------------------------------
    try:
        y, sr_orig = librosa.load(audio_path, sr=None, mono=False)
    except Exception as exc:
        logger.error("Failed to read %s: %s", audio_path, exc)
        return np.zeros(1, dtype=np.float32)

    if y.ndim == 2:  # stereo → mono while avoiding phase cancellation issues
        y = librosa.to_mono(y)
    if sr_orig != sample_rate:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sample_rate)

    if y.size == 0:  # empty file guard
        return np.zeros(1, dtype=np.float32)
    
    # NEW: Normalize audio to improve consistency
    if apply_normalization and np.abs(y).max() > 0:
        y = y / np.abs(y).max() * 0.9  # Scale to 90% of max amplitude
    
    # ------------------------------------------------------------------
    # 2. Feature extraction (Energy + Pitch) at a *feature* rate
    # ------------------------------------------------------------------
    rms_frame_len = int(sample_rate * energy_rms_frame_ms / 1000)
    rms_hop_len = int(sample_rate * energy_rms_hop_ms / 1000)
    
    # NEW: Apply pre-emphasis to boost high frequencies if enabled
    y_for_energy = y
    if apply_preemphasis:
        y_for_energy = librosa.effects.preemphasis(y, coef=0.97)
    
    # Calculate RMS energy on possibly pre-emphasized signal
    rms_energy = librosa.feature.rms(
        y=y_for_energy, 
        frame_length=rms_frame_len,
        hop_length=rms_hop_len, 
        center=True
    )[0]

    # Calculate pitch on original (not pre-emphasized) signal
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,  # Use original signal for pitch
        fmin=pitch_fmin,
        fmax=pitch_fmax,
        sr=sample_rate,
        frame_length=rms_frame_len,
        hop_length=rms_hop_len,
        center=True,
        fill_na=0.0,
    )

    # Align arrays length if pyin pads one extra frame
    n_feat_frames = min(len(rms_energy), len(voiced_prob))
    rms_energy = rms_energy[:n_feat_frames]
    voiced_prob = voiced_prob[:n_feat_frames]
    
    # ------------------------------------------------------------------
    # 3. Initial decision rule (energy + voicing)
    # ------------------------------------------------------------------
    energy_threshold = np.percentile(rms_energy, energy_thresh_percentile)
    energy_threshold = max(energy_threshold, 1e-5)  # protect against zero

    is_energetic = rms_energy > energy_threshold
    is_voiced = voiced_prob > voiced_prob_threshold
    speech_mask = (is_energetic & is_voiced).astype(np.uint8)
    
    # New block to add near line 118 in your function:
    # Further enhancement for low/quiet singing specifically
    if use_alternative_method:
        # [your existing code]
        
        # NEW: Additional pass specifically for low notes (which may have good voicing confidence but low energy)
        low_notes_mask = np.logical_and(
            voiced_prob > 0.3,  # Even more permissive for low notes
            f0 < librosa.note_to_hz("E3")  # Focus on low notes
        ).astype(np.uint8)
        
        # Add low notes to our speech mask
        speech_mask = np.logical_or(speech_mask, low_notes_mask).astype(np.uint8)
        
        # NEW: Special handling for extremely quiet singing with some pitch evidence
        # This is especially important for the beginnings of notes or quiet sustains
        if detect_quiet_beginnings:
            # Find regions with ANY detectable pitch (very permissive)
            any_pitch_evidence = voiced_prob > 0.15
            
            # Find regions with minimal energy (above absolute noise floor)
            minimal_energy = rms_energy > (np.max(rms_energy) * 0.02)  # Just 1% of max
            
            # Combine for extremely subtle singing detection
            quiet_singing_mask = np.logical_and(
                any_pitch_evidence, 
                minimal_energy
            ).astype(np.uint8)
            
            # Only add short segments (for safety - to avoid false positives)
            # This looks for short segments that might be note beginnings
            quiet_diff = np.diff(np.concatenate(([0], quiet_singing_mask, [0])))
            quiet_starts = np.where(quiet_diff == 1)[0]
            quiet_ends = np.where(quiet_diff == -1)[0]
            
            # Target very short segments for special protection (likely note beginnings)
            very_short_mask = np.zeros_like(quiet_singing_mask)
            for s, e in zip(quiet_starts, quiet_ends):
                segment_len = e - s
                if segment_len > 0 and segment_len < 30:  # Very short segments only
                    very_short_mask[s:e] = 1
            
            # Add these protected quiet beginnings
            speech_mask = np.logical_or(speech_mask, very_short_mask).astype(np.uint8)

    # ------------------------------------------------------------------
    # 4. Morphological smoothing & remove short blobs (unchanged)
    # ------------------------------------------------------------------
    if smoothing_filter_size % 2 == 0:
        smoothing_filter_size += 1  # ensure odd

    speech_mask = binary_closing(
        speech_mask,
        structure=np.ones(smoothing_filter_size, dtype=np.uint8),
    ).astype(np.uint8)

    # Remove blobs shorter than minimum length
    min_seg_frames = int(np.round(min_segment_duration_ms / 1000 * sample_rate / rms_hop_len))
    if min_seg_frames > 1:
        diff = np.diff(np.concatenate(([0], speech_mask, [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            if (e - s) < min_seg_frames:
                speech_mask[s:e] = 0

    # Expand segments to catch onsets/offsets
    if expand_segment_ms > 0 and speech_mask.any():
        expand_frames = int(np.round(expand_segment_ms / 1000 * sample_rate / rms_hop_len))
        speech_mask = binary_dilation(
            speech_mask,
            structure=np.ones(expand_frames * 2 + 1, dtype=np.uint8),
        ).astype(np.uint8)

    # ------------------------------------------------------------------
    # 5. Resample labels to VAD frame rate (vectorised, unchanged)
    # ------------------------------------------------------------------
    feature_times = librosa.frames_to_time(np.arange(len(speech_mask)),
                                           sr=sample_rate, hop_length=rms_hop_len)
    num_out_frames = int(np.ceil(len(y) / frame_hop_length))
    if num_out_frames == 0:
        num_out_frames = 1

    vad_times = librosa.frames_to_time(np.arange(num_out_frames),
                                       sr=sample_rate, hop_length=frame_hop_length)

    out_labels = np.interp(vad_times, feature_times, speech_mask).astype(np.float32)
    out_labels = (out_labels >= 0.5).astype(np.float32)

    return out_labels