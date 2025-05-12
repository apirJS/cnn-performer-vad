#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  prepare_data.py Data preparation for MEL-spectrogram VAD
# ───────────────────────────────────────────────────────────────────────
import argparse, csv, random, pathlib, subprocess, sys, json, logging, time
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal


from config import *

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


# ───────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
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
    threshold1 = np.sort(frame_energies1)[int(len(frame_energies1) * 0.3)]
    labels1 = (frame_energies1 > threshold1).astype(np.float32)

    # Generate labels for second speech segment
    frame_energies2 = librosa.feature.rms(
        y=second_speech, frame_length=win_length, hop_length=hop_length
    )[0]

    # Verify frame count matches
    assert (
        len(frame_energies2) == second_frames
    ), f"Frame count mismatch: {len(frame_energies2)} vs {second_frames}"

    # Calculate threshold and assign labels
    threshold2 = np.sort(frame_energies2)[int(len(frame_energies2) * 0.3)]
    labels2 = (frame_energies2 > threshold2).astype(np.float32)

    # Combine all labels
    combined_labels = np.concatenate([labels1, labels_gap, labels2])

    return combined_audio, combined_labels


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
):
    """Generate variable-length positives (speech+noise) & negatives (noise/music) with frame-level labels."""

    logger.info(
        f"Generating {n_pos} positive and {n_neg} negative samples for {split_name} split"
    )
    logger.info(f"Duration range: {duration_range}s, SNR range: {snr_range}dB")

    # Get speech files from both LibriSpeech and MUSAN
    libri = sorted(libri_root.rglob("*.flac"))
    musan_speech = list(musan_root.joinpath("speech").rglob("*.wav"))
    musan_noise = list(musan_root.joinpath("noise").rglob("*.wav"))
    musan_music = list(musan_root.joinpath("music").rglob("*.wav"))

    logger.info(
        f"Found {len(libri)} LibriSpeech files, {len(musan_speech)} MUSAN speech files, "
        f"{len(musan_noise)} noise files, {len(musan_music)} music files"
    )
    assert libri and musan_noise and musan_music, "Missing source audio!"

    # Create combined pool of speech files
    all_speech_files = libri + musan_speech
    logger.info(f"Total speech files available: {len(all_speech_files)}")

    # Determine actual number of samples to use
    if use_full_dataset and split_name == "train":
        # Use all available speech files from both sources
        selected_speech = all_speech_files.copy()
        random.shuffle(selected_speech)  # Shuffle to ensure diversity
        actual_n_pos = len(selected_speech)
        actual_n_neg = actual_n_pos  # Generate equal number of negatives
        logger.info(
            f"Using full dataset: {actual_n_pos} positive samples from combined sources"
        )
    else:
        # Limit to requested sample count, but sample from the combined pool
        actual_n_pos = min(n_pos, len(all_speech_files))
        selected_speech = random.sample(all_speech_files, actual_n_pos)
        actual_n_neg = n_neg
        logger.info(
            f"Using sample of dataset: {actual_n_pos} positive samples from combined sources (requested: {n_pos})"
        )

    # Create directories for audio and frame-level labels
    out_pos.mkdir(parents=True, exist_ok=True)
    out_neg.mkdir(parents=True, exist_ok=True)
    out_pos_labels = out_pos.parent.parent / f"{split_name}_pos_labels"  # Updated path
    out_neg_labels = out_neg.parent.parent / f"{split_name}_neg_labels"  # Updated path
    out_pos_labels.mkdir(parents=True, exist_ok=True)
    out_neg_labels.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Output directories created: {out_pos}, {out_neg}, {out_pos_labels}, {out_neg_labels}"
    )

    # Constants for frame-level calculations
    hop_length = DEFAULT_HOP_LENGTH  # Match your spectrogram hop_length
    win_length = DEFAULT_WIN_LENGTH  # Match your spectrogram win_length

    # → POSITIVE SAMPLES (speech + noise) with enhanced augmentation
    logger.info(
        "Generating positive samples (speech + noise) with enhanced augmentation..."
    )
    start_time = time.time()
    generated_pos = 0
    for idx, sp_path in enumerate(selected_speech):
        if idx % 100 == 0:
            logger.info(f"Generating positive sample {idx}/{actual_n_pos}")

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
            clean_speech = speech.copy()  # Keep a clean copy

            # We'll skip time stretching for gap samples to maintain alignments
            stretch_rate = 1.0
        else:
            # Standard single utterance processing
            full_speech, sr = librosa.load(sp_path, sr=sample_rate)

            # Apply random starting offset if audio is longer than needed
            if len(full_speech) > target_length:
                max_offset = len(full_speech) - target_length
                offset = random.randint(0, max_offset)
                original_speech = full_speech[offset : offset + target_length]
            else:
                # If too short, pad as needed
                original_speech = librosa.util.fix_length(
                    full_speech, size=target_length
                )

            # Store a copy for VAD label generation (before any augmentation)
            clean_speech = original_speech.copy()

            # Start with the original speech for augmentation
            speech = original_speech.copy()
            stretch_rate = 1.0

            # Generate frame-level VAD labels from the original clean speech
            n_frames = 1 + (len(clean_speech) - win_length) // hop_length

            # Calculate energy of each frame
            frame_energies = librosa.feature.rms(
                y=clean_speech, frame_length=win_length, hop_length=hop_length
            )[0]

            # Sort energies and use threshold approach
            sorted_energies = np.sort(frame_energies)
            energy_threshold = sorted_energies[int(len(sorted_energies) * 0.3)]
            vad_labels = (frame_energies > energy_threshold).astype(np.float32)

            # Smooth labels with longer minimum segments
            min_speech_frames = 3  # About 50ms at 16kHz with default hop_length
            for i in range(len(vad_labels) - min_speech_frames + 1):
                if 0 < sum(vad_labels[i : i + min_speech_frames]) < min_speech_frames:
                    vad_labels[i : i + min_speech_frames] = 0

        # ───── ENHANCED VOLUME VARIATION FOR ASMR-LIKE AUDIO ─────
        # Apply more extreme volume variations (0.05-1.4 range)
        if random.random() < 0.2:  # 20% chance of very quiet speech
            volume_factor = random.uniform(0.05, 0.4)  # Very quiet (ASMR-like)
            logger.debug(f"Using very low volume factor: {volume_factor:.2f}")
        else:
            volume_factor = random.uniform(0.7, 1.4)  # Normal range

        speech *= volume_factor

        # Check if speech has vanished due to extreme low volume
        if np.std(speech) < 1e-4:  # speech vanished
            continue  # skip sample

        # Apply standard time/pitch augmentations, but only to non-gap samples
        if not use_speech_gap:
            # Apply time stretching if random condition is met
            if random.random() > 0.5:
                stretch_rate = random.uniform(0.9, 1.1)
                speech = librosa.effects.time_stretch(speech, rate=stretch_rate)

            # Apply pitch shifting if random condition is met
            if random.random() > 0.5:
                speech = librosa.effects.pitch_shift(
                    speech, sr=sample_rate, n_steps=random.uniform(-3.5, 3.5)
                )

            # Fix length to match target
            speech = librosa.util.fix_length(speech, size=target_length)

        # ───── REVERB AND CHANNEL EFFECTS ─────
        # Apply reverb with 30% probability
        apply_reverb_effect = random.random() < 0.3
        if apply_reverb_effect:
            speech = add_reverb(speech, sample_rate)
            logger.debug("Applied reverb effect")

        # Apply EQ with 25% probability
        apply_eq_effect = random.random() < 0.25
        if apply_eq_effect:
            speech = apply_eq(speech, sample_rate)
            logger.debug("Applied EQ effect")

        # ───── BACKGROUND MIXING WITH VARIED SNR ─────
        # Load background noise
        noise_path = random.choice(musan_noise)
        noise, _ = librosa.load(
            noise_path, sr=sample_rate, mono=True, duration=duration
        )
        noise = librosa.util.fix_length(noise, size=len(speech))

        # Apply more extreme SNR variation for ASMR-like cases
        if random.random() < 0.15:  # 15% chance of extremely challenging SNR
            snr_db = random.uniform(-10, -3)  # Very challenging SNR
            logger.debug(f"Using extremely low SNR: {snr_db:.1f} dB")
        else:
            snr_db = random.uniform(*snr_range)  # Normal range

        alpha = 10 ** (-snr_db / 20) * np.std(speech) / (np.std(noise) + 1e-7)
        mix = speech + alpha * noise

        # ───── SAVING ─────
        # Determine source type and add descriptive tags
        source_prefix = "pos_libri" if str(sp_path).endswith(".flac") else "pos_musan"

        # Add descriptive tags for special augmentations
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
        sf.write(out_pos / f"{source_prefix}_{idx:06}.wav", mix, sample_rate)
        np.save(out_pos_labels / f"{source_prefix}_{idx:06}_labels.npy", vad_labels)
        generated_pos += 1

    pos_time = time.time() - start_time
    logger.info(
        f"Completed {generated_pos} positive samples in {pos_time:.2f}s ({pos_time/generated_pos:.3f}s per sample)"
    )

    # → OVERLAPPING SPEECH SAMPLES (more challenging positives)
    n_overlap_samples = generated_pos // 5  # Generate 20% as many overlapping samples

    if n_overlap_samples > 0:
        logger.info(f"Generating {n_overlap_samples} overlapping speech samples...")
        overlap_start_time = time.time()

        for idx in range(n_overlap_samples):
            if idx % 20 == 0:
                logger.info(
                    f"Generating overlapping speech sample {idx}/{n_overlap_samples}"
                )

            # Select two different speech files
            speech1, speech2 = random.sample(all_speech_files, 2)

            # Load and process both speech files
            duration = random.uniform(*duration_range)
            primary, sr = librosa.load(speech1, sr=sample_rate, duration=duration)
            secondary, _ = librosa.load(speech2, sr=sample_rate, duration=duration)

            # Make primary and secondary same length
            primary = librosa.util.fix_length(primary, size=int(duration * sr))
            secondary = librosa.util.fix_length(secondary, size=int(duration * sr))

            # Make secondary speech quieter (foreground vs background speaker)
            snr_db = random.uniform(5, 15)  # Reasonable SNR for overlapping speech
            alpha = 10 ** (-snr_db / 20) * np.std(primary) / (np.std(secondary) + 1e-7)

            # Add noise too for realism
            noise_path = random.choice(musan_noise)
            noise, _ = librosa.load(noise_path, sr=sr, mono=True, duration=duration)
            noise = librosa.util.fix_length(noise, size=len(primary))

            # Create the final mixture (primary speech + quieter secondary speech + low noise)
            mix = primary + alpha * secondary + 0.1 * noise

            # Generate frame-level VAD labels based on energy of primary speech
            n_frames = 1 + (len(primary) - win_length) // hop_length
            frame_energies = librosa.feature.rms(
                y=mix, frame_length=win_length, hop_length=hop_length
            )[0]

            # Same threshold logic as main positive samples
            sorted_energies = np.sort(frame_energies)
            energy_threshold = sorted_energies[int(len(sorted_energies) * 0.3)]
            vad_labels = (frame_energies > energy_threshold).astype(np.float32)

            # Save with special prefix to identify overlapping samples
            mix = np.clip(mix, -1.0, 1.0)
            sf.write(out_pos / f"pos_overlap_{idx:06}.wav", mix, sr)
            np.save(out_pos_labels / f"pos_overlap_{idx:06}_labels.npy", vad_labels)

        overlap_time = time.time() - overlap_start_time
        logger.info(
            f"Completed {n_overlap_samples} overlapping speech samples in {overlap_time:.2f}s"
        )

    # → NEGATIVE SAMPLES (noise only, music only, noise + music)
    logger.info(
        "Generating negative samples with frame-level labels (noise only, music only, noise + music)..."
    )
    start_time = time.time()

    # Adjust the negative sample count to match total positives
    total_positives = generated_pos + (generated_pos // 5)
    actual_n_neg = total_positives

    # Then divide the negatives into categories as before
    n_noise_only = actual_n_neg // 3
    n_music_only = actual_n_neg // 3
    n_noise_music = actual_n_neg - n_noise_only - n_music_only

    # 1. NOISE ONLY samples
    logger.info(f"Generating {n_noise_only} noise-only samples")
    for idx in range(n_noise_only):
        if idx % 50 == 0:
            logger.info(f"Generating noise-only sample {idx}/{n_noise_only}")

        duration = random.uniform(*duration_range)
        n1, n2 = random.choices(musan_noise, k=2)
        a, sr = librosa.load(n1, sr=sample_rate, mono=True, duration=duration)
        b, _ = librosa.load(n2, sr=sr, mono=True, duration=duration)
        mix = librosa.util.fix_length(
            a, size=int(duration * sr)
        ) + 0.5 * librosa.util.fix_length(b, size=int(duration * sr))

        # All frames are non-speech (0)
        n_frames = 1 + (len(mix) - win_length) // hop_length
        vad_labels = np.zeros(n_frames, dtype=np.float32)

        mix = np.clip(mix, -1.0, 1.0)
        sf.write(out_neg / f"neg_noise_{idx:06}.wav", mix, sr)
        np.save(out_neg_labels / f"neg_noise_{idx:06}_labels.npy", vad_labels)

    # 2. MUSIC ONLY samples
    logger.info(f"Generating {n_music_only} music-only samples")
    for idx in range(n_music_only):
        if idx % 50 == 0:
            logger.info(f"Generating music-only sample {idx}/{n_music_only}")

        duration = random.uniform(*duration_range)
        m1, m2 = random.choices(musan_music, k=2)
        a, sr = librosa.load(m1, sr=sample_rate, mono=True, duration=duration)
        b, _ = librosa.load(m2, sr=sr, mono=True, duration=duration)
        mix = librosa.util.fix_length(
            a, size=int(duration * sr)
        ) + 0.5 * librosa.util.fix_length(b, size=int(duration * sr))

        # All frames are non-speech (0)
        n_frames = 1 + (len(mix) - win_length) // hop_length
        vad_labels = np.zeros(n_frames, dtype=np.float32)

        mix = np.clip(mix, -1.0, 1.0)
        sf.write(out_neg / f"neg_music_{idx:06}.wav", mix, sr)
        np.save(out_neg_labels / f"neg_music_{idx:06}_labels.npy", vad_labels)

    # 3. NOISE + MUSIC samples
    logger.info(f"Generating {n_noise_music} noise+music samples")
    for idx in range(n_noise_music):
        if idx % 50 == 0:
            logger.info(f"Generating noise+music sample {idx}/{n_noise_music}")

        duration = random.uniform(*duration_range)
        noise_file = random.choice(musan_noise)
        music_file = random.choice(musan_music)

        noise, sr = librosa.load(
            noise_file, sr=sample_rate, mono=True, duration=duration
        )
        music, _ = librosa.load(music_file, sr=sr, mono=True, duration=duration)

        mix = librosa.util.fix_length(
            noise, size=int(duration * sr)
        ) + 0.5 * librosa.util.fix_length(music, size=int(duration * sr))

        # All frames are non-speech (0)
        n_frames = 1 + (len(mix) - win_length) // hop_length
        vad_labels = np.zeros(n_frames, dtype=np.float32)

        mix = np.clip(mix, -1.0, 1.0)
        sf.write(out_neg / f"neg_mixed_{idx:06}.wav", mix, sr)
        np.save(out_neg_labels / f"neg_mixed_{idx:06}_labels.npy", vad_labels)

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {actual_n_neg} negative samples in {neg_time:.2f}s ({neg_time/actual_n_neg:.3f}s per sample)"
    )


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
        elif split == "val":
            # For validation, we need dev-clean
            download_and_extract(VAL_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)
        else:  # train split
            # For training, we need train-clean-100
            download_and_extract(TRAIN_LIBRISPEECH_URL, root)
            download_and_extract(MUSAN_URL, root)

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


def main():
    parser = argparse.ArgumentParser("Data Preparation for MEL-spectrogram VAD")

    # Base arguments
    parser.add_argument(
        "--root", required=True, help="Root directory for all datasets and outputs"
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
        default=[5, 10],
        help="Min and max duration in seconds for audio clips",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Sample rate for audio processing",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force rebuild even if data already exists"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        help="Use the entire LibriSpeech dataset instead of sample sizes (only affects training set)",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    seed_everything(args.seed)

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
            force_rebuild=args.force,
            use_full_dataset=args.use_full_dataset,
        )
        print(f"✅ Created manifest: {manifest_path}")


if __name__ == "__main__":
    main()
