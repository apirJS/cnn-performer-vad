import numpy as np
import librosa
import scipy.signal as signal
import soundfile as sf
import torch
import torch.nn.functional as F
import torch.hub
import argparse, csv, random, pathlib, subprocess, sys, json, time
import os

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Audio
from typing import Dict, Tuple, Optional
from config import *
from constant import *
from utils import logger

from utils import (
    stratified_fleurs_sample,
    stratified_librispeech_sample,
    apply_eq,
    add_reverb,
    load_and_process_audio,
)

from prepare_data import (
    process_esc50,
    process_musan,
    process_musdb,
    process_urbansound8k,
    process_vocalset,
    ingest_fleurs,
    validate_audio_sample,
    validate_negative_audio_sample,
    generate_silero_vad_labels,
    create_clean_speech_sample_with_silero,
    create_overlapping_speech_with_silero,
    create_speech_music_sample_with_silero,
    create_speech_noise_music_sample_with_silero,
    create_esc50_negative_sample,
    create_negative_sample,
    create_mixed_negative_sample,
    create_music_music_sample,
    create_noise_noise_sample,
    get_balanced_esc50_files,
)


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
    neg_noise_ratio=0.1,
    neg_esc50_ratio=0.2,
    neg_music_ratio=0.2,
    neg_music_noise_ratio=0.1,
    neg_noise_noise_ratio=0.1,
    neg_music_music_ratio=0.1,
    neg_urbansound_ratio=0.2,
    vad_model=None,
):
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
        cleanup_stems=True,  # Save disk space by removing stems
        augment=split_name == "train",  # Only augment training data
        segments_per_track=8,  # Extract 8 segments per track
        segment_length=10,  # 10-second segments
        augmentations_per_segment=5,  # Create 5 variants per segment
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
        root_dir=root_dir.parent, split_name=split_name, sample_rate=sample_rate
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
            cache_dir="D:/belajar/audio/vad/hf_cache",
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
    fleurs_shuffled = stratified_fleurs_sample(
        fleurs_speech, n_samples=min(q_fleurs, len(fleurs_speech))
    )
    musan_shuffled = musan_speech.copy()
    vocalset_shuffled = vocalset_speech.copy()

    random.shuffle(libri_shuffled)
    random.shuffle(fleurs_shuffled)
    random.shuffle(musan_shuffled)
    random.shuffle(vocalset_shuffled)

    # Sample from each shuffled source based on quotas
    selected_libri = libri_shuffled[: min(q_libri, len(libri))]
    selected_fleurs = fleurs_shuffled[: min(q_fleurs, len(fleurs_speech))]
    selected_musan = musan_shuffled[: min(q_musan, len(musan_speech))]
    selected_vocalset = vocalset_shuffled[: min(q_vocalset, len(vocalset_speech))]

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
    logger.info(f"Generating {actual_n_pos} clean speech samples...")
    clean_speech_start_time = time.time()
    generated_clean = 0
    clean_speech_failures = 0

    for idx in tqdm(range(actual_n_pos), desc="Generating clean speech samples"):
        duration = random.uniform(*duration_range)

        audio, vad_labels = create_clean_speech_sample_with_silero(
            selected_speech,
            duration,
            sample_rate,
            win_length,
            hop_length,
            vad_model,
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
    n_noise_speech_samples = int(
        round(generated_clean * 0.30)
    )  # Regular speech with noise
    n_overlap_samples = int(round(generated_clean * 0.20))  # Overlapping speech
    n_music_speech_samples = int(round(generated_clean * 0.25))  # Speech with music
    n_snm_samples = int(round(generated_clean * 0.25))  # Speech with noise and music

    # → POSITIVES SAMPLES (speech + noise) with enhanced augmentation
    logger.info(f"Generating {n_noise_speech_samples} regular speech+noise samples...")
    regular_start_time = time.time()
    speech_noise_generated = 0
    failed_positives = 0

    for idx in tqdm(
        range(n_noise_speech_samples), desc="Generating regular positive samples"
    ):
        # Random duration between duration_range
        duration = random.uniform(*duration_range)
        target_length = int(duration * sample_rate)

        # ----- STEP 1: LOAD AUDIO -----
        # Apply random starting offset if audio is longer than needed
        sp_path = random.choice(selected_speech)
        speech = load_and_process_audio(
            sp_path,
            sample_rate,
            duration=duration,
            target_length=target_length,  # safety pad if needed
            random_offset=True,  # explicit for clarity
        )

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
            volume_factor = random.uniform(0.1, 0.4)  # Very quiet (ASMR-like)
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
            source_type = "pos_sn_libri"
        elif "musan/speech" in str(sp_path):
            source_type = "pos_sn_musan"
        elif "VocalSet" in str(sp_path):
            source_type = "pos_sn_vocalset"
        else:
            source_type = "pos_sn_fleurs"

        # Add descriptive tags for special augmentations
        source_prefix = source_type
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
        speech_noise_generated += 1

    # After all positive generation:
    # Measure actual positive samples generated (safer approach)
    speech_noise_generated = len(list(out_pos.glob("pos_sn*.wav")))

    pos_time = time.time() - regular_start_time
    logger.info(
        f"Completed {speech_noise_generated} positive samples in {pos_time:.2f}s ({pos_time/speech_noise_generated:.3f}s per sample)"
    )

    # → OVERLAPPING SPEECH SAMPLES
    logger.info(f"Generating {n_overlap_samples} overlapping speech samples...")
    overlap_start_time = time.time()
    overlap_speech_generated = 0
    overlap_failures = 0

    if n_overlap_samples > 0:
        logger.info(f"Generating {n_overlap_samples} overlapping speech samples...")
        overlap_start_time = time.time()

        for idx in tqdm(
            range(n_overlap_samples), desc="Generating overlapping speech samples"
        ):
            duration = random.uniform(*duration_range)

            mix, vad_labels = create_overlapping_speech_with_silero(
                selected_speech,
                musan_noise,
                duration,
                sample_rate,
                win_length,
                hop_length,
                vad_model,
            )

            # Validate before saving
            if not validate_audio_sample(mix, sample_rate):
                overlap_failures += 1
                continue

            # Save the validated sample
            sf.write(out_pos / f"pos_overlap_{idx:06}.wav", mix, sample_rate)
            np.save(out_pos_labels / f"pos_overlap_{idx:06}_labels.npy", vad_labels)
            overlap_speech_generated += 1

        overlap_time = time.time() - overlap_start_time
        logger.info(
            f"Completed {overlap_speech_generated} overlapping speech samples in {overlap_time:.2f}s "
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

            mix, vad_labels = create_speech_music_sample_with_silero(
                selected_speech,
                musdb_music,
                duration,
                sample_rate,
                win_length,
                hop_length,
                vad_model,
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
    speech_music_noise_generated = 0
    snm_failures = 0

    if n_snm_samples > 0:
        logger.info(f"Generating {n_snm_samples} speech+noise+music samples...")
        snm_start_time = time.time()

        for idx in tqdm(
            range(n_snm_samples), desc="Generating speech+noise+music samples"
        ):
            duration = random.uniform(*duration_range)

            mix, vad_labels = create_speech_noise_music_sample_with_silero(
                selected_speech,
                all_noise_files,
                musdb_music,
                duration,
                sample_rate,
                win_length,
                hop_length,
                vad_model,
            )

            # Validate before saving
            if not validate_audio_sample(mix, sample_rate):
                snm_failures += 1
                continue

            # Save the validated sample
            sf.write(out_pos / f"pos_snm_{idx:06}.wav", mix, sample_rate)
            np.save(out_pos_labels / f"pos_snm_{idx:06}_labels.npy", vad_labels)
            speech_music_noise_generated += 1

        snm_time = time.time() - snm_start_time
        logger.info(
            f"Completed {speech_music_noise_generated} speech+noise+music samples in {snm_time:.2f}s "
            f"({snm_failures} failed validation)"
        )

    # → NEGATIVE SAMPLES (noise only, music only, noise + music, music + music)
    logger.info(
        "Generating negative samples with frame-level labels (noise only, music only, noise + music)..."
    )
    start_time = time.time()

    # Use the provided n_neg or match the total positive count
    total_positives = (
        generated_clean
        + speech_noise_generated
        + overlap_speech_generated
        + music_speech_generated
        + speech_music_noise_generated
    )

    # Set negative samples to be 80-90% of positive count for a slight positive bias
    actual_n_neg = int(total_positives * 0.85)  # 85% ratio creates a 54/46 split

    # Compute per-type negative quotas once
    n_noise_only = int(round(actual_n_neg * neg_noise_ratio))
    n_esc50_only = int(round(actual_n_neg * neg_esc50_ratio))
    n_music_only = int(round(actual_n_neg * neg_music_ratio))
    n_noise_noise = int(round(actual_n_neg * neg_noise_noise_ratio))  # New quota
    n_music_music = int(round(actual_n_neg * neg_music_music_ratio))  # New quota
    n_urbansound = int(round(actual_n_neg * neg_urbansound_ratio))
    n_mixed = int(round(actual_n_neg * neg_music_noise_ratio))

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
        if not validate_negative_audio_sample(audio, sample_rate):
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
        if not validate_negative_audio_sample(audio, sample_rate):
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
        if not validate_negative_audio_sample(audio, sample_rate):
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
        if not validate_negative_audio_sample(mix, sample_rate):
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
        if not validate_negative_audio_sample(mix, sample_rate):
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
        if not validate_negative_audio_sample(mix, sample_rate):
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
                category="urbansound",
            )

            # Validate before saving
            if not validate_negative_audio_sample(audio, sample_rate):
                urbansound_failures += 1
                continue

            # Save the validated sample
            sf.write(out_neg / f"neg_urbansound_{idx:06}.wav", audio, sample_rate)
            np.save(out_neg_labels / f"neg_urbansound_{idx:06}_labels.npy", vad_labels)
            urbansound_generated += 1

        logger.info(
            f"Generated {urbansound_generated} UrbanSound8K samples ({urbansound_failures} failed validation)"
        )

    neg_time = time.time() - start_time
    logger.info(
        f"Completed {actual_n_neg} negative samples in {neg_time:.2f}s ({neg_time/actual_n_neg:.3f}s per sample)"
    )

    # ── FINAL SUMMARY ─────────────────────────────────────────────────
    summary = {
        "positives_clean": generated_clean,
        "positives_regular": speech_noise_generated,
        "positives_overlap": overlap_speech_generated,
        "positives_music": music_speech_generated,
        "positives_snm": speech_music_noise_generated,
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
