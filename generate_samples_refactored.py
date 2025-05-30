import pathlib
import random
import time
import numpy as np
import soundfile as sf
from utils import logger
from tqdm import tqdm

from prepare_data import (
    process_esc50,
    process_musan,
    process_musdb,
    process_urbansound8k,
    process_vocalset,
    ingest_fleurs,
    validate_audio_sample,
    validate_negative_audio_sample,
    create_clean_speech_sample_with_silero,
    create_overlapping_speech_with_silero,
    create_speech_noise_sample_with_silero,
    create_speech_music_sample_with_silero,
    create_speech_noise_music_sample_with_silero,
    create_esc50_negative_sample,
    create_negative_sample,
    create_mixed_negative_sample,
    create_music_music_sample,
    create_noise_noise_sample,
    get_balanced_esc50_files,
)

from utils import (
    stratified_fleurs_sample,
    stratified_librispeech_sample,
)


from config import *
from constant import *


def _generate_sample_batch(
    num_samples_to_generate: int,
    description: str,
    sample_creation_func,  # This will be a partial or a lambda
    success_counter_list: list,  # Using a list to pass by reference for modification
    failure_counter_list: list,  # Using a list for modification
):
    """
    Generic loop for generating a batch of samples (positive or negative).
    My Lord, this is like a template for making many samples of one kind! (＾◡＾)っ
    """
    if num_samples_to_generate <= 0:
        logger.info(f"Skipping '{description}' as 0 samples are requested.")
        return

    logger.info(f"Generating {num_samples_to_generate} {description}...")
    start_time = time.time()

    # Local counters for this batch
    current_generated = 0
    current_failures = 0

    for i in tqdm(range(num_samples_to_generate), desc=description):
        if sample_creation_func(idx=i):  # Pass index for unique naming
            current_generated += 1
        else:
            current_failures += 1

    # Update overall counters
    success_counter_list[0] += current_generated
    failure_counter_list[0] += current_failures

    elapsed_time = time.time() - start_time
    time_per_sample_str = (
        f"({elapsed_time/current_generated:.3f}s per sample)"
        if current_generated > 0
        else ""
    )
    logger.info(
        f"Completed {current_generated} {description} in {elapsed_time:.2f}s "
        f"{time_per_sample_str} ({current_failures} failed validation)"
    )


# Main refactored function
def make_pos_neg(
    libri_root: pathlib.Path,
    musan_root: pathlib.Path,
    out_pos: pathlib.Path,
    out_neg: pathlib.Path,
    n_pos: int,
    n_neg: int,  # Target n_neg, actual might be adjusted based on positive count
    snr_range=(-5, 20),
    duration_range=(1, 10),
    sample_rate=16000,
    split_name="train",
    fleurs_langs=None,
    fleurs_streaming=False,
    speech_dir=None,  # For pre-existing FLEURS or other speech
    # Ratios for different negative sample types
    neg_noise_ratio=0.1,
    neg_esc50_ratio=0.2,
    neg_music_ratio=0.2,
    neg_music_noise_ratio=0.1,
    neg_noise_noise_ratio=0.1,
    neg_music_music_ratio=0.1,
    neg_urbansound_ratio=0.2,
    vad_model: tuple | None = None,  # <- NEW, backward-compatible
    vad_model_tuple: tuple | None = None,  # <- keep current name
):
    """
    Generates positive (speech-based) and negative (non-speech) audio samples
    for training a Voice Activity Detection (VAD) model.
    """
    logger.info(
        f"Starting data generation: {n_pos} positive & {n_neg} (target) negative samples for '{split_name}' split."
    )
    logger.info(f"Duration range: {duration_range}s, SNR range: {snr_range}dB")

    vad_model_tuple = vad_model_tuple or vad_model

    # --- 1. GATHER ALL RAW AUDIO SOURCE FILES ---
    random.seed(42)  # For deterministic shuffling and sampling

    libri_files = sorted(libri_root.rglob("*.flac"))
    logger.info(f"Found {len(libri_files)} LibriSpeech files initially.")
    libri_files = stratified_librispeech_sample(
        libri_files, n_samples=min(10000, len(libri_files))
    )
    logger.info(
        f"Using {len(libri_files)} LibriSpeech files after stratified sampling."
    )

    # Assuming musan_root is like 'datasets/musan/musan', parent is 'datasets/musan'
    musan_speech_files, musan_noise_files = process_musan(
        musan_root.parent, split_name=split_name, sample_rate=sample_rate
    )
    logger.info(
        f"Processed MUSAN: {len(musan_speech_files)} speech, {len(musan_noise_files)} noise files."
    )

    # Assuming out_pos.parent.parent.parent is the project root or similar for MUSDB
    musdb_music_files = process_musdb(
        out_pos.parent.parent.parent,  # This path seems a bit deep, ensure it's correct for your setup
        split_name=split_name,
        sample_rate=sample_rate,
        cleanup_stems=(split_name != "train"),  # Cleanup if not training to save space
        augment=(split_name == "train"),
        segments_per_track=8,
        segment_length=10,
        augmentations_per_segment=5,
    )
    logger.info(f"Processed MUSDB: {len(musdb_music_files)} music files.")

    # General 'datasets' root directory assumed relative to out_pos
    # Example: if out_pos is 'datasets/prepared/train/pos', then root_dir is 'datasets/prepared'
    # and root_dir.parent is 'datasets'
    datasets_root = out_pos.parent.parent.parent

    esc50_noise_files = process_esc50(
        root_dir=datasets_root, split_name=split_name, sample_rate=sample_rate
    )
    logger.info(f"Processed ESC-50: {len(esc50_noise_files)} noise files.")

    vocalset_speech_files = process_vocalset(
        root_dir=datasets_root,
        split_name=split_name,
        sample_rate=sample_rate,  # VocalSet might be in 'datasets/VocalSet'
    )
    logger.info(f"Processed VocalSet: {len(vocalset_speech_files)} speech files.")

    urbansound_noise_files = process_urbansound8k(
        root_dir=datasets_root, split_name=split_name, sample_rate=sample_rate
    )
    logger.info(f"Processed UrbanSound8K: {len(urbansound_noise_files)} noise files.")

    all_noise_sources = musan_noise_files + esc50_noise_files + urbansound_noise_files
    logger.info(f"Total diverse noise files collected: {len(all_noise_sources)}")

    fleurs_speech_files = []
    if fleurs_langs:
        logger.info(f"Processing FLEURS data for languages: {fleurs_langs}")
        # Assuming speech_dir is for output of FLEURS if not pre-existing
        fleurs_output_dir = pathlib.Path(
            speech_dir or out_pos.parent.parent / "fleurs_speech"
        )
        max_samples_per_lang = {
            "train": MAX_PER_LANG_TRAIN,
            "val": MAX_PER_LANG_VAL,
            "test": MAX_PER_LANG_TEST,
        }.get(
            split_name, MAX_PER_LANG_TRAIN
        )  # Default to train if split_name is unusual

        fleurs_speech_files = ingest_fleurs(
            lang_list=[l.strip().lower() for l in fleurs_langs.split(",")],
            out_dir=fleurs_output_dir,
            sr=sample_rate,
            streaming=fleurs_streaming,
            split=split_name,
            max_per_lang=max_samples_per_lang,
            shuffle_seed=42,
            cache_dir=datasets_root / "hf_cache",  # Example path, adjust as needed
        )
        logger.info(f"Ingested {len(fleurs_speech_files)} FLEURS speech files.")
    elif (
        speech_dir
    ):  # Use existing files if speech_dir is provided and fleurs_langs is not
        speech_dir_path = pathlib.Path(speech_dir)
        if speech_dir_path.exists():
            # Assuming these are preprocessed FLEURS or other compatible speech files
            fleurs_speech_files = list(speech_dir_path.rglob("*.wav")) + list(
                speech_dir_path.rglob("*.flac")
            )
            logger.info(
                f"Using {len(fleurs_speech_files)} existing speech files from {speech_dir}."
            )

    assert (
        libri_files and musan_noise_files and musdb_music_files
    ), "Critical audio sources (LibriSpeech, MUSAN noise, MUSDB music) are missing!"

    # --- 2. PREPARE BALANCED POOL OF SPEECH SOURCES ---

    # Add prefixes to file paths (in-memory only, doesn't rename actual files)
    libri_files = [path.parent / f"libri_{path.name}" for path in libri_files]
    logger.info(f"Added 'libri_' prefix to {len(libri_files)} LibriSpeech file paths")

    musan_speech_files = [
        path.parent / f"musan_{path.name}" for path in musan_speech_files
    ]
    logger.info(
        f"Added 'musan_' prefix to {len(musan_speech_files)} MUSAN speech file paths"
    )

    fleurs_speech_files = [
        path.parent / f"fleurs_{path.name}" for path in fleurs_speech_files
    ]
    logger.info(
        f"Added 'fleurs_' prefix to {len(fleurs_speech_files)} FLEURS speech file paths"
    )

    vocalset_speech_files = [
        path.parent / f"vocalset_{path.name}" for path in vocalset_speech_files
    ]
    logger.info(
        f"Added 'vocalset_' prefix to {len(vocalset_speech_files)} VocalSet file paths"
    )

    all_available_speech = {
        "libri": libri_files,
        "musan": musan_speech_files,
        "fleurs": fleurs_speech_files,
        "vocalset": vocalset_speech_files,
    }

    total_available_speech_count = sum(
        len(files) for files in all_available_speech.values()
    )
    actual_n_pos = min(n_pos, total_available_speech_count)
    logger.info(f"Targeting {actual_n_pos} positive samples based on availability.")

    selected_speech_final_pool = []
    if actual_n_pos > 0:
        source_weights = {
            "libri": 1.0,
            "musan": 1.0,
            "fleurs": 1.0,
            "vocalset": (
                1.0 if vocalset_speech_files else 0.0
            ),  # Only if VocalSet files exist
        }
        # Normalize weights for sources that actually have files
        valid_sources_weight_sum = sum(
            w for src, w in source_weights.items() if all_available_speech[src]
        )

        quotas = {}
        temp_total_quota = 0
        for source_name, files in all_available_speech.items():
            if not files:  # Skip sources with no files
                quotas[source_name] = 0
                continue
            # Calculate quota, ensuring it doesn't exceed available files for that source
            quota = int(
                round(
                    actual_n_pos
                    * source_weights[source_name]
                    / valid_sources_weight_sum
                )
            )
            quotas[source_name] = min(quota, len(files))
            temp_total_quota += quotas[source_name]

        # Adjust quotas if rounding caused sum != actual_n_pos (distribute remainder/deficit)
        # This is a simple adjustment, more sophisticated could be used if needed.
        quota_diff = actual_n_pos - temp_total_quota
        # Prioritize adding to sources with more files or specific important sources
        # For simplicity, add to the largest available source that isn't at its max yet
        sorted_sources_by_avail = sorted(
            all_available_speech.keys(),
            key=lambda s: len(all_available_speech[s]),
            reverse=True,
        )

        for src_name in sorted_sources_by_avail:
            if quota_diff == 0:
                break
            if not all_available_speech[src_name]:
                continue  # Skip if no files

            can_add = len(all_available_speech[src_name]) - quotas[src_name]
            if quota_diff > 0:  # Need to add more samples
                add_amount = min(quota_diff, can_add)
                quotas[src_name] += add_amount
                quota_diff -= add_amount
            elif quota_diff < 0:  # Need to remove samples
                remove_amount = min(abs(quota_diff), quotas[src_name])
                quotas[src_name] -= remove_amount
                quota_diff += remove_amount

        # Final check on quotas
        final_calculated_n_pos = sum(quotas.values())
        if final_calculated_n_pos != actual_n_pos:
            logger.warning(
                f"Quota calculation resulted in {final_calculated_n_pos} instead of {actual_n_pos}. Using {final_calculated_n_pos}."
            )
            actual_n_pos = final_calculated_n_pos

        for source_name, files in all_available_speech.items():
            num_to_select = quotas[source_name]
            if num_to_select == 0:
                logger.info(f"Selected 0 files from {source_name}.")
                continue

            shuffled_files = files.copy()
            random.shuffle(shuffled_files)
            # Special handling for FLEURS stratified sampling if q_fleurs was used that way before
            if (
                source_name == "fleurs" and "stratified_fleurs_sample" in globals()
            ):  # Check if func exists
                selected_files_for_source = stratified_fleurs_sample(
                    shuffled_files, n_samples=num_to_select
                )
            else:
                selected_files_for_source = shuffled_files[:num_to_select]

            selected_speech_final_pool.extend(selected_files_for_source)
            logger.info(
                f"Selected {len(selected_files_for_source)} files from {source_name} (Quota: {num_to_select})."
            )

        random.shuffle(selected_speech_final_pool)
        logger.info(
            f"Total selected speech files for positive samples: {len(selected_speech_final_pool)}"
        )
    else:
        logger.info("No positive samples will be generated as actual_n_pos is 0.")

    # --- 3. CREATE OUTPUT DIRECTORIES ---
    labels_root = out_pos.parent  # e.g., datasets/prepared/<split>/
    out_pos_labels = labels_root / "pos_labels"
    out_neg_labels = labels_root / "neg_labels"

    for p_dir in [out_pos, out_neg, out_pos_labels, out_neg_labels]:
        p_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Output directories created/ensured: {out_pos}, {out_neg}, {out_pos_labels}, {out_neg_labels}"
    )

    # Frame parameters (assuming these are globally defined or configured)
    hop_length = DEFAULT_HOP_LENGTH
    win_length = DEFAULT_WIN_LENGTH

    # --- 4. GENERATE POSITIVE SAMPLES ---
    # Counters for all positive samples
    total_pos_generated = [0]  # Use list for pass-by-reference
    total_pos_failures = [0]

    # 4.1. Clean Speech Samples
    # n_clean_speech = actual_n_pos # Original code generates this many 'clean' ones first
    # Let's define proportions for different positive types based on `actual_n_pos`
    # The original code has specific counts like 0.30, 0.20, 0.25, 0.25 of generated_clean
    # For simplicity and directness, let's aim for proportions of `actual_n_pos`
    # Or, follow the original logic: first generate `actual_n_pos` as 'clean' (or base), then derive others.
    # The original code's `create_clean_speech_sample_with_silero` seems to be the primary positive generator.
    # The subsequent "regular speech+noise", "overlap", etc. are *additional* samples.
    # This can lead to n_pos being much larger than initially specified if not careful.
    # The project plan mentioned on-the-fly SNR mixing. This script pre-generates.
    # which means `n_pos` might be the target for "clean_speech", and others are derived from that count.

    n_target_clean_speech = (
        actual_n_pos  # Let's assume n_pos refers to this primary set.
    )

    # Using a more generic sample generation loop:
    # Need to wrap the specific creation logic for `create_clean_speech_sample_with_silero`
    def clean_speech_creator_wrapper(idx):
        # This function needs `selected_speech_final_pool`, `duration_range`, etc. from outer scope
        # Or pass them via functools.partial if _generate_sample_batch is made more generic
        # For now, direct call:
        duration = random.uniform(*duration_range)
        audio, vad_labels_ = create_clean_speech_sample_with_silero(
            selected_speech_final_pool,
            duration,
            sample_rate,
            win_length,
            hop_length,
            vad_model_tuple,
        )
        if not validate_audio_sample(audio, sample_rate):
            return False  # Failure
        sf.write(out_pos / f"pos_clean_{idx:06}.wav", audio, sample_rate)
        np.save(out_pos_labels / f"pos_clean_{idx:06}_labels.npy", vad_labels_)
        return True  # Success

    _generate_sample_batch(
        n_target_clean_speech,
        "clean speech samples",
        clean_speech_creator_wrapper,
        total_pos_generated,
        total_pos_failures,
    )
    generated_clean_count = total_pos_generated[
        0
    ]  # Number of successfully generated clean samples

    # Proportions for other positive types, based on the *actually generated* clean speech count
    n_noise_speech_samples = int(round(generated_clean_count * 0.30))
    n_overlap_samples = int(round(generated_clean_count * 0.20))
    n_music_speech_samples = int(round(generated_clean_count * 0.25))
    n_snm_samples = int(round(generated_clean_count * 0.25))

    # 4.2. Augmented Speech + Noise Samples (Regular Positives)
    # This uses the create_speech_noise_sample_with_silero helper defined earlier
    def augmented_speech_noise_creator_wrapper(idx):
        if not selected_speech_final_pool:
            return False  # No source speech
        sp_path = random.choice(selected_speech_final_pool)
        return create_speech_noise_sample_with_silero(
            sp_path,
            duration_range,
            sample_rate,
            vad_model_tuple,
            musan_noise_files,  # Assuming musan_noise for this category
            snr_range,
            hop_length,
            win_length,
            out_pos,
            out_pos_labels,
            idx,
            file_prefix_base="pos_sn",
        )

    _generate_sample_batch(
        n_noise_speech_samples,
        "augmented speech+noise samples",
        augmented_speech_noise_creator_wrapper,
        total_pos_generated,
        total_pos_failures,
    )

    # 4.3. Overlapping Speech Samples
    def overlapping_speech_creator_wrapper(idx):
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_overlapping_speech_with_silero(
            selected_speech_final_pool,
            all_noise_sources,  # Or all_noise_sources? Original used musan_noise.
            duration,
            sample_rate,
            win_length,
            hop_length,
            vad_model_tuple,
        )
        if not validate_audio_sample(mix, sample_rate):
            return False
        sf.write(out_pos / f"pos_overlap_{idx:06}.wav", mix, sample_rate)
        np.save(out_pos_labels / f"pos_overlap_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        n_overlap_samples,
        "overlapping speech samples",
        overlapping_speech_creator_wrapper,
        total_pos_generated,
        total_pos_failures,
    )

    # 4.4. Speech + Music Samples
    def speech_music_creator_wrapper(idx):
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_speech_music_sample_with_silero(
            selected_speech_final_pool,
            musdb_music_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            vad_model_tuple,
        )
        if not validate_audio_sample(mix, sample_rate):
            return False
        sf.write(out_pos / f"pos_music_{idx:06}.wav", mix, sample_rate)
        np.save(out_pos_labels / f"pos_music_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        n_music_speech_samples,
        "speech+music samples",
        speech_music_creator_wrapper,
        total_pos_generated,
        total_pos_failures,
    )

    # 4.5. Speech + Noise + Music Samples (SNM)
    def snm_creator_wrapper(idx):
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_speech_noise_music_sample_with_silero(
            selected_speech_final_pool,
            all_noise_sources,
            musdb_music_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            vad_model_tuple,
        )
        if not validate_audio_sample(mix, sample_rate):
            return False
        sf.write(out_pos / f"pos_snm_{idx:06}.wav", mix, sample_rate)
        np.save(out_pos_labels / f"pos_snm_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        n_snm_samples,
        "speech+noise+music samples",
        snm_creator_wrapper,
        total_pos_generated,
        total_pos_failures,
    )

    logger.info(
        f"Total positive samples generated: {total_pos_generated[0]} (with {total_pos_failures[0]} failures)."
    )

    # --- 5. GENERATE NEGATIVE SAMPLES ---
    actual_n_neg = int(
        total_pos_generated[0] * 0.75
    )  # Adjust n_neg based on actual positives
    logger.info(
        f"Targeting {actual_n_neg} negative samples (75% of successful positives)."
    )

    neg_quotas = {
        "noise_only": int(round(actual_n_neg * neg_noise_ratio)),
        "esc50_only": int(round(actual_n_neg * neg_esc50_ratio)),
        "music_only": int(round(actual_n_neg * neg_music_ratio)),
        "noise_noise": int(round(actual_n_neg * neg_noise_noise_ratio)),
        "music_music": int(round(actual_n_neg * neg_music_music_ratio)),
        "urbansound": int(round(actual_n_neg * neg_urbansound_ratio)),
        "mixed_noise_music": int(round(actual_n_neg * neg_music_noise_ratio)),
    }
    for neg_type, q in neg_quotas.items():
        logger.info(f"  - Negative Quota - {neg_type}: {q}")

    total_neg_generated = [0]
    total_neg_failures = [0]

    # 5.1 Noise-only
    def noise_only_creator(idx):
        duration = random.uniform(*duration_range)
        audio, vad_labels_ = create_negative_sample(
            all_noise_sources,
            duration,
            sample_rate,
            win_length,
            hop_length,
            category="noise",
        )
        if not validate_negative_audio_sample(audio, sample_rate):
            return False
        sf.write(out_neg / f"neg_noise_{idx:06}.wav", audio, sample_rate)
        np.save(out_neg_labels / f"neg_noise_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["noise_only"],
        "noise-only negative samples",
        noise_only_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.2 ESC-50 only
    balanced_esc50 = (
        get_balanced_esc50_files(esc50_noise_files) if esc50_noise_files else []
    )

    def esc50_creator(idx):
        if not balanced_esc50:
            return False
        duration = random.uniform(*duration_range)
        audio, vad_labels_, category = create_esc50_negative_sample(
            balanced_esc50, duration, sample_rate, win_length, hop_length
        )
        if not validate_negative_audio_sample(audio, sample_rate):
            return False
        sf.write(out_neg / f"neg_esc50_{category}_{idx:06}.wav", audio, sample_rate)
        np.save(
            out_neg_labels / f"neg_esc50_{category}_{idx:06}_labels.npy", vad_labels_
        )
        return True

    _generate_sample_batch(
        neg_quotas["esc50_only"],
        "ESC-50 negative samples",
        esc50_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.3 Music-only
    def music_only_creator(idx):
        if not musdb_music_files:
            return False
        duration = random.uniform(*duration_range)
        audio, vad_labels_ = create_negative_sample(
            musdb_music_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            category="music",
        )
        if not validate_negative_audio_sample(audio, sample_rate):
            return False
        sf.write(out_neg / f"neg_music_{idx:06}.wav", audio, sample_rate)
        np.save(out_neg_labels / f"neg_music_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["music_only"],
        "music-only negative samples",
        music_only_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.4 Noise + Music (Mixed Negative)
    def mixed_neg_creator(idx):
        if not all_noise_sources or not musdb_music_files:
            return False
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_mixed_negative_sample(
            all_noise_sources,
            musdb_music_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
        )
        if not validate_negative_audio_sample(mix, sample_rate):
            return False
        sf.write(out_neg / f"neg_mixed_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_mixed_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["mixed_noise_music"],
        "mixed noise+music negative samples",
        mixed_neg_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.5 Noise + Noise
    def noise_noise_creator(idx):
        if not all_noise_sources or len(all_noise_sources) < 2:
            return False  # Need at least 2 files to mix
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_noise_noise_sample(
            all_noise_sources, duration, sample_rate, win_length, hop_length
        )
        if not validate_negative_audio_sample(mix, sample_rate):
            return False
        sf.write(out_neg / f"neg_noise_noise_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_noise_noise_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["noise_noise"],
        "noise+noise negative samples",
        noise_noise_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.6 Music + Music
    def music_music_creator(idx):
        if not musdb_music_files or len(musdb_music_files) < 2:
            return False  # Need at least 2 files
        duration = random.uniform(*duration_range)
        mix, vad_labels_ = create_music_music_sample(
            musdb_music_files, duration, sample_rate, win_length, hop_length
        )
        if not validate_negative_audio_sample(mix, sample_rate):
            return False
        sf.write(out_neg / f"neg_music_music_{idx:06}.wav", mix, sample_rate)
        np.save(out_neg_labels / f"neg_music_music_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["music_music"],
        "music+music negative samples",
        music_music_creator,
        total_neg_generated,
        total_neg_failures,
    )

    # 5.7 UrbanSound8K
    def urbansound_creator(idx):
        if not urbansound_noise_files:
            return False
        duration = random.uniform(*duration_range)
        audio, vad_labels_ = create_negative_sample(
            urbansound_noise_files,
            duration,
            sample_rate,
            win_length,
            hop_length,
            category="urbansound",
        )
        if not validate_negative_audio_sample(audio, sample_rate):
            return False
        sf.write(out_neg / f"neg_urbansound_{idx:06}.wav", audio, sample_rate)
        np.save(out_neg_labels / f"neg_urbansound_{idx:06}_labels.npy", vad_labels_)
        return True

    _generate_sample_batch(
        neg_quotas["urbansound"],
        "UrbanSound8K negative samples",
        urbansound_creator,
        total_neg_generated,
        total_neg_failures,
    )

    logger.info(
        f"Total negative samples generated: {total_neg_generated[0]} (with {total_neg_failures[0]} failures)."
    )

    # --- 6. FINAL SUMMARY ---
    # The summary logic from original code can be used here.
    final_total_generated = total_pos_generated[0] + total_neg_generated[0]
    final_total_failures = total_pos_failures[0] + total_neg_failures[0]
    success_rate = (
        final_total_generated / (final_total_generated + final_total_failures) * 100
        if (final_total_generated + final_total_failures) > 0
        else 0
    )

    logger.info("========== DATA-GENERATION COMPLETE ==========")
    logger.info(
        f"Total POSITIVE samples successfully generated: {total_pos_generated[0]}"
    )
    logger.info(
        f"Total NEGATIVE samples successfully generated: {total_neg_generated[0]}"
    )
    logger.info(f"Overall samples successfully generated: {final_total_generated}")
    logger.info(f"Overall samples failed validation/creation: {final_total_failures}")
    logger.info(f"Overall Success Rate: {success_rate:.2f}%")
    logger.info("==============================================")
