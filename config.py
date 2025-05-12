#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  config.py - Shared configuration for VAD project
# ───────────────────────────────────────────────────────────────────────
"""Common configuration settings for VAD project."""

# Audio processing parameters
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_MELS = 80
DEFAULT_N_FFT = 512
DEFAULT_WIN_LENGTH = 512
DEFAULT_HOP_LENGTH = 160

# File paths and directories
DEFAULT_ROOT_DIR = "datasets"
DEFAULT_CACHE_DIR = "mel_cache"
DEFAULT_MODEL_EXPORT_PATH = "vad_model.pt"
DEFAULT_LOG_DIR = "lightning_logs"

# Training parameters
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_MAX_EPOCHS = 20
DEFAULT_POS_WEIGHT = 2.0
DEFAULT_WARMUP_EPOCHS = 0.5
DEFAULT_GRADIENT_CLIP_VAL = 1.0

# Model parameters
DEFAULT_DIMENSION = 384
DEFAULT_LAYERS = 4
DEFAULT_HEADS = 6
DEFAULT_MAX_FRAMES = 2000

# Data augmentation parameters
DEFAULT_TIME_MASK_MAX = 20
DEFAULT_FREQ_MASK_MAX = 10

# URLs for datasets
LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
]
TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
