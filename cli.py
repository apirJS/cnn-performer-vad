#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  cli.py - Unified command-line interface for VAD tools
# ───────────────────────────────────────────────────────────────────────
import argparse
import sys
import logging
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from config import *


def setup_prepare_parser(subparsers):
    """Setup the data preparation command parser."""
    parser = subparsers.add_parser("prepare", help="Download and prepare VAD datasets")

    # Base arguments
    parser.add_argument(
        "--root",
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
        default=0.20,
        help="Fraction of positive samples from FLEURS",
    )
    parser.add_argument(
        "--fraction_libri",
        type=float,
        default=0.60,
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
        help="Use ESC-50 (noise) and VocalSet (singing) datasets in addition to LibriSpeech and MUSAN",
    )
    parser.add_argument(
        "--neg_noise_ratio",
        type=float,
        default=0.30,
        help="Fraction of negative samples that are pure noise",
    )
    parser.add_argument(
        "--neg_esc50_ratio",
        type=float,
        default=0.20,
        help="Fraction of negative samples that are ESC-50 sounds",
    )
    parser.add_argument(
        "--neg_music_ratio",
        type=float,
        default=0.25,
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
        "--use_silero_vad",
        action="store_true",
        default=True,
        help="Use Silero VAD for generating frame-level labels",
    )

    return parser


def setup_train_parser(subparsers):
    """Setup the training command parser."""
    parser = subparsers.add_parser("train", help="Train a VAD model")

    # Add all the training arguments from train.py
    # You can import the build_parser function from train.py
    # and use that to add arguments
    from train import build_parser

    for action in build_parser()._actions:
        if action.dest != "help":
            parser._add_action(action)

    return parser


def setup_eval_parser(subparsers):
    """Setup the evaluation command parser."""
    parser = subparsers.add_parser("evaluate", help="Evaluate a trained VAD model")

    # Model and data
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_root", default=DEFAULT_ROOT_DIR, help="Root directory for test data"
    )
    parser.add_argument(
        "--test_manifest",
        help="Path to test manifest CSV (created automatically if not provided)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=500,
        help="Number of test samples to generate (if preparing data)",
    )
    parser.add_argument(
        "--prepare_data",
        action="store_true",
        help="Prepare test data before evaluation",
    )

    # Mel parameters
    parser.add_argument(
        "--n_mels",
        type=int,
        default=DEFAULT_N_MELS,
        help="Number of mel bands in spectrogram",
    )
    parser.add_argument(
        "--n_fft", type=int, default=DEFAULT_N_FFT, help="FFT size for spectrogram"
    )
    parser.add_argument(
        "--hop", type=int, default=DEFAULT_HOP_LENGTH, help="Hop length for spectrogram"
    )

    # Evaluation settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda/cpu)",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of frames per sequence",
    )

    # Caching
    parser.add_argument(
        "--use_mel_cache",
        action="store_true",
        help="Use cached mel spectrograms if available",
    )
    parser.add_argument(
        "--mel_cache_dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory for cached mel spectrograms",
    )

    return parser


def setup_inference_parser(subparsers):
    """Setup the inference command parser."""
    parser = subparsers.add_parser(
        "inference", help="Run inference with trained VAD model"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir", help="Directory of audio files to run VAD on"
    )
    input_group.add_argument(
        "--input_files", nargs="+", help="List of audio file paths to run VAD on"
    )

    # Model and evaluation
    parser.add_argument(
        "--labels_dir", help="Directory containing frame label .npy files (optional)"
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to trained VAD model (checkpoint .ckpt or TorchScript .pt)",
    )

    # Audio processing parameters
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate expected by model",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=DEFAULT_N_FFT,
        help="n_fft for mel spectrogram (must match training)",
    )
    parser.add_argument(
        "--hop_length",
        type=int,
        default=DEFAULT_HOP_LENGTH,
        help="Hop length for mel spectrogram",
    )
    parser.add_argument(
        "--win_length",
        type=int,
        default=DEFAULT_WIN_LENGTH,
        help="Window length for mel spectrogram",
    )
    parser.add_argument(
        "--n_mels", type=int, default=DEFAULT_N_MELS, help="Number of mel bands"
    )

    # Processing options
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing audio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--output_dir", help="Directory to save prediction results (optional)"
    )

    return parser


def main():
    """Main entry point for VAD CLI."""
    parser = argparse.ArgumentParser("Voice Activity Detection (VAD) Toolkit")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add command subparsers
    setup_prepare_parser(subparsers)
    setup_train_parser(subparsers)
    setup_eval_parser(subparsers)
    setup_inference_parser(subparsers)  # Add the new inference parser

    # Parse arguments
    args = parser.parse_args()

    if args.command == "prepare":
        from prepare_data import main as prepare_main

        prepare_main(args)  # Pass the parsed arguments to prepare_main
    elif args.command == "train":
        from train import main as train_main

        train_main()
    elif args.command == "evaluate":
        from evaluation import main as evaluate_main

        evaluate_main()
    elif args.command == "inference":
        from inference import main as inference_main

        return inference_main(args)  # Return the exit code from inference
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
