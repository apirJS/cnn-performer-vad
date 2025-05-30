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
        default=0.1,
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
        "--neg_music_noise_ratio",
        type=float,
        default=0.1,
        help="Fraction of negative samples that are music + noise",
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
        help="Fraction of negative samples that are UrbanSound8K sounds",
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

    # Data preparation arguments
    parser.add_argument(
        "--test_root", default="datasets", help="Root directory for test data"
    )
    parser.add_argument(
        "--prepare_data", action="store_true", help="Download and prepare test data"
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=1000,
        help="Number of test samples to generate (each for positive and negative)",
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
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate for audio processing",
    )

    # Model and evaluation arguments
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_manifest", help="Path to test manifest CSV (if already created)"
    )
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
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for evaluation",
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
    parser.add_argument(
        "--use_mel_cache",
        action="store_true",
        help="Cache mel spectrograms for faster processing",
    )
    parser.add_argument(
        "--mel_cache_dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory to cache mel spectrograms",
    )
    # Add these arguments in the main function's argument parser
    parser.add_argument(
        "--pytorch_model", 
        help="Path to PyTorch model (.pt) for evaluation"
    )
    parser.add_argument(
        "--quantized_model", 
        help="Path to quantized model (_quantized.pt) for evaluation"
    )
    parser.add_argument(
        "--compare_models", 
        action="store_true",
        help="Compare performance between original and quantized models"
    )
    # Output arguments
    parser.add_argument(
        "--output_dir",
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    # Add after other parser arguments
    parser.add_argument(
        "--two_stage_eval",
        action="store_true",
        help="Use separate validation set for threshold tuning",
    )
    parser.add_argument(
        "--n_validation",
        type=int,
        default=500,
        help="Number of validation samples for threshold tuning",
    )
    parser.add_argument(
        "--boundary_analysis",
        action="store_true",
        help="Perform detailed speech boundary detection analysis",
    )
    parser.add_argument(
        "--transition_window",
        type=int,
        default=10,
        help="Window size (frames) for transition analysis",
    )
    parser.add_argument(
        "--smoothing_window_ms", type=int, default=50, 
        help="Window size for median filtering (in milliseconds)"
    )
    parser.add_argument(
        "--min_segment_duration_ms", type=int, default=200, 
        help="Minimum speech segment duration (in milliseconds)"
    )
    parser.add_argument(
        "--max_gap_duration_ms", type=int, default=100, 
        help="Maximum gap between segments to merge (in milliseconds)"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.5,
        help="IoU threshold for segment matching"
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
    setup_inference_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if args.command == "prepare":
        from prepare_data import main as prepare_main
        prepare_main(args)  # Pass the parsed arguments to prepare_main
    elif args.command == "train":
        from train import main as train_main
        # Remove the 'command' attribute which train.py doesn't expect
        delattr(args, "command")
        train_main(args)  # Pass the parsed arguments
    elif args.command == "evaluate":
        # Convert CLI args to sys.argv format that evaluation.py expects
        from evaluation import main as evaluate_main
        
        # First clear sys.argv and add the script name
        sys.argv = [sys.argv[0]]
        
        # Add all arguments
        for arg_name, arg_value in vars(args).items():
            if arg_name != 'command':  # Skip the command itself
                if isinstance(arg_value, bool):
                    if arg_value:
                        sys.argv.append(f"--{arg_name}")
                elif isinstance(arg_value, list):  # Handle list arguments like duration_range
                    sys.argv.append(f"--{arg_name}")
                    for item in arg_value:
                        sys.argv.append(str(item))
                elif arg_value is not None:
                    sys.argv.append(f"--{arg_name}")
                    sys.argv.append(str(arg_value))
        
        # Now call the evaluation main function which will parse these args
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
