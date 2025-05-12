#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  evaluation.py - Metrics visualization for MEL-spectrogram VAD model
# ───────────────────────────────────────────────────────────────────────
import argparse
import csv
import random
import pathlib
import subprocess
import sys
import json
import logging
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
)

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

import librosa
import soundfile as sf
import pytorch_lightning as pl

# Import model and dataset definitions from train.py
from train import (
    VADLightning,
    CSVMelDataset,
    collate_pad,
    download_and_extract,
    download_file,
    make_pos_neg,
    seed_everything,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Define URLs (same as in prompt)
TEST_CLEAN_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"


def get_dataset_paths(root):
    """Returns consistent paths to shared datasets and test-specific directories."""
    root_path = pathlib.Path(root).expanduser()

    return {
        "root": root_path,
        "librispeech": root_path / "LibriSpeech",
        "musan": root_path / "musan",
        "test_prep": root_path / "test_prepared",
        "test_manifest": root_path / "manifest_test.csv",
    }


def prepare_test_data(args):
    """Download, extract, and prepare audio data for VAD evaluation."""
    logger.info("Starting test data preparation process")

    paths = get_dataset_paths(args.test_root)
    root = paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using root directory: {root}")

    # Track state in a JSON file
    state_file = root / "evaluation_state.json"
    state = {}
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            logger.info(f"Found existing preparation state: {state}")
        except Exception as e:
            logger.error(f"Error loading state file: {e}")

    # Early check if data directories already exist
    test_clean_dir = root / "LibriSpeech" / "test-clean"
    musan_dir = root / "musan"

    if test_clean_dir.exists() and musan_dir.exists():
        logger.info("LibriSpeech test-clean and MUSAN directories already exist")
        # Ensure state reflects that downloads are complete
        if not state.get("downloads_complete", False):
            logger.info("Updating state file to reflect existing data")
            state["downloads_complete"] = True
            with open(state_file, "w") as f:
                json.dump(state, f)
        logger.info("Skipping download step")
    # Download data if not already done
    elif not state.get("downloads_complete", False):
        logger.info("Beginning download of LibriSpeech test-clean and MUSAN datasets")
        logger.info(f"Processing LibriSpeech URL: {TEST_CLEAN_URL}")
        download_and_extract(TEST_CLEAN_URL, root)
        logger.info(f"Processing MUSAN URL: {MUSAN_URL}")
        download_and_extract(MUSAN_URL, root)
        state["downloads_complete"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("All downloads completed successfully")
    else:
        # Verify files actually exist even if state claims downloads are complete
        if not test_clean_dir.exists() or not musan_dir.exists():
            logger.warning(
                "State claims downloads complete, but directories are missing!"
            )
            logger.info("Resetting state and starting downloads again")
            state["downloads_complete"] = False

            # Download missing directories
            if not test_clean_dir.exists():
                logger.info("LibriSpeech test-clean missing, downloading...")
                download_and_extract(TEST_CLEAN_URL, root)

            if not musan_dir.exists():
                logger.info("MUSAN directory missing, downloading...")
                download_and_extract(MUSAN_URL, root)

            state["downloads_complete"] = True
            with open(state_file, "w") as f:
                json.dump(state, f)
            logger.info("Missing files downloaded successfully")
        else:
            logger.info("Downloads already completed and files exist, skipping")

    # Generate positive/negative samples if not done
    if not state.get("samples_generated", False):
        logger.info("Generating test positive and negative audio samples")
        libri_root = root / "LibriSpeech" / "test-clean"
        musan_root = root / "musan"
        prep = root / "prepared"

        # Updated paths for test split
        test_pos, test_neg = prep / "test" / "pos", prep / "test" / "neg"

        make_pos_neg(
            libri_root,
            musan_root,
            test_pos,
            test_neg,
            args.n_test,  # Equal number of positives and negatives
            args.n_test,
            duration_range=args.duration_range,
            sample_rate=args.sample_rate,
            split_name="test",  # Add split name
        )
        state["samples_generated"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("Sample generation completed")
    else:
        logger.info("Test samples already generated, skipping")

    # Create test manifest if not done
    if not state.get("manifest_created", False):
        logger.info("Creating test manifest")
        create_test_manifest(root / "prepared" / "test", root / "manifest_test.csv")
        state["manifest_created"] = True
        with open(state_file, "w") as f:
            json.dump(state, f)
        logger.info("Manifest creation completed")
    else:
        logger.info("Test manifest already created, skipping")

    logger.info("✅ Test data preparation completed successfully")
    print("✅ Test data ready")

    return root / "manifest_test.csv"


def create_test_manifest(prep_dir: pathlib.Path, manifest_path: pathlib.Path):
    """Create test manifest with paths to audio and frame-level labels."""
    logger.info(f"Creating test manifest file")

    pos_clips = list(prep_dir.joinpath("pos").glob("*.wav"))
    neg_clips = list(prep_dir.joinpath("neg").glob("*.wav"))

    logger.info(
        f"Found {len(pos_clips)} positive and {len(neg_clips)} negative samples"
    )

    # Combine and shuffle samples
    all_clips = pos_clips + neg_clips
    random.shuffle(all_clips)

    logger.info(f"Writing manifest: {manifest_path}")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "frame_labels"])
        for p in all_clips:
            is_speech = 1 if "pos_" in p.name else 0
            # Path to corresponding frame labels with updated directory names
            label_dir = "test_pos_labels" if is_speech else "test_neg_labels"
            frame_label_path = prep_dir.parent / label_dir / f"{p.stem}_labels.npy"
            w.writerow([p, is_speech, frame_label_path])

    logger.info(f"Created test manifest with {len(all_clips)} entries")
    return manifest_path


def evaluate_model(model_path, test_manifest, args):
    """Evaluate a trained VAD model and return frame-level predictions and labels."""
    logger.info(f"Loading model from {model_path}")

    # Load model with custom way to handle hyperparameters
    try:
        # First try standard loading
        model = VADLightning.load_from_checkpoint(model_path)
    except AttributeError as e:
        # If we get dictionary error, load with a workaround
        logger.info("Standard loading failed, trying with hyperparameter conversion")

        # Load the checkpoint directly
        checkpoint = torch.load(model_path, map_location=args.device)
        hparams = checkpoint.get("hyper_parameters", {})

        # Convert dict to argparse.Namespace to match expected format
        import argparse

        hp_namespace = argparse.Namespace(**hparams)

        # Create model with converted namespace
        model = VADLightning(hp_namespace)

        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Model loaded with hyperparameter conversion")

    model.eval()
    model.to(args.device)
    logger.info(f"Model loaded to {args.device}")

    # Rest of the function remains the same
    # Create test dataset
    logger.info(f"Creating test dataset from {test_manifest}")
    test_dataset = CSVMelDataset(
        test_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=args.mel_cache_dir if args.use_mel_cache else None,
    )

    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_pad(batch, args.max_frames),
    )
    logger.info(f"Evaluating model on {len(test_dataset)} test samples")

    # Evaluate
    all_preds = []
    all_labels = []
    all_clip_labels = []
    all_clip_preds = []

    with torch.no_grad():
        for batch_idx, (mel, labels, mask) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(test_loader)}")

            mel = mel.to(args.device)
            logits = model(mel)
            preds = torch.sigmoid(logits)

            # Use mask to extract valid frames only
            for i in range(mel.shape[0]):
                valid_frames = mask[i].sum().item()
                sample_preds = preds[i, :valid_frames].cpu().numpy()
                sample_labels = labels[i, :valid_frames].cpu().numpy()

                all_preds.append(sample_preds)
                all_labels.append(sample_labels)

                # Calculate clip-level prediction (average of frame predictions)
                clip_pred = sample_preds.mean()
                # Determine clip-level ground truth (if any frame is 1, clip is positive)
                clip_label = 1.0 if sample_labels.max() > 0.5 else 0.0

                all_clip_preds.append(clip_pred)
                all_clip_labels.append(clip_label)

    # Concatenate results
    frame_preds = np.concatenate(all_preds)
    frame_labels = np.concatenate(all_labels)
    clip_preds = np.array(all_clip_preds)
    clip_labels = np.array(all_clip_labels)

    logger.info(
        f"Evaluation complete with {len(frame_preds)} valid frames and {len(clip_preds)} clips"
    )
    return frame_preds, frame_labels, clip_preds, clip_labels


def plot_metrics(frame_preds, frame_labels, clip_preds, clip_labels, output_dir):
    """Generate and save visualization of various metrics."""
    logger.info("Generating performance visualizations")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set Seaborn style
    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12})

    # ===== FRAME-LEVEL METRICS =====

    # 1. ROC Curve
    logger.info("Plotting frame-level ROC curve")
    fpr, tpr, thresholds = roc_curve(frame_labels, frame_preds)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Frame-Level Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_roc_curve.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curve
    logger.info("Plotting frame-level Precision-Recall curve")
    precision, recall, thresholds = precision_recall_curve(frame_labels, frame_preds)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, lw=2, label=f"PR curve (AUC = {pr_auc:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Frame-Level Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_precision_recall_curve.png", dpi=300)
    plt.close()

    # 3. Confusion Matrix at optimal threshold
    logger.info("Creating frame-level confusion matrix at optimal threshold")
    # Find threshold that maximizes F1 score
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = (
        thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    )

    binary_preds = (frame_preds >= optimal_threshold).astype(int)
    cm = confusion_matrix(frame_labels, binary_preds)
    
    # Calculate correct F1 score from confusion matrix elements
    tn, fp, fn, tp = cm.ravel()
    precision_from_cm = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_from_cm = tp / (tp + fn) if (tp + fn) > 0 else 0
    frame_f1 = 2 * precision_from_cm * recall_from_cm / (precision_from_cm + recall_from_cm) if (precision_from_cm + recall_from_cm) > 0 else 0

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Speech", "Speech"],
        yticklabels=["No Speech", "Speech"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Frame-Level Confusion Matrix (threshold = {optimal_threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output_dir / "frame_confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Histogram of predictions
    logger.info("Creating frame-level prediction histogram")
    plt.figure(figsize=(12, 6))

    # Separate predictions for positive and negative samples
    pos_preds = frame_preds[frame_labels == 1]
    neg_preds = frame_preds[frame_labels == 0]

    plt.hist(neg_preds, bins=50, alpha=0.5, label="Non-speech frames", color="blue")
    plt.hist(pos_preds, bins=50, alpha=0.5, label="Speech frames", color="red")

    plt.axvline(
        x=optimal_threshold,
        color="k",
        linestyle="--",
        label=f"Optimal threshold: {optimal_threshold:.3f}",
    )

    plt.xlabel("Prediction Scores")
    plt.ylabel("Count")
    plt.title("Distribution of Frame-Level Prediction Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "frame_prediction_histogram.png", dpi=300)
    plt.close()

    # 5. Threshold vs Metrics
    logger.info("Plotting frame-level threshold vs metrics")
    # Calculate metrics at various thresholds
    thresholds_for_plot = np.linspace(0, 1, 100)
    f1_values = []
    precision_values = []
    recall_values = []

    for threshold in thresholds_for_plot:
        binary_preds = (frame_preds >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(frame_labels, binary_preds).ravel()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision_values.append(prec)
        recall_values.append(rec)
        f1_values.append(f1)

    plt.figure(figsize=(12, 8))
    plt.plot(thresholds_for_plot, precision_values, label="Precision", lw=2)
    plt.plot(thresholds_for_plot, recall_values, label="Recall", lw=2)
    plt.plot(thresholds_for_plot, f1_values, label="F1 Score", lw=2)
    plt.axvline(
        x=optimal_threshold,
        color="k",
        linestyle="--",
        label=f"Optimal threshold: {optimal_threshold:.3f}",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Frame-Level Threshold vs. Metrics")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "frame_threshold_metrics.png", dpi=300)
    plt.close()

    # ===== CLIP-LEVEL METRICS =====

    # 1. ROC Curve for clips
    logger.info("Plotting clip-level ROC curve")
    clip_fpr, clip_tpr, clip_thresholds = roc_curve(clip_labels, clip_preds)
    clip_roc_auc = auc(clip_fpr, clip_tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(clip_fpr, clip_tpr, lw=2, label=f"ROC curve (AUC = {clip_roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Clip-Level Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "clip_roc_curve.png", dpi=300)
    plt.close()

    # 2. Clip-level Precision-Recall Curve
    logger.info("Plotting clip-level Precision-Recall curve")
    clip_precision, clip_recall, clip_pr_thresholds = precision_recall_curve(
        clip_labels, clip_preds
    )
    clip_pr_auc = auc(clip_recall, clip_precision)

    plt.figure(figsize=(10, 8))
    plt.plot(
        clip_recall, clip_precision, lw=2, label=f"PR curve (AUC = {clip_pr_auc:.3f})"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Clip-Level Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "clip_precision_recall_curve.png", dpi=300)
    plt.close()

    # 3. Find optimal clip threshold and create confusion matrix
    clip_f1_scores = (
        2 * clip_precision * clip_recall / (clip_precision + clip_recall + 1e-10)
    )
    clip_optimal_idx = np.argmax(clip_f1_scores)
    clip_optimal_threshold = (
        clip_pr_thresholds[clip_optimal_idx]
        if clip_optimal_idx < len(clip_pr_thresholds)
        else 0.5
    )

    clip_binary_preds = (clip_preds >= clip_optimal_threshold).astype(int)
    clip_cm = confusion_matrix(clip_labels, clip_binary_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        clip_cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Speech", "Speech"],
        yticklabels=["No Speech", "Speech"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Clip-Level Confusion Matrix (threshold = {clip_optimal_threshold:.3f})")
    plt.tight_layout()
    plt.savefig(output_dir / "clip_confusion_matrix.png", dpi=300)
    plt.close()

    # 4. Scatter plot of clip predictions
    logger.info("Creating clip prediction scatter plot")
    plt.figure(figsize=(10, 8))

    plt.scatter(
        range(len(clip_preds)),
        clip_preds,
        c=[("r" if l > 0.5 else "b") for l in clip_labels],
        alpha=0.6,
    )

    plt.axhline(
        y=clip_optimal_threshold,
        color="k",
        linestyle="--",
        label=f"Optimal threshold: {clip_optimal_threshold:.3f}",
    )

    plt.xlabel("Clip Index")
    plt.ylabel("Prediction Score")
    plt.title("Clip-Level Predictions (red=speech, blue=non-speech)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "clip_predictions.png", dpi=300)
    plt.close()

    # 5. Summary plot of key metrics
    logger.info("Creating summary metrics plot")
    metrics_labels = [
        "Frame ROC-AUC",
        "Frame PR-AUC",
        "Frame F1",
        "Clip ROC-AUC",
        "Clip PR-AUC",
        "Clip F1",
    ]

    clip_f1 = f1_score(clip_labels, clip_binary_preds)

    metrics_values = [roc_auc, pr_auc, frame_f1, clip_roc_auc, clip_pr_auc, clip_f1]

    plt.figure(figsize=(12, 8))
    bars = plt.bar(metrics_labels, metrics_values, color="steelblue")

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    plt.ylim([0, 1.1])
    plt.ylabel("Score")
    plt.title("Summary of Evaluation Metrics")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "summary_metrics.png", dpi=300)
    plt.close()

    # Save metrics to file
    metrics = {
        "frame_level": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "optimal_threshold": float(optimal_threshold),
            "f1_score": float(frame_f1),
        },
        "clip_level": {
            "roc_auc": float(clip_roc_auc),
            "pr_auc": float(clip_pr_auc),
            "optimal_threshold": float(clip_optimal_threshold),
            "f1_score": float(clip_f1),
        },
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"All visualizations saved to {output_dir}")
    logger.info(f"Metrics: {metrics}")

    return metrics


def main():
    logger.info("Starting VAD evaluation script")

    parser = argparse.ArgumentParser("VAD Model Evaluation & Visualization")

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
        default=16000,
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
        "--n_mels", type=int, default=80, help="Number of mel bands in spectrogram"
    )
    parser.add_argument(
        "--n_fft", type=int, default=400, help="FFT size for spectrogram"
    )
    parser.add_argument(
        "--hop", type=int, default=160, help="Hop length for spectrogram"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=2000,
        help="Maximum number of frames per sequence",
    )
    parser.add_argument(
        "--use_mel_cache",
        action="store_true",
        help="Cache mel spectrograms for faster processing",
    )
    parser.add_argument(
        "--mel_cache_dir",
        default="test_mel_cache",
        help="Directory to cache mel spectrograms",
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

    args = parser.parse_args()
    logger.info(f"Parsed arguments: {args}")

    # Windows compatibility: Disable multiprocessing on Windows
    if sys.platform == "win32" and args.num_workers > 0:
        logger.warning(
            "Running on Windows - setting num_workers=0 to avoid multiprocessing issues"
        )
        args.num_workers = 0
        print("⚠️ Windows detected: Setting num_workers=0 for compatibility")

    # Set random seed
    seed_everything(args.seed)

    # Prepare test data if needed
    if args.prepare_data or not args.test_manifest:
        logger.info("Preparing test data")
        test_manifest = prepare_test_data(args)
        args.test_manifest = str(test_manifest)

    # Check if test manifest exists
    if not pathlib.Path(args.test_manifest).exists():
        logger.error(f"Test manifest missing: {args.test_manifest}")
        sys.exit("❌ Test manifest missing. Use --prepare_data first.")

    # Check if model exists
    if not pathlib.Path(args.model_path).exists():
        logger.error(f"Model checkpoint missing: {args.model_path}")
        sys.exit("❌ Model checkpoint missing.")

    # Evaluate model
    logger.info("Starting model evaluation")
    frame_preds, frame_labels, clip_preds, clip_labels = evaluate_model(
        args.model_path, args.test_manifest, args
    )

    # Generate visualizations
    logger.info("Generating metrics visualizations")
    metrics = plot_metrics(
        frame_preds, frame_labels, clip_preds, clip_labels, args.output_dir
    )

    # Print summary
    print(f"\n✅ Evaluation complete!")
    print(f"📊 Frame-level metrics:")
    print(f"  - ROC AUC: {metrics['frame_level']['roc_auc']:.4f}")
    print(f"  - PR AUC: {metrics['frame_level']['pr_auc']:.4f}")
    print(f"  - F1 score: {metrics['frame_level']['f1_score']:.4f}")
    print(f"  - Optimal threshold: {metrics['frame_level']['optimal_threshold']:.4f}")
    print(f"\n📊 Clip-level metrics:")
    print(f"  - ROC AUC: {metrics['clip_level']['roc_auc']:.4f}")
    print(f"  - PR AUC: {metrics['clip_level']['pr_auc']:.4f}")
    print(f"  - F1 score: {metrics['clip_level']['f1_score']:.4f}")
    print(f"  - Optimal threshold: {metrics['clip_level']['optimal_threshold']:.4f}")
    print(f"\nVisualizations saved to: {args.output_dir}")

    logger.info("Evaluation script completed successfully")


if __name__ == "__main__":
    main()
