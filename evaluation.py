#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  evaluation.py - Metrics visualization for MEL-spectrogram VAD model
# ───────────────────────────────────────────────────────────────────────
from config import (
    DEFAULT_N_MELS,
    DEFAULT_N_FFT,
    DEFAULT_HOP_LENGTH,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_MAX_FRAMES,
    DEFAULT_CACHE_DIR,
    DEFAULT_BATCH_SIZE,
)

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader

import argparse
import csv
import random
import pathlib
import sys
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

# Import from train.py and prepare_data.py
from data import CSVMelDataset, collate_pad
from models import VADLightning

from prepare_data import seed_everything, prepare_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    # Use the new modular preparation function
    manifest_path = prepare_dataset(
        args.test_root,
        split="test",
        n_pos=args.n_test,
        n_neg=args.n_test,
        duration_range=args.duration_range,
        sample_rate=args.sample_rate,
        force_rebuild=args.prepare_data,  # Only force rebuild if explicitly requested
    )

    return manifest_path


def prepare_validation_test_split(args):
    """Prepare separate validation and test sets for unbiased evaluation."""
    logger.info("Preparing separate validation and test sets")

    # Prepare validation data for threshold tuning
    val_manifest = prepare_dataset(
        args.test_root,
        split="validation",
        n_pos=args.n_validation,
        n_neg=args.n_validation,
        duration_range=args.duration_range,
        sample_rate=args.sample_rate,
        force_rebuild=args.prepare_data,
    )

    # Prepare test data for final evaluation
    test_manifest = prepare_dataset(
        args.test_root,
        split="test",
        n_pos=args.n_test,
        n_neg=args.n_test,
        duration_range=args.duration_range,
        sample_rate=args.sample_rate,
        force_rebuild=args.prepare_data,
    )

    logger.info(f"Created validation manifest with {args.n_validation*2} samples")
    logger.info(f"Created test manifest with {args.n_test*2} samples")

    return val_manifest, test_manifest


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


def find_segments(binary_sequence):
    """Find continuous segments in a binary sequence.

    Returns a list of tuples (start_idx, end_idx) for each segment.
    """
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(binary_sequence):
        if val and not in_segment:
            # Start of new segment
            in_segment = True
            start = i
        elif not val and in_segment:
            # End of segment
            segments.append((start, i - 1))
            in_segment = False

    # Handle case where sequence ends during a segment
    if in_segment:
        segments.append((start, len(binary_sequence) - 1))

    return segments


def analyze_speech_boundaries(
    frame_preds, frame_labels, threshold=0.5, tolerance_frames=3
):
    """Analyze speech boundary detection performance.

    Args:
        frame_preds: Frame-level prediction scores (0-1)
        frame_labels: Ground truth frame labels (0 or 1)
        threshold: Classification threshold
        tolerance_frames: Number of frames allowed for boundary error

    Returns:
        Dictionary of boundary metrics
    """
    logger.info("Analyzing speech boundary detection performance")
    boundary_metrics = {}

    # Convert to binary predictions
    binary_preds = (frame_preds > threshold).astype(int)

    # Group frame sequences by continuous clips
    # For this demo, we'll assume all frames are from a single continuous sequence
    # In a real implementation, you'd need to process each clip separately

    # Find true speech segments
    true_segments = find_segments(frame_labels > 0.5)
    logger.info(f"Found {len(true_segments)} true speech segments")

    # Find predicted speech segments
    pred_segments = find_segments(binary_preds)
    logger.info(f"Found {len(pred_segments)} predicted speech segments")

    # Count matched segments (with overlap)
    matched_true_segments = 0
    matched_pred_segments = 0

    # Calculate boundary metrics
    onset_errors = []  # Frames of error at speech onset
    offset_errors = []  # Frames of error at speech offset

    # For each true segment, find the best matching predicted segment
    for true_start, true_end in true_segments:
        best_iou = 0
        best_pred_segment = None

        for pred_start, pred_end in pred_segments:
            # Calculate overlap
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)

            if overlap_start <= overlap_end:  # There is overlap
                # Calculate IoU (Intersection over Union)
                intersection = overlap_end - overlap_start + 1
                union = (
                    (true_end - true_start + 1)
                    + (pred_end - pred_start + 1)
                    - intersection
                )
                iou = intersection / union

                if iou > best_iou:
                    best_iou = iou
                    best_pred_segment = (pred_start, pred_end)

        # If we found a matching segment
        if best_iou > 0.5:  # Consider it a match if IoU > 0.5
            matched_true_segments += 1

            # Calculate boundary errors
            pred_start, pred_end = best_pred_segment
            onset_error = pred_start - true_start  # Positive means late detection
            offset_error = pred_end - true_end  # Positive means late cutoff

            onset_errors.append(onset_error)
            offset_errors.append(offset_error)

    # Count matched predicted segments
    for pred_start, pred_end in pred_segments:
        for true_start, true_end in true_segments:
            # Calculate overlap
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)

            if overlap_start <= overlap_end:  # There is overlap
                # Calculate IoU
                intersection = overlap_end - overlap_start + 1
                union = (
                    (true_end - true_start + 1)
                    + (pred_end - pred_start + 1)
                    - intersection
                )
                iou = intersection / union

                if iou > 0.5:  # Consider it a match if IoU > 0.5
                    matched_pred_segments += 1
                    break

    # Calculate metrics
    boundary_metrics["total_true_segments"] = len(true_segments)
    boundary_metrics["total_pred_segments"] = len(pred_segments)
    boundary_metrics["matched_true_segments"] = matched_true_segments
    boundary_metrics["matched_pred_segments"] = matched_pred_segments

    # Segment-level metrics
    boundary_metrics["segment_recall"] = matched_true_segments / max(
        1, len(true_segments)
    )
    boundary_metrics["segment_precision"] = matched_pred_segments / max(
        1, len(pred_segments)
    )

    if matched_true_segments > 0:
        boundary_metrics["segment_f1"] = (
            2
            * boundary_metrics["segment_recall"]
            * boundary_metrics["segment_precision"]
            / (
                boundary_metrics["segment_recall"]
                + boundary_metrics["segment_precision"]
            )
        )
    else:
        boundary_metrics["segment_f1"] = 0.0

    # Boundary timing metrics
    if onset_errors:
        boundary_metrics["mean_onset_error"] = np.mean(onset_errors)
        boundary_metrics["mean_abs_onset_error"] = np.mean(np.abs(onset_errors))
        boundary_metrics["max_onset_error"] = np.max(np.abs(onset_errors))

    if offset_errors:
        boundary_metrics["mean_offset_error"] = np.mean(offset_errors)
        boundary_metrics["mean_abs_offset_error"] = np.mean(np.abs(offset_errors))
        boundary_metrics["max_offset_error"] = np.max(np.abs(offset_errors))

    return boundary_metrics


def plot_transition_errors(
    frame_preds, frame_labels, output_dir, threshold=0.5, window=10
):
    """Visualize error patterns around speech transitions.

    Args:
        frame_preds: Frame-level prediction scores
        frame_labels: Ground truth frame labels
        output_dir: Directory to save visualization
        threshold: Classification threshold
        window: Window size around transitions to analyze
    """
    logger.info("Analyzing errors around speech transitions")
    output_dir = pathlib.Path(output_dir)

    # Convert to binary predictions
    binary_preds = (frame_preds > threshold).astype(int)

    # Find transitions in ground truth
    transitions = []
    transition_types = []  # 1 for onset, 0 for offset

    for i in range(1, len(frame_labels)):
        if frame_labels[i] != frame_labels[i - 1]:
            transitions.append(i)
            transition_types.append(1 if frame_labels[i] > 0.5 else 0)

    logger.info(f"Found {len(transitions)} speech transitions")

    if not transitions:
        logger.warning("No transitions found, skipping transition analysis")
        return

    # Collect prediction errors around transitions
    onset_errors = []  # Errors around speech onset
    offset_errors = []  # Errors around speech offset

    for t, t_type in zip(transitions, transition_types):
        # Get window around transition
        start = max(0, t - window)
        end = min(len(frame_labels), t + window)

        # Get prediction errors in window
        window_preds = binary_preds[start:end]
        window_labels = frame_labels[start:end]
        window_errors = (window_preds != window_labels).astype(float)

        # Align all windows so transition is at index 'window'
        aligned_errors = np.zeros(2 * window)
        offset = window - (t - start)  # Adjust for truncated windows
        aligned_errors[offset : offset + (end - start)] = window_errors

        if t_type == 1:  # onset
            onset_errors.append(aligned_errors)
        else:  # offset
            offset_errors.append(aligned_errors)

    # Average errors at each position relative to transition
    mean_onset_errors = (
        np.mean(onset_errors, axis=0) if onset_errors else np.zeros(2 * window)
    )
    mean_offset_errors = (
        np.mean(offset_errors, axis=0) if offset_errors else np.zeros(2 * window)
    )

    # Plot
    plt.figure(figsize=(12, 6))
    x = np.arange(-window, window)
    plt.plot(x, mean_onset_errors, label="Speech Onset Errors", color="blue")
    plt.plot(x, mean_offset_errors, label="Speech Offset Errors", color="red")
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Frames relative to transition (0 = transition point)")
    plt.ylabel("Error rate")
    plt.title("VAD Errors Around Speech Transitions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "transition_errors.png", dpi=300)
    plt.close()

    logger.info(
        f"Transition error analysis saved to {output_dir / 'transition_errors.png'}"
    )


def plot_metrics(
    frame_preds,
    frame_labels,
    clip_preds,
    clip_labels,
    output_dir,
    fixed_thresholds=None,
):
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
    # Find threshold that maximizes F1 score, or use fixed threshold if provided
    if fixed_thresholds and "frame" in fixed_thresholds:
        optimal_threshold = fixed_thresholds["frame"]
        logger.info(f"Using fixed frame threshold: {optimal_threshold}")
    else:
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        )
        logger.info(f"Found optimal frame threshold: {optimal_threshold}")

    binary_preds = (frame_preds >= optimal_threshold).astype(int)
    cm = confusion_matrix(frame_labels, binary_preds)

    # Calculate correct F1 score from confusion matrix elements
    tn, fp, fn, tp = cm.ravel()
    precision_from_cm = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_from_cm = tp / (tp + fn) if (tp + fn) > 0 else 0
    frame_f1 = (
        2 * precision_from_cm * recall_from_cm / (precision_from_cm + recall_from_cm)
        if (precision_from_cm + recall_from_cm) > 0
        else 0
    )

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
    if fixed_thresholds and "clip" in fixed_thresholds:
        clip_optimal_threshold = fixed_thresholds["clip"]
        logger.info(f"Using fixed clip threshold: {clip_optimal_threshold}")
    else:
        clip_f1_scores = (
            2 * clip_precision * clip_recall / (clip_precision + clip_recall + 1e-10)
        )
        clip_optimal_idx = np.argmax(clip_f1_scores)
        clip_optimal_threshold = (
            clip_pr_thresholds[clip_optimal_idx]
            if clip_optimal_idx < len(clip_pr_thresholds)
            else 0.5
        )
        logger.info(f"Found optimal clip threshold: {clip_optimal_threshold}")

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

    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if model exists
    if not pathlib.Path(args.model_path).exists():
        logger.error(f"Model checkpoint missing: {args.model_path}")
        sys.exit("❌ Model checkpoint missing.")

    # For two-stage evaluation, prepare validation and test sets separately
    if args.two_stage_eval:
        logger.info("Using two-stage evaluation with separate validation set")
        val_manifest, test_manifest = prepare_validation_test_split(args)
        args.val_manifest = str(val_manifest)
        args.test_manifest = str(test_manifest)

        # Check if manifests exist
        for manifest, name in [
            (args.val_manifest, "Validation"),
            (args.test_manifest, "Test"),
        ]:
            if not pathlib.Path(manifest).exists():
                logger.error(f"{name} manifest missing: {manifest}")
                sys.exit(f"❌ {name} manifest missing. Use --prepare_data first.")

        # Evaluate on validation set to find optimal thresholds
        logger.info("Evaluating model on validation set")
        val_frame_preds, val_frame_labels, val_clip_preds, val_clip_labels = (
            evaluate_model(args.model_path, args.val_manifest, args)
        )

        # Find optimal thresholds on validation set
        logger.info("Finding optimal thresholds on validation set")
        val_metrics = plot_metrics(
            val_frame_preds,
            val_frame_labels,
            val_clip_preds,
            val_clip_labels,
            output_dir / "validation",
        )

        # Get optimal thresholds from validation
        fixed_thresholds = {
            "frame": val_metrics["frame_level"]["optimal_threshold"],
            "clip": val_metrics["clip_level"]["optimal_threshold"],
        }

        # Now evaluate on test set with fixed thresholds
        logger.info("Evaluating model on test set with validation-derived thresholds")
        test_frame_preds, test_frame_labels, test_clip_preds, test_clip_labels = (
            evaluate_model(args.model_path, args.test_manifest, args)
        )

        metrics = plot_metrics(
            test_frame_preds,
            test_frame_labels,
            test_clip_preds,
            test_clip_labels,
            output_dir / "test",
            fixed_thresholds=fixed_thresholds,
        )

    else:
        # Original single-stage evaluation
        if args.prepare_data or not args.test_manifest:
            logger.info("Preparing test data")
            test_manifest = prepare_test_data(args)
            args.test_manifest = str(test_manifest)

        # Check if test manifest exists
        if not pathlib.Path(args.test_manifest).exists():
            logger.error(f"Test manifest missing: {args.test_manifest}")
            sys.exit("❌ Test manifest missing. Use --prepare_data first.")

        # Evaluate model
        logger.info("Starting model evaluation")
        frame_preds, frame_labels, clip_preds, clip_labels = evaluate_model(
            args.model_path, args.test_manifest, args
        )

        # Generate visualizations
        logger.info("Generating metrics visualizations")
        metrics = plot_metrics(
            frame_preds, frame_labels, clip_preds, clip_labels, output_dir
        )

        # Store for additional analysis
        test_frame_preds = frame_preds
        test_frame_labels = frame_labels

    # Perform boundary analysis if requested
    if args.boundary_analysis:
        logger.info("Performing speech boundary analysis")
        # Use threshold from metrics for consistency
        threshold = metrics["frame_level"]["optimal_threshold"]
        boundary_metrics = analyze_speech_boundaries(
            test_frame_preds, test_frame_labels, threshold=threshold
        )

        # Save boundary metrics
        with open(output_dir / "boundary_metrics.json", "w") as f:
            json.dump(boundary_metrics, f, indent=4)

        # Plot transition error patterns
        plot_transition_errors(
            test_frame_preds,
            test_frame_labels,
            output_dir,
            threshold=threshold,
            window=args.transition_window,
        )

        # Print boundary metrics
        print(f"\n🔍 Boundary Detection Analysis:")
        print(f"  - Segment F1 Score: {boundary_metrics.get('segment_f1', 'N/A'):.4f}")
        print(
            f"  - Segments: {boundary_metrics.get('matched_true_segments', 0)}/{boundary_metrics.get('total_true_segments', 0)} matched"
        )

        if "mean_abs_onset_error" in boundary_metrics:
            print(
                f"  - Mean onset error: {boundary_metrics['mean_abs_onset_error']:.2f} frames"
            )
        if "mean_abs_offset_error" in boundary_metrics:
            print(
                f"  - Mean offset error: {boundary_metrics['mean_abs_offset_error']:.2f} frames"
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
