#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  evaluation.py - Metrics visualization for MEL-spectrogram VAD model
# ───────────────────────────────────────────────────────────────────────
from tqdm import tqdm
from functools import wraps

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
from sklearn.metrics import f1_score, precision_score, recall_score

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

def merge_short_gaps(segments, max_gap_size=3):
    """Merge speech segments separated by brief gaps."""
    if not segments:
        return []
        
    merged = [segments[0]]
    
    for current in segments[1:]:
        previous_end = merged[-1][1]
        current_start = current[0]
        
        # If gap is small enough, merge
        if current_start - previous_end <= max_gap_size:
            merged[-1] = (merged[-1][0], current[1])
        else:
            merged.append(current)
            
    return merged

def filter_short_segments(segments, min_duration=5):
    """Filter out segments shorter than min_duration frames."""
    return [seg for seg in segments if (seg[1] - seg[0] + 1) >= min_duration]

def smooth_predictions(frame_preds, window_size=7, threshold=0.5):
    """Apply temporal smoothing to reduce over-segmentation."""
    from scipy.ndimage import median_filter, uniform_filter1d
    
    # First apply median filtering to remove brief spikes/drops
    smoothed = median_filter(frame_preds, size=window_size)
    
    # Then apply moving average to smooth transitions
    smoothed = uniform_filter1d(smoothed, size=window_size*2)
    
    # Convert to binary predictions
    binary_preds = (smoothed >= threshold).astype(int)
    
    return binary_preds, smoothed

def batched_inference(max_batches=None, progress=True):
    """Decorator for batched model inference with progress tracking."""
    def decorator(func):
        @wraps(func)
        def wrapper(model, dataloader, *args, **kwargs):
            all_results = []
            total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
            
            iterator = tqdm(dataloader, total=total) if progress else dataloader
            
            with torch.no_grad():
                for i, batch in enumerate(iterator):
                    if max_batches and i >= max_batches:
                        break
                        
                    result = func(model, batch, *args, **kwargs)
                    all_results.append(result)
            
            return all_results
        return wrapper
    return decorator


@batched_inference(progress=True)
def process_batch(model, batch, device):
    """Process a single batch with the model."""
    mel, labels, mask = batch
    mel = mel.to(device)
    logits = model(mel)
    preds = torch.sigmoid(logits)
    
    results = []
    for i in range(mel.shape[0]):
        valid_frames = mask[i].sum().item()
        sample_preds = preds[i, :valid_frames].cpu().numpy()
        sample_labels = labels[i, :valid_frames].cpu().numpy()
        
        # Calculate clip-level prediction and ground truth
        clip_pred = sample_preds.mean()
        clip_label = 1.0 if sample_labels.max() > 0.5 else 0.0
        
        results.append({
            "frame_preds": sample_preds,
            "frame_labels": sample_labels,
            "clip_pred": clip_pred,
            "clip_label": clip_label
        })
        
    return results

def load_model_by_type(model_path, args, model_type="pytorch"):
    """Load a model based on its type (pytorch, lightning, or quantized)."""
    logger.info(f"Loading {model_type} model from {model_path}")
    
    if model_type == "lightning":
        # For Lightning checkpoints, load using the proper Lightning API
        checkpoint = torch.load(model_path, map_location=args.device)
        
        # Create the VADLightning model with hyperparameters from checkpoint
        if "hyper_parameters" in checkpoint:
            hparams = checkpoint["hyper_parameters"]
            from types import SimpleNamespace
            hp_namespace = SimpleNamespace(**hparams)
            
            # Create and load the Lightning model
            pl_model = VADLightning(hp_namespace)
            pl_model.load_state_dict(checkpoint["state_dict"])
            
            # Extract just the network part
            inference_model = pl_model.net
        else:
            logger.error("Lightning checkpoint missing hyperparameters")
            raise ValueError("Invalid Lightning checkpoint format")
    
    elif model_type == "pytorch":
        # Load a standard PyTorch model state dict
        state_dict = torch.load(model_path, map_location=args.device)
        
        # Detect model dimension
        dim = None
        if "pos_embedding" in state_dict:
            dim = state_dict["pos_embedding"].shape[2]
            logger.info(f"Detected model dimension from pos_embedding: {dim}")
        elif "proj.weight" in state_dict:
            dim = state_dict["proj.weight"].shape[0]
            logger.info(f"Detected model dimension from proj.weight: {dim}")
        
        # Default fallback
        if dim is None:
            dim = getattr(args, 'dim', 192)
            logger.info(f"Using dimension from config: {dim}")
        
        # Create model with correct dimensions
        from models import MelPerformer
        inference_model = MelPerformer(
            n_mels=args.n_mels,
            dim=dim,
            layers=getattr(args, 'n_layers', 4),
            heads=getattr(args, 'n_heads', 4),
            max_seq_len=args.max_frames
        )
        
        logger.info(f"Loading PyTorch state dict into model with dim={dim}")
        inference_model.load_state_dict(state_dict)
            
    elif model_type == "quantized":
        # FIXED APPROACH: Load state dict into regular model first, then quantize
        state_dict = torch.load(model_path, map_location=args.device)
        
        # Detect model dimension
        dim = None
        if "pos_embedding" in state_dict:
            dim = state_dict["pos_embedding"].shape[2]
            logger.info(f"Detected model dimension from pos_embedding: {dim}")
        elif "proj.weight" in state_dict:
            dim = state_dict["proj.weight"].shape[0]
            logger.info(f"Detected model dimension from proj.weight: {dim}")
        
        # Default fallback
        if dim is None:
            dim = getattr(args, 'dim', 192)
            logger.info(f"Using dimension from config: {dim}")
        
        # Create a regular model first
        from models import MelPerformer
        base_model = MelPerformer(
            n_mels=args.n_mels,
            dim=dim,
            layers=getattr(args, 'n_layers', 4),
            heads=getattr(args, 'n_heads', 4),
            max_seq_len=args.max_frames
        )
        
        # Load weights into the regular model
        logger.info("Loading weights into base model before quantization")
        base_model.load_state_dict(state_dict)
        
        # Now quantize the model after loading weights
        logger.info("Applying dynamic quantization to loaded model")
        inference_model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
    
    # Set to evaluation mode
    inference_model.eval()
    inference_model.to(args.device)
    
    return inference_model


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
        split="val",
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


def evaluate_model(model_path, test_manifest, args, model_type="lightning"):
    """Evaluate a trained VAD model and return frame-level predictions and labels."""
    logger.info(f"Evaluating {model_type} model: {model_path}")

    # Load the model based on its type
    model = load_model_by_type(model_path, args, model_type)
    
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
    logger.info(f"Evaluating model on {len(test_dataset)} test samples with {len(test_loader)} batches")

    # Use optimized batch processing
    batch_results = process_batch(model, test_loader, args.device)
    
    # Extract and concatenate all results
    all_frame_preds = []
    all_frame_labels = []
    all_clip_preds = []
    all_clip_labels = []
    
    for batch in batch_results:
        for sample in batch:
            all_frame_preds.append(sample["frame_preds"])
            all_frame_labels.append(sample["frame_labels"])
            all_clip_preds.append(sample["clip_pred"])
            all_clip_labels.append(sample["clip_label"])
    
    # Concatenate results
    frame_preds = np.concatenate(all_frame_preds)
    frame_labels = np.concatenate(all_frame_labels)
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


def plot_smoothing_effect(frame_preds, frame_labels, output_dir):
    """Visualize the effect of smoothing on a sample portion of audio."""
    # Take a sample portion (1000 frames) where there's speech and silence
    has_speech = frame_labels.sum(axis=0) > 0
    sample_start = np.where(has_speech)[0][0] - 100
    sample_start = max(0, sample_start)
    sample_end = min(len(frame_preds), sample_start + 1000)
    
    sample_preds = frame_preds[sample_start:sample_end]
    sample_labels = frame_labels[sample_start:sample_end]
    
    # Apply smoothing
    _, smoothed = smooth_predictions(sample_preds, window_size=5)
    binary_preds = (sample_preds > 0.5).astype(int)
    binary_smooth = (smoothed > 0.5).astype(int)
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.subplot(3, 1, 1)
    plt.plot(sample_preds, label='Raw Predictions')
    plt.plot(smoothed, label='Smoothed Predictions')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.legend()
    plt.title('Raw vs Smoothed Predictions')
    
    plt.subplot(3, 1, 2)
    plt.step(range(len(binary_preds)), binary_preds, label='Binary Raw')
    plt.step(range(len(binary_smooth)), binary_smooth, label='Binary Smoothed')
    plt.step(range(len(sample_labels)), sample_labels, label='Ground Truth', linestyle=':')
    plt.legend()
    plt.title('Binary Decisions')
    
    plt.subplot(3, 1, 3)
    plt.step(range(len(binary_preds)), binary_preds != sample_labels, label='Raw Errors')
    plt.step(range(len(binary_smooth)), binary_smooth != sample_labels, label='Smoothed Errors')
    plt.legend()
    plt.title('Error Locations (1=error)')
    
    plt.tight_layout()
    plt.savefig(output_dir / "smoothing_effect.png", dpi=300)
    plt.close()

def smooth_predictions_with_hysteresis(frame_preds, window_size=7, high_threshold=0.6, low_threshold=0.4):
    """Apply hysteresis thresholding for better boundary detection."""
    from scipy.ndimage import median_filter
    
    # First smooth the predictions with median filtering
    smoothed = median_filter(frame_preds, size=window_size)
    
    # Apply hysteresis thresholding
    binary = np.zeros_like(smoothed, dtype=int)
    in_segment = False
    
    for i in range(len(smoothed)):
        if not in_segment and smoothed[i] >= high_threshold:
            # Start of segment (using higher threshold)
            in_segment = True
            binary[i] = 1
        elif in_segment and smoothed[i] >= low_threshold:
            # Continue segment (using lower threshold)
            binary[i] = 1
        elif in_segment and smoothed[i] < low_threshold:
            # End of segment
            in_segment = False
    
    return binary

def analyze_speech_boundaries(frame_preds, frame_labels, threshold=0.5, args=None, output_dir=None):
    """Analyze speech boundary detection performance."""
    logger.info("Analyzing speech boundary detection performance")
    boundary_metrics = {}
    
    # Safely handle output_dir
    if output_dir is None:
        if args and hasattr(args, 'output_dir'):
            output_dir = pathlib.Path(args.output_dir)
        else:
            output_dir = pathlib.Path("evaluation_results")
    else:
        output_dir = pathlib.Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default values when args is None
    sample_rate = getattr(args, 'sample_rate', 16000) if args else 16000
    hop = getattr(args, 'hop', 160) if args else 160
    smoothing_window_ms = getattr(args, 'smoothing_window_ms', 50) if args else 50
    min_segment_duration_ms = getattr(args, 'min_segment_duration_ms', 150) if args else 150  # Changed from 200ms to 150ms
    max_gap_duration_ms = getattr(args, 'max_gap_duration_ms', 150) if args else 150  # Changed from 100ms to 150ms
    
    # Convert time-based parameters to frames
    frames_per_second = sample_rate / hop
    smoothing_window = max(3, int(smoothing_window_ms * frames_per_second / 1000))
    min_segment_frames = max(1, int(min_segment_duration_ms * frames_per_second / 1000))
    max_gap_frames = max(1, int(max_gap_duration_ms * frames_per_second / 1000))
    
    logger.info(f"Using smoothing window: {smoothing_window} frames ({smoothing_window_ms}ms)")
    logger.info(f"Min segment duration: {min_segment_frames} frames ({min_segment_duration_ms}ms)")
    logger.info(f"Max gap: {max_gap_frames} frames ({max_gap_duration_ms}ms)")
    
    # Use hysteresis thresholding instead of simple thresholding
    binary_preds = smooth_predictions_with_hysteresis(
        frame_preds, 
        window_size=smoothing_window, 
        high_threshold=threshold+0.15,  # Increase from 0.1 to 0.15
        low_threshold=threshold-0.08    # Asymmetric threshold for better detection
    )
    
    # Find true speech segments
    true_segments = find_segments(frame_labels > 0.5)
    logger.info(f"Found {len(true_segments)} true speech segments")

    # Find predicted speech segments
    pred_segments = find_segments(binary_preds)
    logger.info(f"Found {len(pred_segments)} predicted speech segments")
    
    # Apply filtering and merging
    pred_segments = filter_short_segments(pred_segments, min_duration=min_segment_frames)

    avg_true_duration = np.mean([end-start+1 for start, end in true_segments])
    if avg_true_duration > 50:  # Long segments in dataset
        # More aggressive merging for datasets with longer segments
        pred_segments = merge_short_gaps(pred_segments, max_gap_size=max_gap_frames*1.5)
    else:
        # Standard merging for datasets with shorter segments
        pred_segments = merge_short_gaps(pred_segments, max_gap_size=max_gap_frames)

    logger.info(f"After merging: {len(pred_segments)} predicted segments")
    

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
        if best_iou > 0.4:  # Consider it a match if IoU > 0.5
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

    if onset_errors:
        boundary_metrics["mean_onset_error"] = float(np.mean(onset_errors))
        boundary_metrics["mean_abs_onset_error"] = float(np.mean(np.abs(onset_errors)))
        # Safely handle max value 
        max_error = np.max(np.abs(onset_errors))
        if np.isfinite(max_error):
            boundary_metrics["max_onset_error"] = float(max_error)
        else:
            boundary_metrics["max_onset_error"] = "Infinity"  # Use string for non-finite values

    # Similarly for offset errors
    if offset_errors:
        boundary_metrics["mean_offset_error"] = float(np.mean(offset_errors))
        boundary_metrics["mean_abs_offset_error"] = float(np.mean(np.abs(offset_errors)))
        max_error = np.max(np.abs(offset_errors))
        if np.isfinite(max_error):
            boundary_metrics["max_offset_error"] = float(max_error)
        else:
            boundary_metrics["max_offset_error"] = "Infinity"

    frames_per_second = args.sample_rate / args.hop

    # Generate duration distribution plots
    duration_stats = plot_segment_durations(
        true_segments, pred_segments, output_dir or pathlib.Path(args.output_dir), frames_per_second
    )

    # Add duration statistics to boundary metrics
    boundary_metrics["duration_stats"] = duration_stats

    # Calculate frame duration in milliseconds
    frame_ms = 1000 * args.hop / args.sample_rate

    # Add detection latency analysis
    latency_metrics = analyze_detection_latency(
        true_segments, pred_segments,output_dir or pathlib.Path(args.output_dir), frame_ms=frame_ms
    )
    boundary_metrics["latency"] = latency_metrics

    # Print latency results
    if "mean_latency_ms" in latency_metrics:
        logger.info(f"Mean speech onset detection latency: {latency_metrics['mean_latency_ms']:.1f} ms")
        logger.info(f"Detection rate: {latency_metrics['detection_rate']*100:.1f}% " +
                f"({latency_metrics['detected_segments']}/{latency_metrics['total_segments']} segments)")

    return boundary_metrics

def analyze_detection_latency(true_segments, pred_segments, output_dir, frame_ms=10):
    """Analyze speech onset detection latency."""
    logger.info("Analyzing speech onset detection latency")
    
    # Ensure output_dir is a Path
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    onset_delays = []  # Will contain latency for each detected speech segment (in ms)
    matched_segments = 0
    undetected_segments = 0
    
    for true_start, true_end in true_segments:
        # Find if this true segment was detected
        detected = False
        min_delay = float('inf')
        
        for pred_start, pred_end in pred_segments:
            # Check if there's reasonable overlap (at least 40% of true segment)
            overlap_start = max(true_start, pred_start)
            overlap_end = min(true_end, pred_end)
            
            if overlap_start <= overlap_end:
                overlap_duration = overlap_end - overlap_start + 1
                true_duration = true_end - true_start + 1
                
                # If reasonable overlap - reduced from 50% to 40%
                if overlap_duration >= 0.4 * true_duration:
                    detected = True
                    # Calculate onset delay - how long after true onset did we detect it?
                    delay = pred_start - true_start 
                    
                    # Keep only the earliest detection
                    if delay < min_delay:
                        min_delay = delay
            
        if detected:
            matched_segments += 1
            # Convert frames to milliseconds
            onset_delays.append(min_delay * frame_ms)
        else:
            undetected_segments += 1
    
    # Filter out extreme outliers (beyond 3 seconds early/late)
    filtered_delays = [d for d in onset_delays if abs(d) < 3000]
    
    # Calculate statistics
    results = {
        "total_segments": len(true_segments),
        "detected_segments": matched_segments,
        "undetected_segments": undetected_segments,
        "detection_rate": matched_segments / len(true_segments) if len(true_segments) > 0 else 0,
        "filtered_outliers": len(onset_delays) - len(filtered_delays)
    }
    
    if filtered_delays:
        results.update({
            "mean_latency_ms": float(np.mean(filtered_delays)),
            "median_latency_ms": float(np.median(filtered_delays)),
            "min_latency_ms": float(np.min(filtered_delays)),
            "max_latency_ms": float(np.max(filtered_delays))
        })
        
        # Create latency histogram visualization
        plt.figure(figsize=(12, 6))
        plt.hist(filtered_delays, bins=30)
        plt.axvline(x=np.mean(filtered_delays), color='r', linestyle='--', 
                  label=f'Mean: {np.mean(filtered_delays):.1f} ms')
        plt.axvline(x=np.median(filtered_delays), color='g', linestyle='--', 
                  label=f'Median: {np.median(filtered_delays):.1f} ms')
        plt.xlabel('Detection Latency (ms)')
        plt.ylabel('Count')
        plt.title('Speech Onset Detection Latency')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "onset_latency.png", dpi=300)
        plt.close()
    
    return results

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
    thresholds = np.append(thresholds, 1.0)
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
        optimal_threshold = thresholds[optimal_idx]

        logger.info(f"Found optimal frame threshold: {optimal_threshold}")

    binary_preds = (frame_preds >= optimal_threshold).astype(int)
    cm = confusion_matrix(frame_labels, binary_preds)
    frame_f1 = f1_score(frame_labels, binary_preds)

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
    thresholds_for_plot = np.linspace(0, 1, 20)  # Reduced from 100 to 20 points
    f1_values = []
    precision_values = []
    recall_values = []

    # Use sampling for large datasets to speed up calculation
    if len(frame_labels) > 100000:
        logger.info(f"Sampling 100,000 frames for threshold metrics (from {len(frame_labels)} total)")
        sample_indices = np.random.choice(len(frame_labels), 100000, replace=False)
        sample_preds = frame_preds[sample_indices]
        sample_labels = frame_labels[sample_indices]
    else:
        sample_preds = frame_preds
        sample_labels = frame_labels

    # Calculate metrics for each threshold
    for threshold in thresholds_for_plot:
        # Create new binary predictions for each threshold
        binary_preds = (sample_preds >= threshold).astype(int)
        
        # Calculate metrics with these predictions
        prec = precision_score(sample_labels, binary_preds, zero_division=0)
        rec = recall_score(sample_labels, binary_preds, zero_division=0)
        f1 = f1_score(sample_labels, binary_preds)
        
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
    clip_pr_thresholds = np.append(clip_pr_thresholds, 1.0)

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
        clip_optimal_threshold = clip_pr_thresholds[clip_optimal_idx]
        logger.info(f"Found optimal clip threshold: {clip_optimal_threshold}")

    clip_binary_preds = (clip_preds >= clip_optimal_threshold).astype(int)
    clip_cm = confusion_matrix(clip_labels, clip_binary_preds)
    clip_f1 = f1_score(clip_labels, clip_binary_preds)

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

    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(
        frame_preds, frame_labels, optimal_threshold
    )

    # Add them to the metrics dictionary
    metrics["frame_level"].update({
        "false_alarm_rate": float(advanced_metrics["false_alarm_rate"]),
        "miss_detection_rate": float(advanced_metrics["miss_detection_rate"]),
        "precision_at_recall90": float(advanced_metrics["precision_at_recall90"])
    })

    # Create a visualization of false alarm vs miss rates
    plt.figure(figsize=(10, 8))
    thresholds_for_plot = np.linspace(0, 1, 100)
    far_values = []
    mdr_values = []

    for t in thresholds_for_plot:
        advanced = calculate_advanced_metrics(frame_preds, frame_labels, t)
        far_values.append(advanced["false_alarm_rate"])
        mdr_values.append(advanced["miss_detection_rate"])

    plt.plot(far_values, mdr_values, 'b-')
    plt.plot(advanced_metrics["false_alarm_rate"], advanced_metrics["miss_detection_rate"], 
            'ro', markersize=10, label=f'Operating point (t={optimal_threshold:.3f})')

    plt.xlabel('False Alarm Rate')
    plt.ylabel('Miss Detection Rate')
    plt.title('Miss vs False Alarm Rate Trade-off')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "miss_false_alarm_tradeoff.png", dpi=300)
    plt.close()

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"All visualizations saved to {output_dir}")
    logger.info(f"Metrics: {metrics}")

    print(f"\n✅ Evaluation complete!")
    print(f"📊 Frame-level metrics:")
    print(f"  - ROC AUC: {metrics['frame_level']['roc_auc']:.4f}")
    print(f"  - PR AUC: {metrics['frame_level']['pr_auc']:.4f}")
    print(f"  - F1 score: {metrics['frame_level']['f1_score']:.4f}")
    print(f"  - False alarm rate: {metrics['frame_level'].get('false_alarm_rate', 'N/A'):.4f}")
    print(f"  - Miss detection rate: {metrics['frame_level'].get('miss_detection_rate', 'N/A'):.4f}")
    print(f"  - Precision@90% Recall: {metrics['frame_level'].get('precision_at_recall90', 'N/A'):.4f}")
    print(f"  - Optimal threshold: {metrics['frame_level']['optimal_threshold']:.4f}")

    return metrics

def plot_segment_durations(true_segments, pred_segments, output_dir, frames_per_second):
    """Plot histograms of segment durations."""
    logger.info("Creating segment duration distribution plot")
    
    # Ensure output_dir is a Path 
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate durations in seconds
    true_durations = [(end-start+1)/frames_per_second for start, end in true_segments]
    pred_durations = [(end-start+1)/frames_per_second for start, end in pred_segments]
    
    plt.figure(figsize=(12, 8))
    
    # Primary plot - histogram
    plt.subplot(2, 1, 1)
    plt.hist(true_durations, bins=20, alpha=0.5, label="True segments")
    plt.hist(pred_durations, bins=20, alpha=0.5, label="Predicted segments")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")
    plt.title("Speech Segment Duration Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Secondary plot - boxplot for comparison
    plt.subplot(2, 1, 2)
    box_data = [true_durations, pred_durations]
    plt.boxplot(box_data, labels=["True segments", "Predicted segments"])
    plt.ylabel("Duration (seconds)")
    plt.title("Segment Duration Comparison")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "segment_durations.png", dpi=300)
    plt.close()
    
    # Calculate and return statistics
    stats = {
        "true_segments": {
            "count": len(true_durations),
            "mean_duration": np.mean(true_durations) if true_durations else 0,
            "median_duration": np.median(true_durations) if true_durations else 0,
            "min_duration": min(true_durations) if true_durations else 0,
            "max_duration": max(true_durations) if true_durations else 0,
        },
        "pred_segments": {
            "count": len(pred_durations),
            "mean_duration": np.mean(pred_durations) if pred_durations else 0,
            "median_duration": np.median(pred_durations) if pred_durations else 0,
            "min_duration": min(pred_durations) if pred_durations else 0,
            "max_duration": max(pred_durations) if pred_durations else 0,
        }
    }
    
    return stats

# Add after other metrics functions (around line 730)

def calculate_advanced_metrics(frame_preds, frame_labels, threshold):
    """Calculate advanced metrics for VAD including miss/false alarm rates."""
    binary_preds = (frame_preds >= threshold).astype(int)
    
    # True/False Positive/Negative counts
    tp = np.sum((binary_preds == 1) & (frame_labels == 1))
    tn = np.sum((binary_preds == 0) & (frame_labels == 0))
    fp = np.sum((binary_preds == 1) & (frame_labels == 0))
    fn = np.sum((binary_preds == 0) & (frame_labels == 1))
    
    # Basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Additional metrics
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    miss_detection_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Precision at specific recall levels
    precisions, recalls, thresholds = precision_recall_curve(frame_labels, frame_preds)
    
    # Interpolate precision at 90% recall
    p_at_r90 = 0
    for i in range(len(recalls) - 1):
        if recalls[i] >= 0.9 >= recalls[i + 1]:
            # Linear interpolation
            p_at_r90 = precisions[i] + (precisions[i + 1] - precisions[i]) * \
                      (0.9 - recalls[i]) / (recalls[i + 1] - recalls[i])
            break
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "miss_detection_rate": miss_detection_rate, 
        "precision_at_recall90": p_at_r90
    }

def merge_nearby_segments(segments, max_gap=3):
    """Merge segments separated by small gaps."""
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    for current in sorted_segments[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged

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
            test_frame_preds, test_frame_labels, threshold=threshold, args=args
        )

        # Save boundary metrics
        try:
            with open(output_dir / "boundary_metrics.json", "w") as f:
                json.dump(boundary_metrics, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving boundary metrics: {e}")
            # Create a simpler version without problematic values
            simple_metrics = {k: v for k, v in boundary_metrics.items() 
                            if isinstance(v, (int, float, str)) and 
                                (not isinstance(v, float) or (not np.isnan(v) and not np.isinf(v)))}
            with open(output_dir / "boundary_metrics_safe.json", "w") as f:
                json.dump(simple_metrics, f, indent=4)

        # Plot transition error patterns
        plot_transition_errors(
            test_frame_preds,
            test_frame_labels,
            output_dir,
            threshold=threshold,
            window=args.transition_window,
        )

        if "latency" in boundary_metrics:
            lat = boundary_metrics["latency"]
            if "mean_latency_ms" in lat:
                print(f"  - Mean onset latency: {lat['mean_latency_ms']:.1f} ms")
                print(f"  - Detection rate: {lat['detection_rate']*100:.1f}% " +
                    f"({lat['detected_segments']}/{lat['total_segments']} segments)")
                
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


    # Store results from different model evaluations
    results = {}
    
    # Check if we have metrics from the standard evaluation
    if args.model_path and 'metrics' in locals():
        # Only store if metrics was actually defined (meaning evaluation succeeded)
        results["lightning"] = metrics
    
    # If pytorch_model is specified, evaluate it
    if args.pytorch_model:
        if not pathlib.Path(args.pytorch_model).exists():
            logger.error(f"PyTorch model missing: {args.pytorch_model}")
        else:
            logger.info(f"Evaluating PyTorch model: {args.pytorch_model}")
            frame_preds, frame_labels, clip_preds, clip_labels = evaluate_model(
                args.pytorch_model, args.test_manifest, args, model_type="pytorch"
            )
            
            pt_metrics = plot_metrics(
                frame_preds, frame_labels, clip_preds, clip_labels, 
                output_dir / "pytorch_model"
            )
            results["pytorch"] = pt_metrics
    
    # If quantized_model is specified, evaluate it
    if args.quantized_model:
        if not pathlib.Path(args.quantized_model).exists():
            logger.error(f"Quantized model missing: {args.quantized_model}")
        else:
            logger.info(f"Evaluating quantized model: {args.quantized_model}")
            frame_preds, frame_labels, clip_preds, clip_labels = evaluate_model(
                args.quantized_model, args.test_manifest, args, model_type="quantized"
            )
            
            quant_metrics = plot_metrics(
                frame_preds, frame_labels, clip_preds, clip_labels, 
                output_dir / "quantized_model"
            )
            results["quantized"] = quant_metrics
    
    # If multiple models were evaluated and comparison was requested, create a comparison
    if args.compare_models and len(results) > 1:
        logger.info("Comparing model performances")
        
        # Create comparison table
        comparison = {
            "frame_f1": {},
            "frame_auc": {},
            "clip_f1": {},
            "clip_auc": {},
        }
        
        for model_type, metrics in results.items():
            comparison["frame_f1"][model_type] = metrics["frame_level"]["f1_score"]
            comparison["frame_auc"][model_type] = metrics["frame_level"]["roc_auc"]
            comparison["clip_f1"][model_type] = metrics["clip_level"]["f1_score"]
            comparison["clip_auc"][model_type] = metrics["clip_level"]["roc_auc"]
        
        # Save comparison to JSON
        with open(output_dir / "model_comparison.json", "w") as f:
            json.dump(comparison, f, indent=4)
        
        # Print comparison
        print("\n📊 Model Performance Comparison:")
        for metric, values in comparison.items():
            print(f"\n{metric.upper()}:")
            for model_type, value in values.items():
                print(f"  - {model_type}: {value:.4f}")
        
        # Create comparison bar chart
        plt.figure(figsize=(12, 8))
        
        # Plot F1 scores
        x = np.arange(len(comparison))
        width = 0.35
        
        models = list(next(iter(comparison.values())).keys())
        metrics = list(comparison.keys())
        
        for i, model_type in enumerate(models):
            values = [comparison[metric][model_type] for metric in metrics]
            plt.bar(x + i*width, values, width, label=model_type)
        
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width/2, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison.png", dpi=300)
        plt.close()

    # Print summary
    print(f"\n✅ Evaluation complete!")

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
