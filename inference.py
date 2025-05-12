#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import torch
import torchaudio
from glob import glob
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

from config import *

# Ensure torchaudio transforms are initialized (MelSpectrogram, etc.)
mel_transform = None
db_transform = None

def prepare_transforms(sample_rate, n_fft, hop_length, win_length, n_mels):
    """Initialize mel and amplitude->dB transforms (global, to avoid re-init per file)."""
    global mel_transform, db_transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, 
        win_length=win_length, n_mels=n_mels, power=1.0
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

def load_model(checkpoint_path, device):
    """Load the VAD model from checkpoint (Lightning) or TorchScript."""
    model = None
    if checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
        # Assume a TorchScript saved model
        model = torch.jit.load(checkpoint_path, map_location=device)
        model.eval()
    else:
        # Assume a PyTorch Lightning checkpoint
        from models import VADLightning  # requires models.py in same directory
        try:
            model = VADLightning.load_from_checkpoint(checkpoint_path, map_location=device)
        except Exception as e:
            # Fallback: load state dict manually
            checkpoint = torch.load(checkpoint_path, map_location=device)
            hparams = checkpoint.get("hyper_parameters", {})
            # Create a namespace for hyperparams
            hp = argparse.Namespace(**hparams)
            model = VADLightning(hp)
            model.load_state_dict(checkpoint["state_dict"])
        model.eval()
    model.to(device)
    return model

def process_audio(file_path, device, sample_rate):
    """Load an audio file and convert to mel spectrogram (on CPU), then to tensor on device."""
    waveform, sr = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        # if stereo, convert to mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    # Clamp to avoid extreme values
    waveform = waveform.clamp(-1.0, 1.0)
    # Compute mel spectrogram
    mel_spec = db_transform(mel_transform(waveform))
    mel_spec = mel_spec.squeeze(0).T  # shape (T, n_mels)
    # Move to device
    mel_spec = mel_spec.to(device)
    return mel_spec

def pad_batch(batch_tensors, max_frames=None):
    """Pad a list of [T, n_mels] tensors to the same length (max_frames or longest in batch)."""
    lengths = [t.shape[0] for t in batch_tensors]
    max_len = max_frames if max_frames is not None else max(lengths)
    max_len = min(max_len, max(lengths))
    n_mels = batch_tensors[0].shape[1]
    batch_size = len(batch_tensors)
    padded = torch.zeros(batch_size, max_len, n_mels, device=batch_tensors[0].device)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=batch_tensors[0].device)
    for i, t in enumerate(batch_tensors):
        seq_len = min(t.shape[0], max_len)
        padded[i, :seq_len, :] = t[:seq_len]
        mask[i, :seq_len] = True
    return padded, mask, lengths

def main(args=None):
    """Run VAD inference on audio files."""
    if args is None:
        # If called directly (not from CLI), parse arguments
        parser = argparse.ArgumentParser(description="VAD Inference Script")
        parser.add_argument("--input_dir", help="Directory of audio files to run VAD on")
        parser.add_argument("--input_files", nargs='+', help="List of audio file paths to run VAD on")
        parser.add_argument("--labels_dir", help="Directory containing frame label .npy files (optional)")
        parser.add_argument("--model_path", required=True, help="Path to trained VAD model (checkpoint .ckpt or TorchScript .pt)")
        parser.add_argument("--sample_rate", type=int, default=DEFAULT_SAMPLE_RATE, help="Audio sample rate expected by model")
        parser.add_argument("--n_fft", type=int, default=DEFAULT_N_FFT, help="n_fft for mel spectrogram (must match training)")
        parser.add_argument("--hop_length", type=int, default=DEFAULT_HOP_LENGTH, help="Hop length for mel spectrogram")
        parser.add_argument("--win_length", type=int, default=DEFAULT_WIN_LENGTH, help="Window length for mel spectrogram")
        parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, help="Number of mel bands")
        parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for processing audio")
        parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device: cuda or cpu")
        parser.add_argument("--output_dir", help="Directory to save prediction results (optional)")
        args = parser.parse_args()

    # Collect audio file paths
    file_list = []
    if args.input_files:
        file_list = args.input_files
    elif args.input_dir:
        file_list = sorted(glob(os.path.join(args.input_dir, "*.wav")))
    else:
        print("Please specify --input_dir or --input_files")
        return 1
    if len(file_list) == 0:
        print("No audio files found to process.")
        return 1

    # Prepare transforms and model
    prepare_transforms(args.sample_rate, args.n_fft, args.hop_length, args.win_length, args.n_mels)
    device = torch.device(args.device)
    model = load_model(args.model_path, device)

    # If model is a Lightning module, get underlying net
    # We assume if it's Lightning, it has .net attribute for actual nn.Module
    vad_net = model.net if hasattr(model, "net") else model

    all_frame_preds = []
    all_frame_labels = []
    clip_preds = []
    clip_labels = []

    # Save predictions if output directory specified
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Process in batches
    for start in range(0, len(file_list), args.batch_size):
        batch_files = file_list[start:start+args.batch_size]
        # Load and transform each file in batch (on CPU), then move to device
        mel_list = [process_audio(fp, device, args.sample_rate) for fp in batch_files]
        # Pad batch
        mel_batch, mask, lengths = pad_batch(mel_list)
        # Run model
        with torch.no_grad():
            logits = vad_net(mel_batch)  # shape (B, T) logits
            probs = torch.sigmoid(logits)  # probabilities
        probs = probs.cpu().numpy()
        mask = mask.cpu().numpy()

        # Process each file in batch
        for i, fp in enumerate(batch_files):
            valid_len = mask[i].sum()  # number of true frames
            frame_probs = probs[i, :valid_len]
            all_frame_preds.append(frame_probs)
            # Clip-level prediction = mean probability
            clip_pred = float(frame_probs.mean())
            clip_preds.append(clip_pred)
            # Load frame labels if available
            if args.labels_dir:
                base = os.path.splitext(os.path.basename(fp))[0]
                label_path = os.path.join(args.labels_dir, base + "_labels.npy")
                if os.path.exists(label_path):
                    frame_labels = np.load(label_path).astype(np.float32)
                    frame_labels = frame_labels[:valid_len]  # truncate to match
                    all_frame_labels.append(frame_labels)
                    # Determine clip label (1 if any speech frame)
                    clip_label = 1.0 if frame_labels.max() > 0.5 else 0.0
                    clip_labels.append(clip_label)
                else:
                    # If no label file, skip metrics for this file
                    all_frame_labels.append(None)
                    clip_labels.append(None)
            else:
                all_frame_labels.append(None)
                clip_labels.append(None)
            
            # Print or log per-file results
            print(f"{fp}: clip_speech_confidence={clip_pred:.3f}")
            
            # Save prediction to output directory if specified
            if args.output_dir:
                base = os.path.splitext(os.path.basename(fp))[0]
                pred_path = os.path.join(args.output_dir, f"{base}_pred.npy")
                np.save(pred_path, frame_probs)

    # Compute metrics if we have labels for all files
    have_labels = args.labels_dir and all(x is not None for x in all_frame_labels)
    if have_labels:
        # Concatenate all frame-level predictions and labels
        y_true = np.concatenate([lbls for lbls in all_frame_labels if lbls is not None])
        y_pred = np.concatenate([pred for pred in all_frame_preds])
        # Frame-level ROC AUC
        frame_roc_auc = roc_auc_score(y_true, y_pred)
        # Frame-level optimal F1
        prec, rec, thr = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
        best_frame_f1 = f1_scores[best_idx] if best_idx < len(f1_scores) else 0.0
        # At threshold 0.5 also compute F1 for reference
        frame_f1_0_5 = f1_score(y_true, (y_pred >= 0.5).astype(int))
        # Clip-level metrics
        clip_labels_arr = np.array([c for c in clip_labels if c is not None])
        clip_preds_arr = np.array([p for p, c in zip(clip_preds, clip_labels) if c is not None])
        clip_roc_auc = roc_auc_score(clip_labels_arr, clip_preds_arr) if clip_labels_arr.size > 0 else None
        clip_f1 = f1_score(clip_labels_arr, (clip_preds_arr >= 0.5).astype(int)) if clip_labels_arr.size > 0 else None

        print("\nOverall metrics:")
        print(f"Frame ROC AUC: {frame_roc_auc:.3f}")
        print(f"Frame F1 (optimal threshold = {best_thr:.2f}): {best_frame_f1:.3f}")
        print(f"Frame F1 (threshold=0.5): {frame_f1_0_5:.3f}")
        if clip_roc_auc is not None:
            print(f"Clip ROC AUC: {clip_roc_auc:.3f}")
        if clip_f1 is not None:
            print(f"Clip F1 (threshold=0.5): {clip_f1:.3f}")
    else:
        print("\nInference complete. No ground truth labels provided, so metrics were not computed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())