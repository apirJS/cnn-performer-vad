import numpy as np
import torch
import pathlib
from torch.utils.data import DataLoader
import argparse
import logging

from data import CSVMelDataset, collate_pad
from evaluation import load_model_by_type, batched_inference, process_batch
from config import DEFAULT_N_MELS, DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_MAX_FRAMES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser("Generate VAD Predictions")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--test_manifest", required=True, help="Path to test manifest CSV")
    parser.add_argument("--output_dir", default="evaluation_results", help="Directory to save results")
    parser.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS)
    parser.add_argument("--n_fft", type=int, default=DEFAULT_N_FFT)
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=0 if sys.platform == "win32" else 4)
    parser.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--model_type", default="pytorch", choices=["pytorch", "lightning", "quantized"])
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model_by_type(args.model_path, args, model_type=args.model_type)
    
    # Create dataset and dataloader
    logger.info(f"Creating dataset from {args.test_manifest}")
    test_dataset = CSVMelDataset(
        args.test_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=16000,  # Default sample rate
        cache_dir=None,  # No caching for simplicity
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_pad(batch, args.max_frames),
    )
    
    # Process batches
    logger.info("Running model inference...")
    batch_results = process_batch(model, test_loader, args.device)
    
    # Extract predictions and labels
    all_frame_preds = []
    all_frame_labels = []
    
    for batch in batch_results:
        for sample in batch:
            all_frame_preds.append(sample["frame_preds"])
            all_frame_labels.append(sample["frame_labels"])
    
    # Concatenate results
    frame_preds = np.concatenate(all_frame_preds)
    frame_labels = np.concatenate(all_frame_labels)
    
    # Save to files
    logger.info(f"Saving predictions to {output_dir}/frame_predictions.npy")
    np.save(output_dir / "frame_predictions.npy", frame_preds)
    np.save(output_dir / "frame_labels.npy", frame_labels)
    
    print(f"✅ Saved frame predictions and labels to {output_dir}")
    print(f"   - Predictions shape: {frame_preds.shape}")
    print(f"   - Labels shape: {frame_labels.shape}")

if __name__ == "__main__":
    import sys
    main()