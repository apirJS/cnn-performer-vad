#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  train.py - MEL‑spectrogram VAD model training
# ───────────────────────────────────────────────────────────────────────
import argparse
import logging
import multiprocessing
import pathlib
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
import os

# Import from our modules
from config import *
from data import CSVMelDataset, VADDataModule, collate_pad, BoundaryEnhancedDataset
from prepare_data import seed_everything

from models import VADLightning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_best_precision():  # PERBAIKAN: Hapus parameter yang tidak digunakan
    """Return one of 'fp32', 'fp16-mixed' or 'bf16-mixed'."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available → using fp32")
        return "fp32"

    # PERBAIKAN: Hapus referensi ke device yang tidak digunakan
    props = torch.cuda.get_device_properties(0)  # Selalu gunakan device 0
    cc_major = props.major

    if torch.cuda.is_bf16_supported():
        logger.info("Using bf16-mixed")
        return "bf16-mixed"
    elif cc_major >= 7:  # Turing (7.5) or later
        logger.info("Using fp16-mixed")
        return "fp16-mixed"
    else:
        logger.info("GPU has no Tensor-core acceleration → fp32")
        return "fp32"


def build_parser():
    """Build and configure the argument parser."""
    logger.info("Building argument parser")
    p = argparse.ArgumentParser("Mel‑spectrogram Performer VAD")

    # workspace / data paths
    p.add_argument(
        "--vad_root",
        default=DEFAULT_ROOT_DIR,
        help="Root directory for all datasets and outputs",
    )
    p.add_argument("--train_manifest", help="Path to training manifest CSV")
    p.add_argument("--val_manifest", help="Path to validation manifest CSV")
    p.add_argument("--test_manifest", help="Path to test manifest for final evaluation")
    p.add_argument(
        "--test_after_training",
        action="store_true",
        help="Run evaluation on test set after training",
    )

    # mel parameters
    p.add_argument(
        "--n_mels",
        type=int,
        default=DEFAULT_N_MELS,
        help="Number of mel bands in spectrogram",
    )
    p.add_argument(
        "--n_fft", type=int, default=DEFAULT_N_FFT, help="FFT size for spectrogram"
    )
    p.add_argument(
        "--hop", type=int, default=DEFAULT_HOP_LENGTH, help="Hop length for spectrogram"
    )
    p.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Audio sample rate in Hz",
    )

    # data augmentation
    p.add_argument(
        "--time_mask_max",
        type=int,
        default=DEFAULT_TIME_MASK_MAX,
        help="Maximum time mask length for SpecAugment",
    )
    p.add_argument(
        "--freq_mask_max",
        type=int,
        default=DEFAULT_FREQ_MASK_MAX,
        help="Maximum frequency mask length for SpecAugment",
    )

    # caching
    p.add_argument(
        "--use_mel_cache",
        action="store_true",
        help="Cache Mel spectrograms to disk for faster training",
    )
    p.add_argument(
        "--mel_cache_dir",
        default=DEFAULT_CACHE_DIR,
        help="Directory to store cached Mel spectrograms",
    )

    # model architecture
    p.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIMENSION,
        help="Transformer embedding dimension",
    )
    p.add_argument(
        "--n_layers",
        type=int,
        default=DEFAULT_LAYERS,
        help="Number of transformer layers",
    )
    p.add_argument(
        "--n_heads", type=int, default=DEFAULT_HEADS, help="Number of attention heads"
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        help="Maximum number of frames to use in a sequence",
    )

    # export options
    p.add_argument(
        "--export_model",
        action="store_true",
        help="Export the trained model for inference",
    )
    p.add_argument(
        "--export_path",
        default=DEFAULT_MODEL_EXPORT_PATH,
        help="Path to save the exported model",
    )
    # In the build_parser() function, add these arguments:
    p.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export the model to ONNX format for cross-platform deployment",
    )
    p.add_argument(
        "--export_quantized",
        action="store_true",
        help="Export a quantized version of the model for edge deployment",
    )

    # training parameters
    cpu_count = max(1, multiprocessing.cpu_count() - 2)
    p.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size"
    )
    p.add_argument(
        "--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate"
    )
    p.add_argument(
        "--max_epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum training epochs",
    )
    p.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    p.add_argument(
        "--num_workers",
        type=int,
        default=cpu_count,
        help=f"Number of data loading workers (default: {cpu_count}, based on CPU count)",
    )
    p.add_argument(
        "--gradient_clip_val",
        type=float,
        default=DEFAULT_GRADIENT_CLIP_VAL,
        help="Gradient clipping to prevent exploding gradients",
    )
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=DEFAULT_WARMUP_EPOCHS,
        help="Number of epochs for learning rate warmup",
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=DEFAULT_POS_WEIGHT,
        help="Positive class weight for BCE loss to handle imbalance",
    )
    p.add_argument(
        "--auto_batch_size",
        action="store_true",
        help="Automatically determine optimal batch size based on GPU memory",
    )

    # checkpoint / miscellaneous
    p.add_argument("--ckpt_path", help="Path to checkpoint for resuming training")
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients before optimizer step",
    )
    p.add_argument(
        "--log_dir", default=DEFAULT_LOG_DIR, help="Directory for TensorBoard logs"
    )
    p.add_argument(
        "--boundary_focused_loss",
        action="store_true",
        help="Use boundary-focused loss for improved segment detection",
    )

    logger.info("Argument parser configured")
    return p


def create_datasets(args):
    """Create and return training and validation datasets."""
    logger.info("Creating datasets")
    cache_dir = pathlib.Path(args.mel_cache_dir) if args.use_mel_cache else None

    # Select dataset class based on args
    dataset_class = BoundaryEnhancedDataset if getattr(args, 'use_enhanced_dataset', False) else CSVMelDataset
    
    # Create training dataset
    logger.info(f"Creating training dataset from {args.train_manifest} using {dataset_class.__name__}")
    train_dataset = dataset_class(
        args.train_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
        **({"boundary_focus_prob": args.boundary_focus_prob} if dataset_class == BoundaryEnhancedDataset else {})
    )

    # Create validation dataset - always use standard dataset for validation
    logger.info(f"Creating validation dataset from {args.val_manifest}")
    val_dataset = CSVMelDataset(
        args.val_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    )

    return train_dataset, val_dataset


def estimate_optimal_batch_size(model, n_mels, max_frames):
    """Estimate optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        logger.info("No GPU available, using default batch size")
        return DEFAULT_BATCH_SIZE

    # Get available GPU memory
    gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
    gpu_memory_gb = gpu_memory_bytes / (1024**3)

    # Estimate memory requirements
    n_params = sum(p.numel() for p in model.parameters())
    param_memory_gb = n_params * 4 / (1024**3)  # 4 bytes per parameter

    # Memory for single sample with gradients (roughly 3x forward pass)
    sample_memory_gb = 3 * max_frames * n_mels * 4 / (1024**3)

    # Tambahkan safety margin 1.3x untuk Conv2D feature maps
    sample_memory_gb *= 1.3  # Safety margin untuk menghindari OOM

    # Reserve memory for PyTorch's own usage and other overhead (30%)
    available_memory_gb = gpu_memory_gb * 0.7 - param_memory_gb

    # Calculate maximum batch size based on available memory
    max_batch_size = max(1, int(available_memory_gb / sample_memory_gb))

    # Cap at reasonable limits
    optimal_batch_size = min(64, max(1, max_batch_size))

    logger.info(f"Estimated GPU memory: {gpu_memory_gb:.2f} GB")
    logger.info(f"Parameter memory: {param_memory_gb:.2f} GB")
    logger.info(f"Sample memory: {sample_memory_gb:.2f} GB per sample")
    logger.info(f"Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


def setup_environment(args):
    """Prepare the environment for training."""
    # Set random seed
    seed_everything(args.seed)

    # Verify manifests exist
    for m in [args.train_manifest, args.val_manifest]:
        if not m or not pathlib.Path(m).exists():
            logger.error(f"Manifest missing: {m}")
            sys.exit("❌ Manifest missing. Run prepare_data.py first.")

    # Check for Windows and disable multiprocessing if needed
    if sys.platform == "win32" and args.num_workers > 0:
        logger.warning(
            "Running on Windows - disabling multiprocessing (num_workers=0) to avoid pickling issues"
        )
        args.num_workers = 0
        print("⚠️ Windows detected: Disabled multiprocessing for compatibility")

    return args


def create_callbacks():
    """Create and return training callbacks."""
    logger.info("Setting up training callbacks")

    callbacks = {}

    # Checkpoint callback
    cb_ckpt = ModelCheckpoint(
        monitor="val_loss",  # Focus on boundary performance
        mode="min",
        save_top_k=3,
        filename="{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}",
        auto_insert_metric_name=False,
    )
    callbacks["checkpoint"] = cb_ckpt

    # Learning rate monitor
    cb_lr = LearningRateMonitor(logging_interval="epoch")
    callbacks["lr_monitor"] = cb_lr

    # Early stopping
    cb_early_stop = EarlyStopping(
        monitor="val_loss", patience=7, mode="min", verbose=True
    )
    callbacks["early_stopping"] = cb_early_stop

    return callbacks


def setup_trainer(args, callbacks):
    """Configure and return the PyTorch Lightning trainer."""
    logger.info("Setting up trainer")

    # Determine precision
    precision = get_best_precision()

    # Determine accelerator and strategy
    accelerator = "gpu" if args.gpus else "cpu"
    strategy = "ddp" if args.gpus > 1 else "auto"

    logger.info(f"Using {accelerator} with precision={precision}")
    if args.gpus > 1:
        logger.info(f"Using {strategy} strategy for {args.gpus} GPUs")

    # Create logger
    tb_logger = TensorBoardLogger(save_dir=args.log_dir)

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.gpus or 1,
        max_epochs=args.max_epochs,
        callbacks=list(callbacks.values()),
        log_every_n_steps=25,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        strategy=strategy,
        logger=tb_logger,
        precision=(
            32 if precision == "fp32" else ("bf16" if precision == "bf16-mixed" else 16)
        ),
    )

    return trainer


def export_model(
    model_path, output_path, export_onnx=False, quantize=False, data_module=None
):
    """Export a trained model for inference."""
    logger.info(f"Loading checkpoint from {model_path}")

    # Load the checkpoint and model
    checkpoint = torch.load(model_path, map_location="cpu")
    model = VADLightning.load_from_checkpoint(
        model_path, hp=checkpoint["hyper_parameters"], map_location="cpu"
    )
    model.eval()

    # Get the underlying network
    net = model.net

    # Save PyTorch model (original version)
    logger.info(f"Saving PyTorch model to {output_path}")
    torch.save(net.state_dict(), output_path)
    logger.info("PyTorch model export completed")

    # Log results
    logger.info(f"Model export completed")
    output_files = [output_path]

    print(f"✅ Model exported to: {', '.join(output_files)}")

    return net  # Return the original network for possible quantization


def export_quantized_model(model, hparams, output_path, data_module=None):
    """Export a quantized version of the model using the original architecture."""
    logger.info(f"Preparing model for quantization")

    # Extract the network part directly from the Lightning model passed in
    net = model.net
    net.eval()

    # Apply quantization directly to the loaded model - quantize both Linear AND Conv2d layers
    logger.info(f"Applying dynamic quantization to model")
    quantized_model = torch.quantization.quantize_dynamic(
        net,
        {nn.Linear, nn.Conv2d},  # Quantize both layer types for better compression
        dtype=torch.qint8,
    )

    # Save directly
    logger.info(f"Saving quantized model to {output_path}")
    torch.save(quantized_model.state_dict(), output_path)

    # Log size reduction
    original_path = output_path.replace("_quantized.pt", ".pt")
    if os.path.exists(original_path):
        original_size = os.path.getsize(original_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100
        logger.info(
            f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% reduction)"
        )
        print(
            f"✅ Model quantized and saved: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% smaller)"
        )

    return quantized_model


def custom_test(model, dataloader):
    """Custom test function that explicitly collects and processes test outputs."""
    device = next(model.parameters()).device
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]

            # Run test step directly
            output = model.test_step(batch, batch_idx)
            outputs.append(output)

    if not outputs:
        logger.warning("No outputs collected during testing")
        return {}

    # Calculate aggregate metrics
    avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean().item()
    avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean().item()

    # Concatenate predictions and labels
    all_probs = torch.cat([x["test_probs"] for x in outputs])
    all_labels = torch.cat([x["test_labels"] for x in outputs])

    # Calculate AUROC and F1 score
    try:
        from sklearn.metrics import roc_auc_score, f1_score

        probs_np = all_probs.cpu().numpy()
        labels_np = all_labels.cpu().numpy()

        auroc = roc_auc_score(labels_np, probs_np)
        f1 = f1_score(labels_np, (probs_np > 0.5).astype(int))

        return {
            "test_loss": avg_loss,
            "test_acc": avg_acc,
            "test_auroc": auroc,
            "test_f1": f1,
        }
    except Exception as e:
        logger.warning(f"Error calculating advanced metrics: {e}")
        return {"test_loss": avg_loss, "test_acc": avg_acc}


# Replace the existing perform_testing function


def perform_testing(trainer, model, args, checkpoint_callback):
    """Run evaluation on the test set."""
    logger.info(f"Running final evaluation on test set: {args.test_manifest}")

    # Check if we have a checkpoint to use
    if (
        not hasattr(checkpoint_callback, "best_model_path")
        or not checkpoint_callback.best_model_path
    ):
        logger.warning("No best checkpoint found, skipping test evaluation")
        return None

    # Load the best model with proper hyperparameter handling
    logger.info(f"Using best checkpoint: {checkpoint_callback.best_model_path}")

    # Load checkpoint and extract hyperparameters
    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
    hparams = checkpoint.get("hyper_parameters", {})

    # Convert to Namespace if it's a dictionary
    if isinstance(hparams, dict):
        from types import SimpleNamespace

        hp_namespace = SimpleNamespace(**hparams)
    else:
        hp_namespace = hparams

    # Load model with the hp parameter
    best_model = VADLightning.load_from_checkpoint(
        checkpoint_callback.best_model_path, hp=hp_namespace, map_location="cpu"
    )

    # Create test dataset
    cache_dir = pathlib.Path(args.mel_cache_dir) if args.use_mel_cache else None
    test_dataset = CSVMelDataset(
        args.test_manifest,
        args.n_mels,
        args.n_fft,
        args.hop,
        sample_rate=args.sample_rate,
        cache_dir=cache_dir,
        time_mask_max=args.time_mask_max,
        freq_mask_max=args.freq_mask_max,
    )

    # Create test dataloader
    from torch.utils.data import DataLoader

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_pad(batch, args.max_frames),
        pin_memory=True,
    )

    # Run test using our custom test function
    logger.info("Testing model performance using custom test function...")
    test_results = custom_test(best_model, test_dataloader)

    logger.info(f"Test results: {test_results}")
    print(f"✅ Test results: {test_results}")

    return test_results


def main(cli_args=None):
    """Main function for VAD model training."""
    logger.info("Starting VAD training script")

    # Parse arguments and setup environment
    if cli_args is None:
        args = build_parser().parse_args()
    else:
        args = cli_args

    logger.info(f"Parsed arguments: {args}")

    # PERBAIKAN: Panggil setup_environment untuk set seed & validasi path
    args = setup_environment(args)

    # Create datasets and data module
    train_dataset, val_dataset = create_datasets(args)
    
    # Log dataset types - this helps debugging
    logger.info(f"Train dataset type: {type(train_dataset).__name__}")
    logger.info(f"Val dataset type: {type(val_dataset).__name__}")

    # Create model
    logger.info("Creating VAD model")
    model = VADLightning(args)

    # Auto-determine batch size if requested
    if args.auto_batch_size and torch.cuda.is_available():
        logger.info("Estimating optimal batch size based on GPU memory")
        args.batch_size = estimate_optimal_batch_size(
            model, args.n_mels, args.max_frames
        )
        print(f"📊 Auto-determined batch size: {args.batch_size}")

    logger.info(f"Creating data module with batch_size={args.batch_size}")
    # Force num_workers=0 on Windows regardless of args setting
    num_workers = 0 if sys.platform == "win32" else args.num_workers
    if sys.platform == "win32" and args.num_workers > 0:
        logger.warning(
            "Windows detected: Forcing num_workers=0 to avoid pickling issues"
        )

    dm = VADDataModule(
        train_dataset,
        val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,  # Use the forced 0 value on Windows
        max_frames=args.max_frames,
    )

    # Log memory usage estimate
    n_params = sum(p.numel() for p in model.parameters())
    mem_est = n_params * 4 / (1024**2)  # Approx mem in MB for parameters
    batch_mem = args.batch_size * args.max_frames * args.n_mels * 4 / (1024**2)

    logger.info(f"Model parameters: {n_params:,}")
    logger.info(
        f"Estimated GPU memory: {mem_est:.1f}MB (params) + {batch_mem:.1f}MB (batch)"
    )
    print(f"Model parameters: {n_params:,}")
    print(f"Estimated GPU memory: {mem_est:.1f}MB (params) + {batch_mem:.1f}MB (batch)")

    # Setup callbacks and trainer
    callbacks = create_callbacks()
    trainer = setup_trainer(args, callbacks)

    # Train the model
    logger.info("Starting training")

    if args.ckpt_path:
        print(f"Loading checkpoint with partial initialization: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location="cpu")
        # Initialize model with non-strict loading
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        # Then fit without checkpoint path
        trainer.fit(model, dm)
    else:
        # Regular training from scratch
        trainer.fit(model, dm)

    logger.info("Training completed")

    # Run test evaluation if requested
    if args.test_after_training and args.test_manifest:
        perform_testing(trainer, model, args, callbacks["checkpoint"])

    if (
        (args.export_model or args.export_quantized)  # Removed export_onnx
        and hasattr(callbacks["checkpoint"], "best_model_path")
        and callbacks["checkpoint"].best_model_path
    ):
        logger.info(
            f"Exporting best model from {callbacks['checkpoint'].best_model_path}"
        )

        # Load checkpoint and extract hyperparameters
        checkpoint = torch.load(
            callbacks["checkpoint"].best_model_path, map_location="cpu"
        )
        hparams = checkpoint.get("hyper_parameters", {})

        # Convert to Namespace if it's a dictionary
        if isinstance(hparams, dict):
            from types import SimpleNamespace

            hp_namespace = SimpleNamespace(**hparams)
        else:
            hp_namespace = hparams

        # Load model with the hp parameter
        best_model = VADLightning.load_from_checkpoint(
            callbacks["checkpoint"].best_model_path, hp=hp_namespace, map_location="cpu"
        )

        # Export standard PyTorch model
        net = export_model(
            callbacks["checkpoint"].best_model_path,
            args.export_path,
            export_onnx=False,  # Never try ONNX export
            quantize=False,
        )

        # Quantization (if requested)
        if args.export_quantized:
            export_quantized_model(
                best_model,
                best_model.hparams,
                args.export_path.replace(".pt", "_quantized.pt"),
                data_module=None,
            )


if __name__ == "__main__":
    sys.exit(main())
