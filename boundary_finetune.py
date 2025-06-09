#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  boundary_finetune.py - Enhanced fine-tuning for VAD model
# ───────────────────────────────────────────────────────────────────────

import os
import sys
from train import main as train_main
from argparse import Namespace


def run_boundary_finetuning():
    """Fine-tune VAD model with enhanced boundary detection."""
    
    # Setup args for fine-tuning
    args = Namespace(
        # Paths
        vad_root="datasets",
        train_manifest="datasets/manifest_train.csv",
        val_manifest="datasets/manifest_val.csv",
        test_manifest="datasets/manifest_test.csv",
        mel_cache_dir="mel_cache",
        ckpt_path="D:\\belajar\\audio\\vad\\lightning_logs\\finetuning\\lightning_logs\\version_1\\checkpoints\\18-0.0164-0.9327.ckpt",
        log_dir="lightning_logs/boundary_enhanced_finetuning",
        export_path="D:\\belajar\\audio\\vad\\models\\vad_boundary_enhanced.pt",
        
        # Model parameters
        n_mels=80,
        n_fft=512,
        hop=160,
        sample_rate=16000,
        dim=192,
        n_layers=4,
        n_heads=4,
        max_frames=1000,
        
        # Enhanced training params
        lr=0.0001,  # Higher learning rate (2x original)
        max_epochs=20,  # Train longer
        batch_size=16,
        gradient_clip_val=1.0,
        warmup_epochs=0,
        pos_weight=1.0,
        seed=42,
        gpus=1,
        accumulate_grad_batches=1,  # Reduced from 2 to have more update steps
        
        # Enhanced augmentation
        time_mask_max=50,
        freq_mask_max=20,  # Increased from 10
        
        # New boundary-specific parameters
        boundary_weight=5.0,  # Increased from 3.0
        boundary_window=3,    # Consider 3 frames on each side of boundary
        boundary_focus_prob=0.7,  # Probability of boundary-focused augmentation
        
        # Options
        use_mel_cache=True,
        num_workers=0 if sys.platform == "win32" else 4,
        test_after_training=True,
        export_model=True,
        export_quantized=True,
        auto_batch_size=False,
        boundary_focused_loss=True,
        export_onnx=False,
        use_enhanced_dataset=True,  # Use the new BoundaryEnhancedDataset
    )
    
    # Run the training with boundary-focused settings
    print("Starting enhanced boundary-focused fine-tuning...")
    train_main(args)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    run_boundary_finetuning()