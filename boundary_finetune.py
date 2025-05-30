#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  boundary_finetune.py - Fine-tune VAD model with boundary-focused loss
# ───────────────────────────────────────────────────────────────────────

import os
import sys
from train import main as train_main
from argparse import Namespace

def run_boundary_finetuning():
    """Fine-tune VAD model with boundary-focused loss."""
    
    # Setup args for fine-tuning
    args = Namespace(
        # Paths
        vad_root="datasets",
        train_manifest="datasets/manifest_train.csv",
        val_manifest="datasets/manifest_val.csv",
        test_manifest="datasets/manifest_test.csv",
        mel_cache_dir="mel_cache",
        ckpt_path="D:\\belajar\\audio\\vad\\lightning_logs\\lightning_logs\\version_8\\checkpoints\\14-0.0152-0.9280.ckpt",
        log_dir="lightning_logs/finetuning",
        export_path="D:\\belajar\\audio\\vad\\models\\model_boundary_tuned.pt",
        
        # Model parameters (use same as original)
        n_mels=80,
        n_fft=512,
        hop=160,
        sample_rate=16000,
        dim=192,
        n_layers=4,
        n_heads=4,
        max_frames=1000,
        
        # Training params
        lr=0.00005,  # Use lower learning rate for fine-tuning
        max_epochs=20,
        batch_size=16,
        gradient_clip_val=1.0,
        warmup_epochs=0,  # No warmup needed for fine-tuning
        pos_weight=1.0,  # BoundaryFocalLoss handles this
        seed=42,
        gpus=1,
        accumulate_grad_batches=2,
        
        # Augmentation
        time_mask_max=50,
        freq_mask_max=10,
        
        # Options
        use_mel_cache=True,
        num_workers=0 if sys.platform == "win32" else 4,
        test_after_training=True,
        export_model=True,
        export_quantized=True,
        auto_batch_size=False,
        boundary_focused_loss=True,  # Enable boundary-focused loss
        export_onnx=False,
    )
    
    # Run the training with boundary-focused settings
    print("Starting boundary-focused fine-tuning...")
    train_main(args)
    print("Fine-tuning complete!")

if __name__ == "__main__":
    run_boundary_finetuning()