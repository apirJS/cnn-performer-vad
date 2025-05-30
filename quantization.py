#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  quantization.py - Model quantization utilities for VAD
# ───────────────────────────────────────────────────────────────────────
import os
import time
import logging
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader, Subset
import numpy as np

from models import MelPerformer, VADLightning
from data import CSVMelDataset, collate_pad
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class QuantizationReady(nn.Module):
    """Wrapper to make the model quantization-ready by adding quant/dequant stubs."""
    def __init__(self, model):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
        
    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


def prepare_for_static_quantization(model):
    """Prepare model for static quantization by replacing key components."""
    # Replace ReLU with ReLU that supports quantization
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, nn.ReLU(inplace=False))
        # Recursively apply to all submodules
        elif len(list(module.children())) > 0:
            prepare_for_static_quantization(module)
    
    return model


def create_calibration_dataloader(dataset_path, n_mels, n_fft, hop, 
                                 sample_rate, batch_size=8, num_samples=100):
    """Create a small dataloader for calibration."""
    dataset = CSVMelDataset(
        dataset_path,
        n_mels=n_mels,
        n_fft=n_fft,
        hop=hop,  # Changed from hop_length=hop to hop=hop
        sample_rate=sample_rate,
    )
    
    # Take a subset for calibration
    if len(dataset) > num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        calibration_dataset = Subset(dataset, indices)
    else:
        calibration_dataset = dataset
    
    dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_pad(batch, DEFAULT_MAX_FRAMES)
    )
    
    return dataloader


def calibrate_model(model, dataloader, device="cpu"):
    """Calibrate the model with representative data."""
    logger.info("Calibrating model with representative data...")
    model.eval()
    with torch.no_grad():
        for i, (mel, _, _) in enumerate(dataloader):
            if i % 10 == 0:
                logger.info(f"Calibration batch {i}/{len(dataloader)}")
            model(mel.to(device))
    logger.info("Calibration complete")


def dynamic_quantization(model_path, output_path=None, linear_only=True):
    """Apply dynamic quantization to a trained model."""
    logger.info(f"Loading model from {model_path}")
    
    # Determine if this is a Lightning or PyTorch model
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint and "hyper_parameters" in checkpoint:
        # Lightning model
        from types import SimpleNamespace
        hp = SimpleNamespace(**checkpoint["hyper_parameters"])
        model = VADLightning(hp)
        model.load_state_dict(checkpoint["state_dict"])
        model = model.net  # Extract just the network part
        logger.info("Loaded Lightning model")
    else:
        # PyTorch model
        from models import MelPerformer
        # Try to detect dimension from file
        if "pos_embedding" in checkpoint:
            dim = checkpoint["pos_embedding"].shape[2]
        else:
            dim = DEFAULT_DIMENSION
            
        model = MelPerformer(
            n_mels=DEFAULT_N_MELS,
            dim=dim,
            layers=DEFAULT_LAYERS,
            heads=DEFAULT_HEADS,
            max_seq_len=DEFAULT_MAX_FRAMES
        )
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded PyTorch model with dimension {dim}")
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Set of layers to quantize
    quantize_layers = {nn.Linear}
    if not linear_only:
        quantize_layers.update({nn.Conv2d})
    
    # Apply dynamic quantization
    logger.info(f"Applying dynamic quantization to {len(quantize_layers)} layer types")
    start_time = time.time()
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        quantize_layers,
        dtype=torch.qint8
    )
    quantization_time = time.time() - start_time
    logger.info(f"Quantization completed in {quantization_time:.2f} seconds")
    
    # Save quantized model
    if output_path is None:
        output_path = model_path.replace(".pt", "_quantized.pt")
        if output_path == model_path:  # No extension change happened
            output_path = model_path + "_quantized"
    
    # Save the model
    torch.save(quantized_model.state_dict(), output_path)
    logger.info(f"Saved quantized model to {output_path}")
    
    # Report size reduction
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    logger.info(f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% reduction)")
    print(f"✅ Model quantized and saved: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% smaller)")
    
    return quantized_model


def static_quantization(model_path, calibration_dataset, output_path=None):
    """Apply static quantization with calibration."""
    logger.info(f"Loading model from {model_path}")
    
    # Similar to dynamic_quantization, load the model first
    checkpoint = torch.load(model_path, map_location="cpu")
    if "state_dict" in checkpoint and "hyper_parameters" in checkpoint:
        # Lightning model
        from types import SimpleNamespace
        hp = SimpleNamespace(**checkpoint["hyper_parameters"])
        pl_model = VADLightning(hp)
        pl_model.load_state_dict(checkpoint["state_dict"])
        model = pl_model.net  # Extract just the network part
    else:
        # PyTorch model
        from models import MelPerformer
        # Try to detect dimension
        if "pos_embedding" in checkpoint:
            dim = checkpoint["pos_embedding"].shape[2]
        else:
            dim = DEFAULT_DIMENSION
            
        model = MelPerformer(
            n_mels=DEFAULT_N_MELS,
            dim=dim,
            layers=DEFAULT_LAYERS,
            heads=DEFAULT_HEADS,
            max_seq_len=DEFAULT_MAX_FRAMES
        )
        model.load_state_dict(checkpoint)
    
    # 1. Prepare the model for static quantization
    model = prepare_for_static_quantization(model)
    
    # 2. Wrap in QuantizationReady wrapper
    model = QuantizationReady(model)
    
    # 3. Set model to eval mode
    model.eval()
    
    # 4. Set the quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # For x86
    # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # For ARM
    
    # 5. Prepare for quantization (adds observers)
    torch.quantization.prepare(model, inplace=True)
    
    # 6. Calibrate with the representative dataset
    calibration_loader = create_calibration_dataloader(
        calibration_dataset,
        DEFAULT_N_MELS,
        DEFAULT_N_FFT,
        DEFAULT_HOP_LENGTH,
        DEFAULT_SAMPLE_RATE,
        batch_size=8,
        num_samples=100
    )
    calibrate_model(model, calibration_loader)
    
    # 7. Convert to quantized model
    torch.quantization.convert(model, inplace=True)
    
    # Save quantized model
    if output_path is None:
        output_path = model_path.replace(".pt", "_static_quantized.pt")
        if output_path == model_path:  # No extension change happened
            output_path = model_path + "_static_quantized"
    
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved static quantized model to {output_path}")
    
    # Report size
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quantized_size / original_size) * 100
    
    logger.info(f"Model size: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% reduction)")
    print(f"✅ Model quantized (static) and saved: {original_size:.2f}MB → {quantized_size:.2f}MB ({reduction:.1f}% smaller)")
    
    return model


def benchmark_models(original_model, quantized_model, input_shape=(1, 500, 80), num_runs=50):
    """Benchmark inference speed of original vs quantized model."""
    logger.info("Benchmarking model performance...")
    
    # Set both models to eval mode
    original_model.eval()
    quantized_model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(dummy_input)
            _ = quantized_model(dummy_input)
    
    # Benchmark original model
    original_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = original_model(dummy_input)
            original_times.append(time.time() - start_time)
    
    # Benchmark quantized model
    quantized_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = quantized_model(dummy_input)
            quantized_times.append(time.time() - start_time)
    
    # Calculate statistics
    avg_original = np.mean(original_times) * 1000  # Convert to ms
    avg_quantized = np.mean(quantized_times) * 1000  # Convert to ms
    speedup = avg_original / avg_quantized
    
    logger.info(f"Original model: {avg_original:.2f}ms per inference")
    logger.info(f"Quantized model: {avg_quantized:.2f}ms per inference")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    print(f"🚀 Speedup with quantization: {speedup:.2f}x faster")
    print(f"   - Original: {avg_original:.2f}ms")
    print(f"   - Quantized: {avg_quantized:.2f}ms")
    
    return {
        "original_ms": avg_original,
        "quantized_ms": avg_quantized,
        "speedup": speedup
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize VAD models')
    parser.add_argument('--model_path', required=True, help='Path to model to quantize')
    parser.add_argument('--output_path', help='Path to save quantized model')
    parser.add_argument('--method', choices=['dynamic', 'static'], default='dynamic',
                       help='Quantization method to use')
    parser.add_argument('--calibration_dataset', help='Path to dataset for static quantization calibration')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark original vs quantized model')
    parser.add_argument('--quantize_conv', action='store_true', help='Also quantize Conv2D layers (not just Linear)')
    
    args = parser.parse_args()
    
    if args.method == 'dynamic':
        quantized_model = dynamic_quantization(
            args.model_path, 
            args.output_path,
            linear_only=not args.quantize_conv
        )
    elif args.method == 'static':
        if not args.calibration_dataset:
            raise ValueError("Calibration dataset required for static quantization")
        quantized_model = static_quantization(
            args.model_path,
            args.calibration_dataset,
            args.output_path
        )
    
    # Load original model for benchmarking if requested
    if args.benchmark:
        checkpoint = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in checkpoint and "hyper_parameters" in checkpoint:
            # Lightning model
            from types import SimpleNamespace
            hp = SimpleNamespace(**checkpoint["hyper_parameters"])
            pl_model = VADLightning(hp)
            pl_model.load_state_dict(checkpoint["state_dict"])
            original_model = pl_model.net
        else:
            # PyTorch model
            from models import MelPerformer
            # Try to detect dimension
            if "pos_embedding" in checkpoint:
                dim = checkpoint["pos_embedding"].shape[2]
            else:
                dim = DEFAULT_DIMENSION
                
            original_model = MelPerformer(
                n_mels=DEFAULT_N_MELS,
                dim=dim,
                layers=DEFAULT_LAYERS,
                heads=DEFAULT_HEADS,
                max_seq_len=DEFAULT_MAX_FRAMES
            )
            original_model.load_state_dict(checkpoint)
        
        benchmark_models(original_model, quantized_model)