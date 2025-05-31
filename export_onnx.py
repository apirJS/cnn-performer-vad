#!/usr/bin/env python
# filepath: d:\belajar\audio\vad\export_onnx.py
import torch
import argparse
import logging
import sys
from models import VADLightning, MelPerformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("export_wrapper")


# Create an export-friendly wrapper class
class ExportableVAD(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self._disable_reversible_layers()
        self._freeze_random_projections()
    
    def _disable_reversible_layers(self):
        """Disable all reversible computation"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'reversible'):
                logger.info(f"Disabling reversible for: {name}")
                module.reversible = False
    
    def _freeze_random_projections(self):
        """Force regenerate and freeze all random projections"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'feature_redraw_interval'):
                logger.info(f"Freezing projections for: {name}")
                module.feature_redraw_interval = None
                
                # Force regenerate projections if possible
                if hasattr(module, 'redraw_projections'):
                    with torch.no_grad():
                        module.redraw_projections()
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

def main():
    parser = argparse.ArgumentParser(description="Export VAD model to ONNX")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--output", default="vad_exported.onnx", help="Output path")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    args = parser.parse_args()
    
    # Load the model
    logger.info(f"Loading model from: {args.ckpt}")
    try:
        ckpt = torch.load(args.ckpt, map_location='cpu')  # Explicitly load to CPU
        if "hyper_parameters" in ckpt:
            # Lightning checkpoint
            hp = ckpt["hyper_parameters"]
            logger.info("Loading Lightning checkpoint")
            vad_model = VADLightning.load_from_checkpoint(args.ckpt, hp=hp, map_location='cpu')
            model = vad_model.net  # Get just the MelPerformer part
        else:
            # Regular PyTorch checkpoint
            logger.info("Loading regular checkpoint")
            model = MelPerformer(n_mels=80, dim=192, layers=4, heads=4)
            model.load_state_dict(ckpt)
        
        # Ensure the model is on CPU and in eval mode
        model = model.cpu().eval()
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Create the exportable wrapper
    wrapper = ExportableVAD(model)
    wrapper.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 500, 80)
    
    # Try a test run
    logger.info("Testing forward pass...")
    with torch.no_grad():
        test_output = wrapper(dummy_input)
        logger.info(f"Test output shape: {test_output.shape}")
    
    # Export to ONNX
    logger.info(f"Exporting to ONNX with opset {args.opset}...")
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            args.output,
            verbose=False,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {1: 'sequence_length'},
                'output': {1: 'sequence_length'},
            }
        )
        logger.info(f"Successfully exported to {args.output}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()