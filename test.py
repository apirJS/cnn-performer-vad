import torch
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
from models import VADLightning

def test_onnx_model():
    # Load PyTorch model
    print("Loading PyTorch model...")
    checkpoint = torch.load("D:/belajar/audio/vad/lightning_logs/finetuning/lightning_logs/version_1/checkpoints/18-0.0164-0.9327.ckpt" , map_location="cpu")
    hp = checkpoint["hyper_parameters"]
    vad_model = VADLightning.load_from_checkpoint("D:/belajar/audio/vad/lightning_logs/finetuning/lightning_logs/version_1/checkpoints/18-0.0164-0.9327.ckpt", hp=hp).eval().cpu()
    model = vad_model.net

    # Freeze projections - crucial for matching behavior
    for name, module in model.named_modules():
        if hasattr(module, 'feature_redraw_interval'):
            print(f"Freezing {name}")
            module.feature_redraw_interval = 0

    # Load ONNX model
    onnx_path = "D:/belajar/audio/vad/models/vad.onnx"
    print(f"Loading ONNX model from {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Get input and output names
    input_name = sess.get_inputs()[0].name
    print(f"ONNX input name: {input_name}")
    output_name = sess.get_outputs()[0].name
    print(f"ONNX output name: {output_name}")

    # Test with fixed sequence length and controlled random seed
    torch.manual_seed(42)  # For reproducible results
    x = torch.randn(1, 1000, 80)
    
    # Get PyTorch prediction
    with torch.no_grad():
        # Raw logits for comparison if ONNX also returns logits
        pt_logits = model(x).cpu().numpy()
        # Apply sigmoid for probability space (final predictions)
        pt_probs = torch.sigmoid(torch.tensor(pt_logits)).numpy()

    # Run ONNX model
    onnx_out = sess.run(None, {input_name: x.numpy()})[0]
    
    # Check if ONNX output is in probability space (0-1) or logit space
    is_prob_space = np.all((onnx_out >= 0) & (onnx_out <= 1))
    print(f"ONNX output appears to be in {'probability space (0-1)' if is_prob_space else 'logit space'}")
    
    # Compare appropriate values (logits to logits or probs to probs)
    if is_prob_space:
        # Compare probabilities
        pytorch_compare = pt_probs
        max_diff = np.abs(pt_probs - onnx_out).max()
        print(f"PyTorch probs range: {pt_probs.min():.4f} to {pt_probs.max():.4f}")
        print(f"ONNX probs range: {onnx_out.min():.4f} to {onnx_out.max():.4f}")
    else:
        # Compare logits
        pytorch_compare = pt_logits
        max_diff = np.abs(pt_logits - onnx_out).max()
        print(f"PyTorch logits range: {pt_logits.min():.4f} to {pt_logits.max():.4f}")
        print(f"ONNX logits range: {onnx_out.min():.4f} to {onnx_out.max():.4f}")
    
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Calculate correlation to see if outputs are related even if range differs
    correlation = np.corrcoef(pytorch_compare.flatten(), onnx_out.flatten())[0, 1]
    print(f"Correlation between PyTorch and ONNX outputs: {correlation:.6f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.scatter(pytorch_compare.flatten(), onnx_out.flatten(), alpha=0.3)
    plt.plot([pytorch_compare.min(), pytorch_compare.max()], 
             [pytorch_compare.min(), pytorch_compare.max()], 'r--')
    plt.xlabel("PyTorch Predictions")
    plt.ylabel("ONNX Predictions")
    plt.title("PyTorch vs ONNX Predictions")
    plt.savefig("onnx_validation_fixed.png")
    print("Comparison plot saved to onnx_validation_fixed.png")
    
    # If correlation is high but values differ, plot a sample of outputs to check pattern
    if correlation > 0.9 and max_diff > 0.1:
        plt.figure(figsize=(12, 5))
        sample_idx = np.random.randint(0, pytorch_compare.shape[1], 100)
        # Fix here: Adjust indexing for 2D arrays
        plt.plot(sample_idx, pytorch_compare[0, sample_idx], 'b-', label='PyTorch')
        plt.plot(sample_idx, onnx_out[0, sample_idx], 'r-', label='ONNX')
        plt.legend()
        plt.title("Sample of outputs - might need rescaling or sigmoid")
        plt.savefig("output_samples.png")
        print("Sample outputs saved to output_samples.png")

if __name__ == "__main__":
    test_onnx_model()