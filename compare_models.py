#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  compare_models.py - Compare regular and boundary-tuned models
# ───────────────────────────────────────────────────────────────────────

import subprocess
import pathlib

def compare_models():
    """Compare regular and boundary-tuned models with optimized parameters."""
    
    # Paths to models
    regular_model = "D:\\belajar\\audio\\vad\\lightning_logs\\lightning_logs\\version_8\\checkpoints\\14-0.0152-0.9280.ckpt"
    boundary_tuned_model = "D:\\belajar\\audio\\vad\\models\\model_boundary_tuned.pt"
    
    # Output directories
    regular_output = "D:\\belajar\\audio\\vad\\comparison\\regular"
    boundary_output = "D:\\belajar\\audio\\vad\\comparison\\boundary_tuned"
    
    # Ensure output directories exist
    for dir_path in [regular_output, boundary_output]:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Common evaluation parameters
    eval_params = [
        "--test_manifest", "D:\\belajar\\audio\\vad\\datasets\\manifest_test.csv",
        "--boundary_analysis", 
        "--smoothing_window_ms", "50",
        "--min_segment_duration_ms", "150", 
        "--max_gap_duration_ms", "150"
    ]
    
    # Evaluate regular model
    print("Evaluating regular model...")
    regular_cmd = [
        "python", "cli.py", "evaluate",
        "--model_path", regular_model,
        "--output_dir", regular_output
    ] + eval_params
    subprocess.run(regular_cmd)
    
    # Evaluate boundary-tuned model
    print("Evaluating boundary-tuned model...")
    boundary_cmd = [
        "python", "cli.py", "evaluate",
        "--model_path", boundary_tuned_model,
        "--output_dir", boundary_output
    ] + eval_params
    subprocess.run(boundary_cmd)
    
    print("Comparison complete!")
    print(f"Results saved to:")
    print(f"  - Regular model: {regular_output}")
    print(f"  - Boundary-tuned model: {boundary_output}")

if __name__ == "__main__":
    compare_models()