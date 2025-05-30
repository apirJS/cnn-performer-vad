#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  test_boundary_settings.py - Test different boundary detection settings
# ───────────────────────────────────────────────────────────────────────

import os
import sys
import pathlib
import subprocess

def run_evaluation(model_path, output_dir, settings):
    """Run evaluation with specific boundary detection settings."""
    cmd = [
        "python", "cli.py", "evaluate",
        "--model_path", model_path,
        "--test_manifest", "D:\\belajar\\audio\\vad\\datasets\\manifest_test.csv",
        "--output_dir", output_dir,
        "--boundary_analysis",
        "--smoothing_window_ms", str(settings["window"]), 
        "--min_segment_duration_ms", str(settings["min_dur"]),
        "--max_gap_duration_ms", str(settings["max_gap"])
    ]
    
    print(f"Running evaluation with settings: {settings}")
    subprocess.run(cmd)

def main():
    # Base model path
    model_path = "D:\\belajar\\audio\\vad\\lightning_logs\\lightning_logs\\version_8\\checkpoints\\14-0.0152-0.9280.ckpt"
    
    # Test different settings combinations
    settings_to_test = [
        {"name": "baseline", "window": 70, "min_dur": 200, "max_gap": 100},
        {"name": "optimized", "window": 50, "min_dur": 150, "max_gap": 150},
        {"name": "aggressive_merging", "window": 50, "min_dur": 100, "max_gap": 200},
        {"name": "precise_boundaries", "window": 30, "min_dur": 100, "max_gap": 100},
    ]
    
    # Create base output directory
    base_output_dir = pathlib.Path("D:\\belajar\\audio\\vad\\param_test")
    base_output_dir.mkdir(exist_ok=True)
    
    # Run evaluations for each setting
    for settings in settings_to_test:
        output_dir = base_output_dir / settings["name"]
        run_evaluation(
            model_path, 
            str(output_dir),
            {
                "window": settings["window"],
                "min_dur": settings["min_dur"],
                "max_gap": settings["max_gap"]
            }
        )
        
    print("All evaluations complete!")
    print(f"Results saved to {base_output_dir}")

if __name__ == "__main__":
    main()