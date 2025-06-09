#!/usr/bin/env python

import os
from boundary_finetune import run_boundary_finetuning
from evaluation import main as evaluate_model

def main():
    # Step 1: Run the boundary fine-tuning
    print("Step 1: Running boundary-focused fine-tuning...")
    run_boundary_finetuning()
    
    # Step 2: Evaluate the fine-tuned model
    print("\nStep 2: Evaluating fine-tuned model...")
    export_path = "D:\\belajar\\audio\\vad\\models\\vad_boundary_enhanced.pt"
    
    # Check if the model was exported successfully
    if os.path.exists(export_path):
        # Run evaluation with boundary analysis
        os.system(f'python evaluation.py --model_path "{export_path}" --test_manifest datasets/manifest_test.csv --boundary_analysis')
    else:
        print(f"Error: Model file not found at {export_path}")

if __name__ == "__main__":
    main()