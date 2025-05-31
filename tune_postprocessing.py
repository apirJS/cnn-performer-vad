import numpy as np
import json
from pathlib import Path
from evaluation import analyze_speech_boundaries

# Load saved predictions and labels
frame_preds = np.load('evaluation_results/frame_predictions.npy')
frame_labels = np.load('evaluation_results/frame_labels.npy')

# Parameters to tune
smoothing_windows = [5, 7, 10, 15]
min_segment_durations = [80, 100, 130, 160]
max_gaps = [30, 50, 70, 90]
high_thresholds = [0.5, 0.55, 0.6, 0.65]
low_thresholds = [0.3, 0.35, 0.4, 0.45]

# Parsing arguments for analyze_speech_boundaries
class Args:
    def __init__(self):
        self.sample_rate = 16000
        self.hop = 160
        self.output_dir = 'evaluation_results/tuning'
        self.iou_threshold = 0.5
        
best_f1 = 0
best_config = {}

# Create output dir
Path('evaluation_results/tuning').mkdir(parents=True, exist_ok=True)

# Test combinations
for smoothing in smoothing_windows:
    for min_dur in min_segment_durations:
        for gap in max_gaps:
            for high_t in high_thresholds:
                for low_t in low_thresholds:
                    # Only test valid threshold combinations
                    if low_t >= high_t:
                        continue
                    
                    args = Args()
                    args.smoothing_window_ms = smoothing
                    args.min_segment_duration_ms = min_dur
                    args.max_gap_duration_ms = gap
                    
                    # Analyze with current config
                    print(f"Testing: smooth={smoothing}, min_dur={min_dur}, gap={gap}, thresholds={high_t}/{low_t}")
                    boundary_metrics = analyze_speech_boundaries(
                        frame_preds, 
                        frame_labels,
                        threshold=0.5,  # Base threshold, high/low will be adjusted
                        args=args,
                        output_dir=f'evaluation_results/tuning/{smoothing}_{min_dur}_{gap}_{high_t}_{low_t}'
                    )
                    
                    # Check if this is better
                    f1 = boundary_metrics.get('segment_f1', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_config = {
                            'smoothing_window_ms': smoothing,
                            'min_segment_duration_ms': min_dur,
                            'max_gap_duration_ms': gap,
                            'high_threshold': high_t,
                            'low_threshold': low_t,
                            'segment_f1': f1,
                            'matched_segments': boundary_metrics.get('matched_true_segments', 0),
                            'total_segments': boundary_metrics.get('total_true_segments', 0)
                        }
                        print(f"New best config found! F1: {f1:.4f}")

# Save best configuration
with open('evaluation_results/tuning/best_config.json', 'w') as f:
    json.dump(best_config, f, indent=4)
    
print(f"Best configuration: {best_config}")