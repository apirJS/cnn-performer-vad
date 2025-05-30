# SileroVAD and Singing Voice Detection Playground
import librosa
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from IPython.display import Audio, display

from prepare_data import initialize_silero_vad, generate_silero_vad_labels, get_original_path
from utils import label_singing_vocals_heuristically

# Initialize Silero VAD model
model, utils = initialize_silero_vad()
vad_model = (model, utils)

def analyze_audio(
    audio_path,
    silero_threshold=0.5,
    singing_energy_threshold=5,
    singing_voiced_threshold=0.2,
    use_normalization=True,
    use_preemphasis=True,
    use_alt_method=True,
    detect_quiet=True
):
    """Analyze audio using both Silero VAD and singing voice detection"""
    
    # Load audio
    audio_path = Path(audio_path)
    original_path = get_original_path(audio_path)
    
    # Display audio waveform and allow playback
    y, sr = librosa.load(original_path, sr=16000)
    
    # Generate Silero VAD labels
    silero_labels = generate_silero_vad_labels(
        y, sr, model,
        threshold=silero_threshold,
        hop_length=160,
        win_length=400,
        utils=utils
    )
    
    # Generate singing voice detection labels
    singing_labels = label_singing_vocals_heuristically(
        original_path,
        frame_hop_length=160,
        sample_rate=sr,
        energy_thresh_percentile=singing_energy_threshold,
        voiced_prob_threshold=singing_voiced_threshold,
        apply_normalization=use_normalization,
        apply_preemphasis=use_preemphasis,
        use_alternative_method=use_alt_method,
        detect_quiet_beginnings=detect_quiet
    )
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # 1. Waveform
    axes[0].plot(np.arange(len(y)) / sr, y)
    axes[0].set_title("Audio Waveform")
    axes[0].set_ylabel("Amplitude")
    
    # 2. Silero VAD labels
    vad_times = librosa.frames_to_time(np.arange(len(silero_labels)), sr=sr, hop_length=160)
    axes[1].plot(vad_times, silero_labels, 'r')
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].set_title(f"Silero VAD Labels (threshold={silero_threshold})")
    axes[1].set_ylabel("VAD Label")
    
    # 3. Singing voice detection labels
    vad_times = librosa.frames_to_time(np.arange(len(singing_labels)), sr=sr, hop_length=160)
    axes[2].plot(vad_times, singing_labels, 'g')
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].set_title(f"Singing Voice Detection (energy={singing_energy_threshold}%, voiced={singing_voiced_threshold})")
    axes[2].set_ylabel("VAD Label")
    axes[2].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()
    
    # Calculate statistics
    silero_active = np.mean(silero_labels) * 100
    singing_active = np.mean(singing_labels) * 100
    agreement = np.mean(silero_labels == singing_labels) * 100
    
    print(f"Silero VAD detected activity: {silero_active:.2f}% of frames")
    print(f"Singing voice detection: {singing_active:.2f}% of frames")
    print(f"Agreement between methods: {agreement:.2f}%")
    
    # Display audio for playback
    display(Audio(y, rate=sr))
    
    return y, sr, silero_labels, singing_labels

def debug_singing_detection(audio_path):
    """Visualize all features used in singing voice detection"""
    audio_path = Path(audio_path)
    original_path = get_original_path(audio_path)
    y, sr = librosa.load(original_path, sr=16000)
    
    # Extract features 
    rms_frame_len = int(sr * 25 / 1000)
    rms_hop_len = int(sr * 10 / 1000)
    
    # Apply pre-emphasis for energy calculation
    y_for_energy = librosa.effects.preemphasis(y, coef=0.97)
    rms_energy = librosa.feature.rms(
        y=y_for_energy, 
        frame_length=rms_frame_len,
        hop_length=rms_hop_len, 
        center=True
    )[0]
    
    # Calculate pitch 
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=rms_frame_len,
        hop_length=rms_hop_len,
        center=True,
        fill_na=0.0,
    )
    
    # Generate energy threshold
    energy_thresh_percentile = 5  # Default
    energy_threshold = np.percentile(rms_energy, energy_thresh_percentile)
    
    # Generate final labels using our function
    labels = label_singing_vocals_heuristically(
        original_path,
        frame_hop_length=160,
        sample_rate=sr,
        energy_thresh_percentile=energy_thresh_percentile,
        voiced_prob_threshold=0.2,
    )
    
    # Create plots
    fig, axes = plt.subplots(5, 1, figsize=(16, 15), sharex=True)
    
    # 1. Waveform
    time = np.arange(len(y)) / sr
    axes[0].plot(time, y)
    axes[0].set_title("Audio Waveform")
    axes[0].set_ylabel("Amplitude")
    
    # 2. RMS Energy
    feature_times = librosa.frames_to_time(np.arange(len(rms_energy)), sr=sr, hop_length=rms_hop_len)
    axes[1].plot(feature_times, rms_energy)
    axes[1].axhline(y=energy_threshold, color='r', linestyle='--', label=f'{energy_thresh_percentile}th percentile')
    axes[1].set_title("RMS Energy")
    axes[1].set_ylabel("Energy")
    axes[1].legend()
    
    # 3. Pitch (F0)
    valid_f0 = f0.copy()
    valid_f0[np.isnan(valid_f0)] = 0
    axes[2].scatter(feature_times, valid_f0, s=2, alpha=0.7)
    axes[2].set_title("Pitch (F0)")
    axes[2].set_ylabel("Frequency (Hz)")
    
    # 4. Voicing Probability
    axes[3].plot(feature_times, voiced_prob)
    axes[3].axhline(y=0.2, color='g', linestyle='--', label='Threshold 0.2')
    axes[3].axhline(y=0.3, color='y', linestyle='--', label='Threshold 0.3')
    axes[3].axhline(y=0.45, color='r', linestyle='--', label='Threshold 0.45')
    axes[3].set_title("Voicing Probability")
    axes[3].set_ylabel("Probability")
    axes[3].legend()
    
    # 5. Final VAD Labels
    vad_times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=160)
    axes[4].plot(vad_times, labels, 'g')
    axes[4].set_ylim([-0.1, 1.1])
    axes[4].set_title("Singing Voice Detection Final Labels")
    axes[4].set_ylabel("VAD Label")
    axes[4].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()
    
    # Display audio for playback
    display(Audio(y, rate=sr))

def compare_parameters(
    audio_path,
    energy_thresholds=[5, 10, 20],
    voiced_thresholds=[0.2, 0.3, 0.45]
):
    """Compare different parameter combinations for singing voice detection"""
    
    audio_path = Path(audio_path)
    original_path = get_original_path(audio_path)
    y, sr = librosa.load(original_path, sr=16000)
    
    # Generate reference Silero VAD labels
    silero_labels = generate_silero_vad_labels(
        y, sr, model,
        hop_length=160,
        win_length=400,
        utils=utils
    )
    
    # Setup plot
    n_rows = len(energy_thresholds) * len(voiced_thresholds) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, n_rows*2), sharex=True)
    
    # Plot waveform
    axes[0].plot(np.arange(len(y)) / sr, y)
    axes[0].set_title("Audio Waveform")
    axes[0].set_ylabel("Amplitude")
    
    # Test all parameter combinations
    i = 1
    for energy in energy_thresholds:
        for voiced in voiced_thresholds:
            # Generate labels
            labels = label_singing_vocals_heuristically(
                original_path,
                frame_hop_length=160,
                sample_rate=sr,
                energy_thresh_percentile=energy,
                voiced_prob_threshold=voiced,
                apply_normalization=True,
                apply_preemphasis=True,
                use_alternative_method=True,
                detect_quiet_beginnings=True
            )
            
            # Plot
            vad_times = librosa.frames_to_time(np.arange(len(labels)), sr=sr, hop_length=160)
            axes[i].plot(vad_times, labels, 'g')
            axes[i].set_ylim([-0.1, 1.1])
            axes[i].set_title(f"Energy={energy}%, Voiced={voiced} → {np.mean(labels)*100:.2f}% active")
            axes[i].set_ylabel("VAD Label")
            i += 1
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    
    # Display audio for playback
    display(Audio(y, rate=sr))

# Example usage - testing a very quiet singing file:
audio_path = "D:/belajar/audio/vad/datasets/FULL/male2/long_tones/straight/m2_long_straight_e.wav"

# Basic comparison of methods
analyze_audio(audio_path, 
              silero_threshold=0.5,
              singing_energy_threshold=5, 
              singing_voiced_threshold=0.2)

# Detailed debugging visualization
debug_singing_detection(audio_path)

# Compare multiple parameter combinations
compare_parameters(
    audio_path,
    energy_thresholds=[3, 5, 10], 
    voiced_thresholds=[0.15, 0.2, 0.3]
)