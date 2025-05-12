#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  inference_enhanced.py - Optimized inference script for VAD model
# ───────────────────────────────────────────────────────────────────────
import argparse
import logging
import pathlib
import os
import sys
import time
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from typing import Optional, Tuple, Union
from scipy.ndimage import gaussian_filter1d, binary_opening, binary_closing

# Import model definitions from train.py
from train import VADLightning

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load a trained VAD model from checkpoint."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Load checkpoint
        model = VADLightning.load_from_checkpoint(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Try alternative loading method
        try:
            checkpoint = torch.load(model_path, map_location=device)
            hparams = checkpoint.get("hyper_parameters", {})
            
            # Convert dict to argparse.Namespace
            import argparse
            hp_namespace = argparse.Namespace(**hparams)
            
            # Create model with converted namespace
            model = VADLightning(hp_namespace)
            
            # Load state dict
            model.load_state_dict(checkpoint["state_dict"])
            logger.info("Model loaded with alternative method")
        except Exception as nested_e:
            logger.error(f"Failed to load model with alternative method: {nested_e}")
            sys.exit(f"❌ Could not load model: {e}")
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    return model

def compute_spectral_flatness(
    waveform: torch.Tensor,
    sample_rate: int = 16000, 
    n_fft: int = 400,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute spectral flatness for improved silence detection."""
    y = waveform.numpy().squeeze()
    
    # Compute spectral flatness - helps detect silence and non-speech regions
    spectral_flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=n_fft, hop_length=hop_length
    ).squeeze()
    
    return spectral_flatness

def process_audio(
    audio_path: str, 
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    win_length: int = 400,
    n_mels: int = 80
) -> Tuple[torch.Tensor, torch.Tensor, float, np.ndarray, np.ndarray]:
    """
    Load and process audio file to mel spectrogram for VAD inference.
    Also compute frame energy and spectral features for enhanced silence detection.
    
    Returns:
        mel: Mel spectrogram tensor (1, T, n_mels)
        waveform: Audio waveform tensor (1, samples)
        duration: Audio duration in seconds
        frame_energy: Energy per frame (T,)
        spectral_flatness: Spectral flatness per frame (T,)
    """
    logger.info(f"Processing audio file: {audio_path}")
    
    # Load audio file
    try:
        if audio_path.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            waveform, sr = torchaudio.load(audio_path)
        else:
            logger.warning(f"Unsupported file format. Trying with librosa...")
            waveform, sr = librosa.load(audio_path, sr=None)
            waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        sys.exit(f"❌ Could not load audio file: {e}")
    
    # Get duration
    duration = waveform.shape[1] / sr
    
    # Resample if needed
    if sr != sample_rate:
        logger.info(f"Resampling from {sr}Hz to {sample_rate}Hz")
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    
    # Ensure mono audio
    if waveform.shape[0] > 1:
        logger.info(f"Converting {waveform.shape[0]} channels to mono")
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Compute frame energy (RMS energy)
    wav_np = waveform.numpy().squeeze()
    frame_energy = librosa.feature.rms(
        y=wav_np, 
        frame_length=win_length, 
        hop_length=hop_length
    ).squeeze()
    
    # Compute spectral flatness for improved silence detection
    spectral_flatness = compute_spectral_flatness(
        waveform, sample_rate, n_fft, hop_length
    )
    
    # Create mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=1.0
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
    # Apply transforms
    mel = db_transform(mel_transform(waveform)).squeeze(0).transpose(0, 1)  # (T, n_mels)
    
    # Add batch dimension
    mel = mel.unsqueeze(0)  # (1, T, n_mels)
    
    logger.info(f"Processed audio with duration {duration:.2f}s, mel shape: {mel.shape}")
    return mel, waveform, duration, frame_energy, spectral_flatness

def run_inference(
    model: torch.nn.Module,
    mel: torch.Tensor, 
    device: str = "cpu",
    threshold: float = 0.46  # Set default threshold to optimal value from evaluation
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run VAD inference on mel spectrogram.
    
    Args:
        model: Trained VAD model
        mel: Mel spectrogram tensor (1, T, n_mels)
        device: Device to run inference on
        threshold: Decision threshold for speech/non-speech
        
    Returns:
        probs_np: Frame-level VAD probabilities (T,)
        predictions: Frame-level VAD predictions (T,)
    """
    logger.info(f"Running inference on mel spectrogram with shape {mel.shape}")
    
    # Move to device
    mel = mel.to(device)
    
    # Run inference
    with torch.no_grad():
        start_time = time.time()
        logits = model(mel)
        elapsed = time.time() - start_time
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
    # Convert to numpy for further processing
    probs_np = probs.cpu().numpy().squeeze()
    
    # Apply threshold
    predictions = (probs_np >= threshold).astype(np.float32)
    
    logger.info(f"Inference completed in {elapsed:.3f}s, output shape: {probs_np.shape}")
    return probs_np, predictions

def post_process_predictions(
    predictions: np.ndarray, 
    probs: np.ndarray,
    frame_energy: np.ndarray,
    spectral_flatness: np.ndarray = None,
    energy_threshold_percentile: int = 40,  # Increased for better silence detection
    min_speech_frames: int = 10,            # Increased for more stable segments
    min_silence_frames: int = 15,           # Increased for better gap detection
    smoothing_sigma: float = 1.5            # Increased for smoother predictions
) -> np.ndarray:
    """
    Apply enhanced post-processing to VAD predictions to improve segmentation.
    
    Args:
        predictions: Raw VAD predictions (0/1 values)
        probs: VAD probability values for smoothing
        frame_energy: Frame energy values for energy masking
        spectral_flatness: Spectral flatness for improved silence detection
        energy_threshold_percentile: Percentile for energy threshold
        min_speech_frames: Minimum number of consecutive speech frames
        min_silence_frames: Minimum number of consecutive silence frames
        smoothing_sigma: Sigma for Gaussian smoothing of probabilities
        
    Returns:
        processed_predictions: Post-processed VAD predictions
    """
    logger.info("Applying enhanced post-processing to VAD predictions")
    
    # 1. Apply smoothing to probabilities
    smoothed_probs = gaussian_filter1d(probs, sigma=smoothing_sigma)
    smoothed_predictions = (smoothed_probs >= 0.46).astype(np.float32)  # Use optimal threshold
    
    # 2. Apply energy masking
    # Calculate energy threshold (automatically adaptive)
    energy_threshold = np.percentile(frame_energy, energy_threshold_percentile)
    logger.info(f"Energy threshold (percentile {energy_threshold_percentile}): {energy_threshold:.6f}")
    
    # Mask low-energy frames as silence regardless of VAD prediction
    energy_mask = (frame_energy > energy_threshold).astype(np.float32)
    
    # 3. Use spectral flatness for additional silence detection if available
    if spectral_flatness is not None:
        # Normalize and invert spectral flatness (lower values = more speech-like)
        norm_flatness = spectral_flatness / np.max(spectral_flatness)
        flatness_threshold = np.percentile(norm_flatness, 70)  # Higher percentile for stricter filtering
        
        # Create mask where low flatness values (speech) are 1, high flatness (silence/noise) are 0
        flatness_mask = (norm_flatness < flatness_threshold).astype(np.float32)
        
        # Combine with energy mask (both must agree for speech)
        combined_mask = energy_mask * flatness_mask
        energy_mask = combined_mask
    
    # Apply the mask to predictions
    masked_predictions = smoothed_predictions * energy_mask
    
    # 4. Apply morphological operations to remove small gaps and spikes
    # Convert to boolean for binary operations
    predictions_bool = masked_predictions.astype(bool)
    
    # Remove small silence gaps within speech segments (binary closing)
    if min_silence_frames > 1:
        closed_predictions = binary_closing(predictions_bool, structure=np.ones(min_silence_frames))
    else:
        closed_predictions = predictions_bool
        
    # Remove small speech segments (binary opening)
    if min_speech_frames > 1:
        opened_predictions = binary_opening(closed_predictions, structure=np.ones(min_speech_frames))
    else:
        opened_predictions = closed_predictions
    
    # Convert back to float32
    processed_predictions = opened_predictions.astype(np.float32)
    
    # Calculate improvement metrics
    speech_ratio_before = np.mean(predictions)
    speech_ratio_after = np.mean(processed_predictions)
    
    logger.info(f"Post-processing: Speech ratio before: {speech_ratio_before*100:.1f}%, after: {speech_ratio_after*100:.1f}%")
    return processed_predictions

def visualize_results(
    waveform: torch.Tensor,
    vad_probs: np.ndarray,
    vad_predictions: np.ndarray,
    frame_energy: np.ndarray,
    spectral_flatness: np.ndarray = None,
    sample_rate: int = 16000,
    hop_length: int = 160,
    threshold: float = 0.46,  # Updated to optimal threshold
    output_path: Optional[str] = None
):
    """Visualize audio waveform with VAD probabilities, predictions and audio features."""
    logger.info("Visualizing VAD results")
    
    # Convert waveform to numpy
    waveform_np = waveform.squeeze().cpu().numpy()
    
    # Create time axis for waveform
    waveform_time = np.arange(len(waveform_np)) / sample_rate
    
    # Create time axis for VAD frames
    vad_time = np.arange(len(vad_probs)) * hop_length / sample_rate
    
    # Determine number of subplots
    n_plots = 5 if spectral_flatness is not None else 4
    
    # Create visualization
    plt.figure(figsize=(14, 12))
    
    # Plot waveform
    plt.subplot(n_plots, 1, 1)
    plt.plot(waveform_time, waveform_np, color='black', alpha=0.7)
    plt.title("Audio Waveform")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    
    # Plot frame energy
    plt.subplot(n_plots, 1, 2)
    energy_time = np.arange(len(frame_energy)) * hop_length / sample_rate
    
    # Normalize energy for better visualization
    normalized_energy = frame_energy / (np.max(frame_energy) + 1e-10)
    
    plt.plot(energy_time, normalized_energy, color='purple', alpha=0.8)
    
    # Plot energy threshold
    energy_threshold = np.percentile(frame_energy, 40)  # Using 40th percentile
    normalized_threshold = energy_threshold / (np.max(frame_energy) + 1e-10)
    plt.axhline(y=normalized_threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Energy threshold ({normalized_threshold:.3f})')
    
    plt.title("Frame Energy")
    plt.ylabel("Normalized Energy")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Plot spectral flatness if available
    current_plot = 3
    if spectral_flatness is not None:
        plt.subplot(n_plots, 1, current_plot)
        flatness_time = np.arange(len(spectral_flatness)) * hop_length / sample_rate
        
        # Normalize spectral flatness
        norm_flatness = spectral_flatness / np.max(spectral_flatness)
        
        plt.plot(flatness_time, norm_flatness, color='orange', alpha=0.8)
        
        # Plot flatness threshold
        flatness_threshold = np.percentile(norm_flatness, 70)
        plt.axhline(y=flatness_threshold, color='r', linestyle='--', alpha=0.5,
                   label=f'Flatness threshold ({flatness_threshold:.3f})')
        
        plt.title("Spectral Flatness (higher = more noise-like)")
        plt.ylabel("Normalized Flatness")
        plt.grid(alpha=0.3)
        plt.legend()
        
        current_plot += 1
    
    # Plot VAD probabilities
    plt.subplot(n_plots, 1, current_plot)
    plt.plot(vad_time, vad_probs, color='blue', alpha=0.8)
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, 
                label=f'Threshold ({threshold:.2f})')
    plt.title("VAD Probabilities")
    plt.ylabel("Speech Probability")
    plt.grid(alpha=0.3)
    plt.legend()
    
    # Plot binary VAD predictions
    plt.subplot(n_plots, 1, current_plot + 1)
    plt.step(vad_time, vad_predictions, color='green', alpha=0.8, where='post')
    plt.title("VAD Predictions (Speech Segments)")
    plt.xlabel("Time (s)")
    plt.ylabel("Speech (1) / Non-speech (0)")
    plt.yticks([0, 1])
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def get_speech_segments(
    predictions: np.ndarray, 
    hop_length: int = 160, 
    sample_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 200  # Increased for better segmentation
) -> list:
    """
    Extract speech segment timestamps from frame-level predictions.
    
    Args:
        predictions: Frame-level VAD predictions (T,)
        hop_length: Hop length used for feature extraction
        sample_rate: Audio sample rate
        min_speech_duration_ms: Minimum speech segment duration in milliseconds
        min_silence_duration_ms: Minimum silence duration to split segments
        
    Returns:
        segments: List of (start_time, end_time) tuples in seconds
    """
    logger.info("Extracting speech segments from predictions")
    
    # Calculate minimum speech and silence frames
    min_speech_frames = int(min_speech_duration_ms / 1000 * sample_rate / hop_length)
    min_silence_frames = int(min_silence_duration_ms / 1000 * sample_rate / hop_length)
    
    logger.info(f"Minimum speech frames: {min_speech_frames}, minimum silence frames: {min_silence_frames}")
    
    # Find speech segments
    speech_segments = []
    in_speech = False
    speech_start = 0
    silence_counter = 0
    
    for i, pred in enumerate(predictions):
        if not in_speech and pred == 1:
            # Speech start
            in_speech = True
            speech_start = i
            silence_counter = 0
        elif in_speech and pred == 0:
            # Potential speech end - count consecutive silence frames
            silence_counter += 1
            
            # If we have enough silence frames, end the segment
            if silence_counter >= min_silence_frames:
                in_speech = False
                speech_length = i - speech_start - silence_counter
                
                # Only keep if segment is long enough
                if speech_length >= min_speech_frames:
                    # Convert frame indices to time
                    start_time = speech_start * hop_length / sample_rate
                    end_time = (i - silence_counter) * hop_length / sample_rate
                    speech_segments.append((start_time, end_time))
                
                silence_counter = 0
        elif in_speech and pred == 1:
            # Reset silence counter if we're back to speech
            silence_counter = 0
    
    # Handle the case where speech runs until the end
    if in_speech:
        speech_length = len(predictions) - speech_start
        if speech_length >= min_speech_frames:
            start_time = speech_start * hop_length / sample_rate
            end_time = len(predictions) * hop_length / sample_rate
            speech_segments.append((start_time, end_time))
    
    logger.info(f"Found {len(speech_segments)} speech segments")
    return speech_segments

def extract_speech_segments(
    audio_path: str, 
    segments: list, 
    output_dir: str,
    sample_rate: int = 16000
):
    """
    Extract speech segments from audio file and save them.
    
    Args:
        audio_path: Path to input audio file
        segments: List of (start_time, end_time) tuples in seconds
        output_dir: Directory to save extracted segments
        sample_rate: Target sample rate for extracted segments
    """
    logger.info(f"Extracting {len(segments)} speech segments to {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Extract and save each segment
    for i, (start, end) in enumerate(segments):
        # Convert time to samples
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        # Extract segment
        segment = y[start_sample:end_sample]
        
        # Create output filename
        output_path = os.path.join(output_dir, f"{base_name}_segment_{i+1:03d}.wav")
        
        # Save segment
        sf.write(output_path, segment, sr)
        
        logger.info(f"Saved segment {i+1} ({end-start:.2f}s) to {output_path}")
    
    logger.info(f"All segments extracted successfully")

def main():
    """Main function for VAD inference."""
    parser = argparse.ArgumentParser(description="Run enhanced VAD inference on audio files")
    
    # Model arguments
    parser.add_argument(
        "--model_path", 
        default="lightning_logs/lightning_logs/version_1/checkpoints/04-0.0038-0.9810.ckpt",
        help="Path to trained VAD model checkpoint"
    )
    
    # Audio arguments
    parser.add_argument(
        "--audio_path", 
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=16000, 
        help="Sample rate for audio processing"
    )
    
    # Mel spectrogram arguments
    parser.add_argument(
        "--n_mels", 
        type=int, 
        default=80, 
        help="Number of mel bands in spectrogram"
    )
    parser.add_argument(
        "--n_fft", 
        type=int, 
        default=400, 
        help="FFT size for spectrogram"
    )
    parser.add_argument(
        "--hop_length", 
        type=int, 
        default=160, 
        help="Hop length for spectrogram"
    )
    parser.add_argument(
        "--win_length", 
        type=int, 
        default=400, 
        help="Window length for spectrogram"
    )
    
    # Inference arguments
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.46,  # Updated to optimal threshold from evaluation
        help="Threshold for speech/non-speech decision"
    )
    parser.add_argument(
        "--device", 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    
    # Post-processing arguments
    parser.add_argument(
        "--energy_threshold", 
        type=int, 
        default=40, # Increased for better silence detection
        help="Energy threshold percentile for silence detection"
    )
    parser.add_argument(
        "--min_speech_duration", 
        type=int, 
        default=250,
        help="Minimum speech segment duration in milliseconds"
    )
    parser.add_argument(
        "--min_silence_duration", 
        type=int, 
        default=200, # Increased for better gap detection
        help="Minimum silence duration between segments in milliseconds"
    )
    parser.add_argument(
        "--smoothing_sigma", 
        type=float, 
        default=1.5, # Increased for smoother predictions
        help="Sigma for Gaussian smoothing of probabilities"
    )
    
    # Output arguments
    parser.add_argument(
        "--output_plot", 
        default=None,
        help="Path to save visualization plot"
    )
    parser.add_argument(
        "--extract_segments", 
        action="store_true",
        help="Extract and save speech segments"
    )
    parser.add_argument(
        "--segments_dir", 
        default="speech_segments",
        help="Directory to save extracted speech segments"
    )
    
    args = parser.parse_args()
    
    # Validate audio path
    if not os.path.exists(args.audio_path):
        logger.error(f"Audio file not found: {args.audio_path}")
        sys.exit("❌ Audio file not found")
    
    # Validate model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit("❌ Model file not found")
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Process audio
    mel, waveform, duration, frame_energy, spectral_flatness = process_audio(
        args.audio_path,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels
    )
    
    # Run inference
    vad_probs, raw_predictions = run_inference(
        model,
        mel,
        device=args.device,
        threshold=args.threshold
    )
    
    # Apply post-processing
    vad_predictions = post_process_predictions(
        raw_predictions,
        probs=vad_probs,
        frame_energy=frame_energy,
        spectral_flatness=spectral_flatness,
        energy_threshold_percentile=args.energy_threshold,
        min_speech_frames=int(args.min_speech_duration / 1000 * args.sample_rate / args.hop_length),
        min_silence_frames=int(args.min_silence_duration / 1000 * args.sample_rate / args.hop_length),
        smoothing_sigma=args.smoothing_sigma
    )
    
    # Visualize results
    visualize_results(
        waveform,
        vad_probs,
        vad_predictions,
        frame_energy,
        spectral_flatness=spectral_flatness,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        threshold=args.threshold,
        output_path=args.output_plot
    )
    
    # Get speech segments
    segments = get_speech_segments(
        vad_predictions,
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        min_speech_duration_ms=args.min_speech_duration,
        min_silence_duration_ms=args.min_silence_duration
    )
    
    # Extract speech segments if requested
    if args.extract_segments and segments:
        extract_speech_segments(
            args.audio_path,
            segments,
            args.segments_dir,
            sample_rate=args.sample_rate
        )
        
        print(f"\n✅ Extracted {len(segments)} speech segments to {args.segments_dir}")
    elif args.extract_segments:
        print("\n⚠️ No speech segments found meeting the minimum duration")
    
    # Print summary of results
    speech_percentage = np.mean(vad_predictions) * 100
    print(f"\n✅ VAD inference completed!")
    print(f"📊 Audio duration: {duration:.2f} seconds")
    print(f"📊 Speech detected: {speech_percentage:.1f}% of the time")
    print(f"📊 Threshold used: {args.threshold}")
    print(f"📊 Energy threshold percentile: {args.energy_threshold}")
    
    # Print segment information
    if segments:
        print(f"\n🔊 Found {len(segments)} speech segments:")
        for i, (start, end) in enumerate(segments):
            print(f"  - Segment {i+1:>2}: {start:.2f}s to {end:.2f}s ({end-start:.2f}s)")
    else:
        print("\n⚠️ No speech segments detected")
        
    logger.info("Inference script completed successfully")

if __name__ == "__main__":
    main()