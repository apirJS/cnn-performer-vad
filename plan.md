## End-to-End Project Plan

*(from raw audio to a production-ready VAD checkpoint)*

---
### 1  Data Preparation

| Step                                     | What happens                                                                                                                                                                                                                    | Key scripts / folders                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **1.1  Collect & stage corpora**         | • **Speech**: LibriSpeech (clean), FLEURS (102 langs), VocalSet (singing), MUSAN speech<br>• **Negatives**: MUSAN noise, ESC-50, UrbanSound8K, MUSDB18HQ music                                                                  | `prepare_data.py:process_*()` functions          |
| **1.2  Pre-process audio**               | • Resample to **16 kHz** mono<br>• Normalize audio<br>• Validate quality (rejecting low RMS, DC bias, clipping)                                                                                                                 | `utils.py:load_and_process_audio()`              |
| **1.3  Automatic frame-level labelling** | • Use **Silero-VAD** to generate frame labels for speech<br>• Create multi-type positive samples (clean, noisy, overlapped, music-mixed)<br>• Write `*_labels.npy` alongside each WAV                                           | `prepare_data.py:generate_silero_vad_labels()`   |
| **1.4  Hard-negative mining**            | • Create balanced quota of challenging negatives:<br>• Pure noise/music samples<br>• Mixed noise+music combinations<br>• ESC-50 environmental sounds<br>• Urban sounds<br>• Music-to-music mixtures                              | `generate_samples_refactored.py:make_pos_neg()`  |
| **1.5  Augmentation**                    | • Apply time-stretch (0.85-1.15)<br>• Pitch shift (±2 semitones)<br>• Add reverb and EQ<br>• Vary SNR from -10dB to +25dB<br>• Create whispered speech samples                                                                  | `prepare_data.py:safe_augment_audio()`           |
| **1.6  Manifest generation**             | • Create CSV with columns: `path,label,frame_labels`<br>• 85% neg-to-pos ratio for balanced learning<br>• Option to cache pre-computed Mel-spectrograms                                                                         | `prepare_data.py:create_manifest()`              |
---

### 2  Dataset & Loader

| Component              | Implementation                                                                                                           | Notes                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| **CSVMelDataset**      | `CSVMelDataset(manifest, n_mels=64, n_fft=400, hop=160, win=400)`                                                        | On-the-fly WAV → log-Mel conversion                              |
| **Data augmentation**  | `spec_augment(mel)` applies time/freq masking                                                                            | 70% chance during training                                       |
| **Error handling**     | Auto-retry with max 3 attempts, filters corrupted files                                                                  | Falls back to librosa if torchaudio fails                        |
| **Caching**            | Optional mel-spec caching to disk with parameter validation                                                              | `cache_dir="/path/to/cache"`                                     |
| **Collation**          | `collate_pad(batch, max_frames=1000)` handles variable-length sequences                                                  | Returns `(mel, labels, mask)` with padding                       |
| **DataModule**         | PyTorch Lightning wrapper with configurable workers and batch size                                                       | Handles both training and validation loaders                     |

```python
# Dataset creation
train_ds = CSVMelDataset(
    manifest="manifests/train.csv",
    n_mels=64,                    # Spectrogram height
    time_mask_max=50,             # Max time mask size
    freq_mask_max=10,             # Max frequency mask size
    cache_dir="cache/mels"        # Optional caching
)

# Lightning DataModule
data_module = VADDataModule(
    train_dataset=train_ds,
    val_dataset=val_ds,
    batch_size=16,
    num_workers=4,
    max_frames=1000               # Max sequence length
)
```

---

### 3  Model Architecture

| Block                 | Shape                                                     | Notes                                                           |
| --------------------- | --------------------------------------------------------- | --------------------------------------------------------------- |
| **Log-Mel extractor** | `T × 64`                                                  | `n_fft=400`, `hop=160` (10 ms)                                  |
| **Conv stack**        | `T × 256`                                                 | 2 × Conv1d(k=3) + BatchNorm + ReLU                              |
| **Positional embed**  | `T × 256`                                                 | Learned position embeddings, capped at `max_seq_len`            |
| **Performer encoder** | 4 layers × 4 heads, `d_model = 256`, `max_seq_len = 1000` | Reversible layers, nb_features=512, dropout=0.2                 |
| **Classifier**        | `T × 1`                                                   | Linear(256→128) → LayerNorm → ReLU → Dropout(0.2) → Linear(1)   |

Loss: `BCEWithLogitsLoss(pos_weight=cfg.pos_weight)`

OneCycleLR scheduler with 5% warmup + AdamW(lr=1e-4, weight_decay=1e-4) + gradient clip 0.5 for stability.

---

### 4  Training Procedure

| Item                   | Default value                                                      |
| ---------------------- | ------------------------------------------------------------------ |
| Batch size             | 4-8 (auto-detected for GTX 1650 4GB)                               |
| Windows fed            | Max 1000 frames (10 sec at 16kHz with hop=160)                     |
| Optimiser              | AdamW, lr=1e-4, weight_decay=1e-4                                  |
| Scheduler              | OneCycleLR with 1.0 epoch warmup                                   |
| Epochs                 | 32 (early-stop patience=3 on val_loss)                             |
| Precision              | fp16-mixed on supported GPUs, fp32 otherwise                       |
| Gradient clipping      | 0.5 to prevent exploding gradients                                 |
| Windows adjustments    | num_workers=0 (multiprocessing disabled for compatibility)         |
| Metrics (torchmetrics) | F1Score, AUROC, Accuracy — logits not thresholded                  |
| Checkpointing          | Top 3 models by val_loss: `{epoch:02d}-{val_loss:.4f}-{val_f1:.4f}`|
| Reproducibility        | seed_everything(42) at entry point                                 |
| Export options         | TorchScript, ONNX, and quantized models for deployment             |
---

### 5  Evaluation Suite

1. **Frame-level metrics**
   - Precision-Recall and ROC curves with AUC calculation
   - Optimal threshold selection: 
     ```python
     precision, recall, thresholds = precision_recall_curve(frame_labels, frame_preds)
     thresholds = np.append(thresholds, 1.0)  # Fix for length mismatch
     f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
     optimal_threshold = thresholds[np.argmax(f1_scores)]
     ```
   - Threshold vs. metrics plots (Precision, Recall, F1)
   - Prediction distribution histograms

2. **Clip-level evaluation**
   - Metrics aggregated using clip means, with separate thresholds
   - ROC and PR curves specifically for clip classification
   - Prediction scatter plots by clip index

3. **Boundary detection analysis**
   - Speech segment matching with IoU > 0.5
   - Onset/offset error measurements in frames
   - Transition error patterns visualization
   - Segment-level precision/recall/F1

4. **Two-stage evaluation**
   - Validation set for threshold tuning
   - Test set for unbiased final metrics
   - Option to use fixed thresholds for consistency

5. **Visual inspection tools**
   - Confusion matrices at optimal thresholds
   - Transition error plots showing error patterns around speech boundaries
   - Summary metrics visualization for easy comparison

All results are saved as both data (JSON) and visualizations (PNG) in `evaluation_results/<run-id>/`.

---

### 6  Inference & Deployment

```bash
python inference.py \
  --checkpoint lightning_logs/version_2/checkpoints/06-0.1672-0.9295.ckpt \
  --audio my_clip.wav another_clip.flac \
  --threshold 0.5 \
  --device cuda
```

### Key Features

- **Processes multiple audio files in a single command**
- **Extracts speech regions** with configurable speech probability threshold
- **Applies intelligent region merging** with `min_speech` (0.2s) and `min_gap` (0.1s) parameters
- **Robust checkpoint loading** that handles different model versions

#### Outputs per audio file:
1. `./audio/{filename}_speech_segments.csv` with columns: `start_sec,end_sec`
2. `./audio/{filename}_speech.png` visualization with dual-panel display:
  - **Top**: Raw audio waveform with speech regions highlighted in green
  - **Bottom**: Mel spectrogram with speech regions highlighted in cyan

---

### Implementation Details

- Performs on-the-fly mel spectrogram conversion using **librosa**
- Handles both **CPU and GPU inference** with device selection
- High-quality visualization with **300 DPI** for publications
- Efficient batched processing for multiple files

---

### Deployment Options

- Run locally as **CLI tool**
- Integrates with streaming audio via **chunked processing**
- Can be packaged as **Docker container** for consistent environment
- Supports **CPU-only environments** for edge deployment

---

### 7  Experiment Tracking & CI

* **Hydra** config tree (`configs/`) records every run.
* **git tag** after each milestone (`v1.0-data`, `v1.0-model`).
* GitHub Action: lint → 5-min CPU smoke train → unit tests on labelers & eval.

---

### 8  Milestones & Expected Gains

| Milestone                      | Metric uplift vs current            | ETA    |
| ------------------------------ | ----------------------------------- | ------ |
| Clean label pipeline (Phase 3) | +12 pp real-world frame-F1          | 3 days |
| Hard-neg sampling              | –40 % FP on music                   | +½ day |
| Short-window training          | –30 % GPU hours                     | +½ day |
| Eval bug fixes                 | Metrics agree across train/val/test | +½ day |

---

### Deliverables

1. **`vad/` package** (pip-installable)
2. **`inference.py`** CLI entry point
3. **`USAGE.md`** with end-to-end recipes
4. **Pre-trained checkpoint** + sample CSV/PNG
5. **Dockerfile** for immediate deployment

That’s the full life-cycle—collect, label, augment, train, evaluate, and ship—laid out so each stage is testable and swappable without rerunning the whole pipeline.
