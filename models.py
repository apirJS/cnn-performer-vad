#!/usr/bin/env python
# ───────────────────────────────────────────────────────────────────────
#  models.py - Model definitions for VAD
# ───────────────────────────────────────────────────────────────────────
import logging
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import F1Score, AUROC
import torch.quantization
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from torchmetrics import MetricCollection


from config import *

# Configure module logger
logger = logging.getLogger(__name__)

try:
    from performer_pytorch import Performer
except ImportError:
    import subprocess
    import sys

    logger.warning("Performer library not found, installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "performer-pytorch"])
    from performer_pytorch import Performer

    logger.info("Successfully installed performer-pytorch")


# Add this after your existing TorchScriptWrapper class or around line 150
class TorchScriptWrapper(nn.Module):
    """
    Wrapper class to make the model TorchScript and ONNX-compatible.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Forward pass without variable arguments."""
        return self.model(x)


class MelPerformer(nn.Module):
    """
    Model that performs frame-level VAD using Performer attention over mel spectrograms.
    With 2D depthwise separable convolutions to capture local time-frequency patterns.
    """

    def __init__(
        self,
        n_mels=80,
        dim=192,
        layers=4,
        heads=4,
        dim_head=None,
        max_seq_len=DEFAULT_MAX_FRAMES,
    ):
        super().__init__()
        if dim_head is None:
            dim_head = dim // heads

        logger.info(
            f"Initializing frame-level MelPerformer with n_mels={n_mels}, dim={dim}, "
            f"layers={layers}, heads={heads}, dim_head={dim_head}"
        )

        # Depthwise Separable Convolution implementation
        class DepthwiseSeparableConv2d(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, padding=1):
                super(DepthwiseSeparableConv2d, self).__init__()
                self.depthwise = nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    groups=in_channels,
                )
                self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

            def forward(self, x):
                x = self.depthwise(x)
                x = self.pointwise(x)
                return x

        self.boundary_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
        )

        # Add a GRU layer for better temporal modeling of transitions
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Create fusion layer for combining transformer and GRU outputs
        self.fusion = nn.Linear(dim * 3, dim)  # 3 = dim + 2*dim/2 (bidirectional GRU)

        # Replace Conv1d with depthwise separable Conv2d for time-frequency pattern extraction
        self.conv_layers = nn.Sequential(
            # Opsi 1: Tingkatkan input channel ke 4 untuk depthwise yang sebenarnya
            nn.Conv2d(1, 4, kernel_size=1, bias=False),  # Expander 1→4
            DepthwiseSeparableConv2d(4, dim // 2, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(),
            # Second layer tetap sama
            DepthwiseSeparableConv2d(dim // 2, dim, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )

        # Initialize all weights properly
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        # Add positional encoding
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.transformer = Performer(
            # Parameters dasar - biarkan tetap sesuai input
            dim=dim,
            depth=layers,
            heads=heads,
            dim_head=dim // heads,
            # Attention behavior
            causal=False,  # Tetap False untuk VAD yang mempertimbangkan konteks dua arah
            nb_features=384,  # Turunkan sedikit jika performa lambat
            feature_redraw_interval=1000,  # Tingkatkan untuk stabilitas lebih baik
            generalized_attention=True,  # Coba aktifkan untuk pola kompleks
            # Memory & performance optimization
            reversible=True,  # Bagus untuk sequence panjang
            ff_chunks=2,  # Tingkatkan ke 2 jika memory terbatas
            # Normalization & stabilitas
            use_scalenorm=False,  # Standard berfungsi baik
            use_rezero=False,  # Aktifkan untuk training yang lebih stabil
            # Network capacity & regularization
            ff_glu=True,  # Pertahankan untuk kapasitas ekspresi yang lebih baik
            ff_dropout=0.1,  # Tingkatkan sedikit untuk regularisasi lebih baik
            attn_dropout=0.1,  # Tingkatkan sedikit untuk regularisasi lebih baik
        )

        # Classifier remains the same
        self.clf = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 1),
        )

        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.parameters()):,} parameters"
        )

    def forward(self, x):  # x (B,T,n_mels)
        # Reshape for 2D convolution: [B,T,n_mels] -> [B,1,n_mels,T]
        batch_size, time_steps, n_mels = x.shape
        x = x.permute(0, 2, 1).contiguous()  # [B,n_mels,T]
        x = x.unsqueeze(1)  # [B,1,n_mels,T]

        # Apply 2D convolution layers
        x = self.conv_layers(x)  # [B,dim,n_mels,T]
        
        # Average over mel dimension and transpose back
        x = x.mean(dim=2)  # [B,dim,T]
        x = x.permute(0, 2, 1).contiguous()  # [B,T,dim]
        
        x = self.proj(x)  # (B,T,dim)
        x = self.norm(x)
        
        # Add positional encoding
        seq_len = min(x.shape[1], self.max_seq_len)
        x = x[:, :seq_len] + self.pos_embedding[:, :seq_len]
        
        # Store transformer outputs
        transformer_out = self.transformer(x)  # (B,T,dim)
        
        # Run GRU for better temporal modeling of transitions
        gru_out, _ = self.gru(transformer_out)  # [B,T,2*dim]
        
        # Fuse outputs for better boundary detection
        fused = torch.cat([transformer_out, gru_out], dim=-1)  # [B,T,3*dim]
        fused = self.fusion(fused)  # [B,T,dim]
        
        # Main VAD output
        vad_output = self.clf(transformer_out).squeeze(-1)  # [B,T]
        
        # Boundary detection output
        boundary_output = self.boundary_head(fused).squeeze(-1)  # [B,T]
        
        if self.training:
            return vad_output, boundary_output
        else:
            return vad_output  # Return only VAD output in inference mode


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for handling class imbalance.

    Parameters:
    - alpha: Weight for positive class (1) examples. Negative class (0)
             examples are weighted by (1-alpha). Default 0.25.
    - gamma: Focusing parameter that reduces the loss contribution from
             easy examples. Default 2.0.
    - label_smoothing: Amount of smoothing to apply to the labels. Default 0.0.
    - reduction: How to reduce the loss ('mean' or 'sum'). Default 'mean'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

        # Pre-compute smoothed values for binary labels
        if label_smoothing > 0:
            self.pos_smooth = 1.0 - label_smoothing / 2.0  # 1 -> 0.975 (for 0.05)
            self.neg_smooth = label_smoothing / 2.0  # 0 -> 0.025 (for 0.05)
        else:
            self.pos_smooth = 1.0
            self.neg_smooth = 0.0

    def forward(self, inputs, targets):
        # Apply label smoothing
        if self.label_smoothing > 0:
            # Create a new tensor for smoothed targets
            smoothed_targets = torch.zeros_like(targets)
            # Where targets are 1, use pos_smooth value; elsewhere, use neg_smooth
            smoothed_targets = torch.where(
                targets > 0.5,
                torch.ones_like(targets) * self.pos_smooth,
                torch.ones_like(targets) * self.neg_smooth,
            )
            targets = smoothed_targets

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # Probability of being correct

        # Apply alpha differently to positive and negative examples
        # Use original binary labels for determining class weights
        alpha_weight = (
            self.alpha * (targets > self.neg_smooth).float()
            + (1 - self.alpha) * (targets <= self.neg_smooth).float()
        )

        # Complete focal loss formula
        F_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


class BoundaryFocalLoss(nn.Module):
    """Enhanced Focal Loss with adaptive boundary weighting."""

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        boundary_weight=5.0,
        window_size=3,
        label_smoothing=0.05,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.window_size = window_size  # Consider more frames around boundary
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, mask=None):
        # Apply label smoothing
        smoothed_targets = torch.where(
            targets > 0.5,
            torch.ones_like(targets) * (1.0 - self.label_smoothing / 2.0),
            torch.ones_like(targets) * self.label_smoothing / 2.0,
        )

        # Find boundaries (transitions in labels)
        boundaries = torch.zeros_like(targets)

        # Handle different tensor dimensions
        if targets.dim() == 1:
            # Improved boundary detection by looking at multiple frames
            transitions = torch.zeros_like(targets)
            for i in range(1, len(targets)):
                if targets[i] != targets[i - 1]:  # Direct transition
                    transitions[
                        max(0, i - self.window_size) : min(
                            len(targets), i + self.window_size + 1
                        )
                    ] = 1.0
            boundaries = transitions
        else:
            # Case: Batched tensor with shape [batch_size, sequence_length]
            for b in range(targets.shape[0]):
                transitions = torch.zeros_like(targets[b])
                for i in range(1, targets.shape[1]):
                    if targets[b, i] != targets[b, i - 1]:  # Direct transition
                        start_idx = max(0, i - self.window_size)
                        end_idx = min(targets.shape[1], i + self.window_size + 1)
                        transitions[start_idx:end_idx] = 1.0
                boundaries[b] = transitions

        # Create a weight tensor with adaptive weighting based on confidence
        # Higher weight at boundaries, lower weight for confident predictions
        weights = torch.ones_like(targets) + boundaries * (self.boundary_weight - 1.0)

        # Calculate confidence to add additional adaptive weighting
        with torch.no_grad():
            probs = torch.sigmoid(inputs)
            confidence = (
                torch.abs(probs - 0.5) * 2
            )  # 0 for uncertain (0.5), 1 for certain (0 or 1)
            # Reduce weight for very confident predictions
            adaptive_factor = 1.0 - confidence * 0.5  # Scale from 0.5 to 1.0
            weights = (
                weights * adaptive_factor
            )  # Lower weight for confident predictions

        # Apply focal loss formula
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction="none"
        )
        pt = torch.exp(-BCE_loss)  # Probability of being correct

        # Apply alpha differently to positive and negative examples
        alpha_weight = (
            self.alpha * (targets > 0.5).float()
            + (1 - self.alpha) * (targets <= 0.5).float()
        )

        # Complete focal loss formula with boundary weighting
        F_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss * weights

        # Apply mask if provided
        if mask is not None:
            F_loss = F_loss * mask
            return (
                F_loss.sum() / mask.sum()
                if mask.sum() > 0
                else torch.tensor(0.0, device=F_loss.device)
            )
        else:
            return F_loss.mean()


class VADLightning(pl.LightningModule):
    """
    Lightning wrapper for frame-level VAD model with Focal loss & comprehensive metrics logging.
    """

    def __init__(self, hp):
        super().__init__()
        logger.info(f"Initializing frame-level VADLightning with hyperparameters: {hp}")
        # Convert dictionary to object-like access if needed
        if isinstance(hp, dict):
            from types import SimpleNamespace

            hp = SimpleNamespace(**hp)

        # Log hyperparameters
        logger.info(
            f"Initializing frame-level VADLightning with hyperparameters: {hp.__dict__ if hasattr(hp, '__dict__') else hp}"
        )

        # Store hyperparameters to enable checkpoint loading
        self.save_hyperparameters(hp.__dict__ if hasattr(hp, "__dict__") else hp)

        # Create model
        self.net = MelPerformer(
            hp.n_mels, hp.dim, hp.n_layers, hp.n_heads, max_seq_len=hp.max_frames
        )

        # Replace BCEWithLogitsLoss with FocalLoss
        # Note: FocalLoss already handles class imbalance with alpha, so pos_weight is not needed
        if hasattr(hp, "boundary_focused_loss") and hp.boundary_focused_loss:
            logger.info("Using boundary-focused loss for fine-tuning")
            boundary_weight = getattr(hp, "boundary_weight", 3.0)
            boundary_window = getattr(hp, "boundary_window", 3)
            logger.info(f"Boundary weight: {boundary_weight}, window: {boundary_window}")
            self.loss = BoundaryFocalLoss(
                alpha=0.25,
                gamma=2.0,
                boundary_weight=boundary_weight,
                window_size=boundary_window,
                label_smoothing=0.05,
            )
        else:
            self.loss = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)

        # FIX: Biarkan F1Score menggunakan threshold default 0.5
        self.train_metrics = MetricCollection(
            {"f1": BinaryF1Score(threshold=0.5), "auroc": BinaryAUROC()}
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        logger.info(
            "Frame-level VADLightning model initialized with FocalLoss (α=0.25, γ=2) + label smoothing 0.05"
        )

    def forward(self, x):
        return self.net(x)


    def _step(self, batch, tag):
        """Shared step logic for train/val."""
        x, y, mask = batch  # Includes mask for valid frames
        
        # Check if we received a completely empty/invalid batch
        if not mask.any():
            logger.warning(f"Received batch with no valid frames in {tag} step")
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log(f"{tag}_loss", dummy_loss, prog_bar=True, on_epoch=True)
            return dummy_loss
        
        # Forward pass - handle different return formats in training vs inference
        outputs = self(x)
        
        # Process outputs based on whether we're using boundary detection
        if isinstance(outputs, tuple) and len(outputs) == 2:
            # Boundary detection mode - unpack outputs
            vad_logits, boundary_logits = outputs
            using_boundary = True
        else:
            # Standard mode or inference - only VAD output
            vad_logits = outputs
            using_boundary = False
            boundary_logits = None  # Will not be used
        
        # Apply mask to get only valid frames
        valid_vad_logits = vad_logits[mask]
        valid_targets = y[mask]
        
        # Apply main VAD loss
        vad_loss = self.loss(valid_vad_logits, valid_targets)
        
        # Initialize total loss with VAD loss
        loss = vad_loss
        
        # Add boundary detection loss if we're using boundary detection
        if using_boundary and boundary_logits is not None:
            # Get valid boundary logits
            valid_boundary_logits = boundary_logits[mask]
            
            # Create boundary targets (1 at transitions, 0 elsewhere)
            valid_boundary_targets = torch.zeros_like(valid_targets)
            
            # Process batch elements separately
            offset = 0
            for b in range(x.shape[0]):
                # Count valid frames in this batch element
                n_valid = mask[b].sum().item()
                if n_valid <= 1:
                    continue
                
                # Get targets for this element
                element_targets = y[b, mask[b]]
                
                # Find transitions
                for i in range(1, n_valid):
                    idx = offset + i
                    if idx < len(valid_targets) and i-1 < len(element_targets) and i < len(element_targets):
                        if element_targets[i] != element_targets[i-1]:
                            valid_boundary_targets[idx] = 1.0
                
                offset += n_valid
            
            # Apply boundary detection loss with higher weight for positive class
            boundary_loss = F.binary_cross_entropy_with_logits(
                valid_boundary_logits,
                valid_boundary_targets,
                pos_weight=torch.tensor([5.0], device=self.device)
            )
            
            # Add weighted boundary loss
            boundary_weight = 0.5
            loss = vad_loss + boundary_weight * boundary_loss
            
            # Log additional metrics
            self.log(f"{tag}_boundary_loss", boundary_loss, prog_bar=True, on_epoch=True)
        
        # For metrics calculation and logging
        preds = torch.sigmoid(valid_vad_logits)
        binary_targets = (valid_targets > 0.5).float()
        
        # Calculate accuracy
        acc = ((preds > 0.5) == binary_targets.bool()).float().mean()
        
        # Log metrics
        self.log(f"{tag}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_vad_loss", vad_loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True)
        
        # Use stateful metrics with binary targets
        if tag == "train":
            self.train_metrics.update(preds, binary_targets)
        else:
            self.val_metrics.update(preds, binary_targets)
        
        return loss

    def training_step(self, b, _):
        logger.debug("Executing training step")
        return self._step(b, "train")

    def validation_step(self, b, _):
        logger.debug("Executing validation step")
        return self._step(b, "val")  # Add 'return' to capture the metrics


    def configure_optimizers(self):
        """Configure optimizer with layer-specific learning rates."""
        # Group parameters: higher LR for boundary-specific layers
        boundary_params = []
        encoder_params = []
        
        # Collect parameters by groups
        for name, param in self.named_parameters():
            if 'boundary_head' in name or 'gru' in name or 'fusion' in name:
                boundary_params.append(param)
            else:
                encoder_params.append(param)
        
        # Create parameter groups with different LRs
        param_groups = [
            {'params': encoder_params, 'lr': self.hparams.lr},                 # Base LR for encoder
            {'params': boundary_params, 'lr': self.hparams.lr * 5.0}  # Higher LR for boundary parts
        ]
        
        # Create optimizer
        opt = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-2,
            betas=(0.9, 0.999),
        )
        
        # Use OneCycleLR for faster convergence
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=[self.hparams.lr, self.hparams.lr * 5.0],  # Match parameter groups
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,  # Warm-up for 20% of training
                div_factor=25.0,  # Initial LR = max_lr/25
                final_div_factor=1000.0,  # Final LR = max_lr/1000
            ),
            "interval": "step",
            "frequency": 1,
        }
        
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Test step for final model evaluation."""
        mel, labels, mask = batch
        
        # Check if we received a completely empty/invalid batch
        if not mask.any():
            logger.warning(f"Received batch with no valid frames in test step")
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return {
                "test_loss": dummy_loss,
                "test_acc": torch.tensor(0.0, device=self.device),
                "test_probs": torch.tensor([0.5], device=self.device),
                "test_labels": torch.tensor([0], device=self.device),
            }
        
        # Forward pass - this will call self.forward which handles inference mode correctly
        logits = self(mel)  # Only returns VAD logits in eval/test mode
        
        # Apply loss only on valid (non-padded) frames using the mask
        valid_logits = logits[mask]
        valid_targets = labels[mask]
        
        # Calculate loss using the same loss function
        loss = self.loss(valid_logits, valid_targets)
        
        # For metrics, we need to use binary thresholding on the predictions
        preds = torch.sigmoid(valid_logits)
        
        # For accuracy, compare with original binary targets (not smoothed)
        binary_targets = (valid_targets > 0.5).float()
        acc = ((preds > 0.5) == binary_targets.bool()).float().mean()
        
        # Store for epoch-end calculation
        return {
            "test_loss": loss,
            "test_acc": acc,
            "test_probs": preds.detach().cpu(),
            "test_labels": binary_targets.detach().cpu(),
        }

    # Replace the on_test_epoch_end method:

    def on_test_epoch_end(self):
        """Calculate test metrics at epoch end."""
        # Get outputs from the trainer's state
        outputs = None

        # Try different ways to access outputs based on PyTorch Lightning version
        if hasattr(self.trainer, "test_loop"):
            if hasattr(self.trainer.test_loop, "predictions"):
                # Recent versions store in predictions
                outputs = self.trainer.test_loop.predictions
            elif hasattr(self.trainer.test_loop, "outputs"):
                # Some versions store in outputs
                outputs = self.trainer.test_loop.outputs

        # Flatten outputs if needed (for multiple dataloaders)
        if outputs and isinstance(outputs[0], list):
            outputs = [item for sublist in outputs for item in sublist]

        if not outputs:
            logger.warning("No test outputs found in trainer state")
            return

        # Aggregate outputs
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["test_acc"] for x in outputs]).mean()

        # Concatenate all predictions and labels
        all_probs = torch.cat([x["test_probs"] for x in outputs])
        all_labels = torch.cat([x["test_labels"] for x in outputs])

        # Calculate AUROC and F1
        try:
            from sklearn.metrics import roc_auc_score, f1_score

            # Convert to numpy for sklearn
            probs_np = all_probs.numpy()
            labels_np = all_labels.numpy()

            # Calculate metrics
            auroc = roc_auc_score(labels_np, probs_np)
            f1 = f1_score(labels_np, (probs_np > 0.5).astype(int))

            # Log metrics
            self.log("test_loss", avg_loss, prog_bar=True)
            self.log("test_acc", avg_acc, prog_bar=True)
            self.log("test_auroc", auroc, prog_bar=True)
            self.log("test_f1", f1, prog_bar=True)

            logger.info(
                f"Test Results: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUROC={auroc:.4f}, F1={f1:.4f}"
            )
            print(
                f"✅ Test Results: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUROC={auroc:.4f}, F1={f1:.4f}"
            )

        except Exception as e:
            logger.warning(f"Could not calculate test metrics: {e}")
            self.log("test_loss", avg_loss, prog_bar=True)
            self.log("test_acc", avg_acc, prog_bar=True)

            logger.info(f"Test Results: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
            print(f"✅ Test Results: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}")
