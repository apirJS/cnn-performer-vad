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


class MelPerformer(nn.Module):
    """
    Model that performs frame-level VAD using Performer attention over mel spectrograms.
    With 2D depthwise separable convolutions to capture local time-frequency patterns.
    """

    def __init__(
        self,
        n_mels=64,
        dim=256,
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

        x = self.transformer(x)  # (B,T,dim)
        # Apply classifier to each time step
        return self.clf(x).squeeze(-1)  # (B,T) - one prediction per frame


class TorchScriptWrapper(nn.Module):
    """
    Wrapper class to make the model TorchScript-compatible by
    removing variable arguments from forward methods.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Forward pass without variable arguments."""
        return self.model(x)


# Then update the export_model function to use this wrapper:
# No need to modify this file if using from train.py


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
    """Focal Loss with enhanced weighting for speech boundaries."""

    def __init__(
        self, alpha=0.25, gamma=2.0, boundary_weight=2.0, label_smoothing=0.05
    ):
        super(BoundaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
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
            # Case: Flattened tensor (validation with mask)
            # Calculate transitions in the flattened sequence
            transitions = torch.abs(torch.cat([targets[0:1], targets[:-1]]) - targets)
            boundaries = transitions
        else:
            # Case: Batched tensor with shape [batch_size, sequence_length]
            for b in range(targets.shape[0]):
                transitions = torch.abs(targets[b, 1:] - targets[b, :-1])
                pad = torch.zeros(1, device=targets.device)
                transitions = torch.cat([pad, transitions], dim=0)
                boundaries[b] = transitions

        # Create a weight tensor - higher weight at boundaries
        weights = torch.ones_like(targets) + boundaries * (self.boundary_weight - 1.0)

        # Apply focal loss formula
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)  # Probability of being correct

        # Apply alpha differently to positive and negative examples
        alpha_weight = (
            self.alpha * (targets > 0.5).float()
            + (1 - self.alpha) * (targets <= 0.5).float()
        )

        # Complete focal loss formula with boundary weighting
        F_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss * weights

        # Apply mask if provided (which is different from the mask already applied in _step)
        if mask is not None:
            F_loss = F_loss * mask
            return F_loss.sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0, device=F_loss.device)
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
        if hasattr(hp, 'boundary_focused_loss') and hp.boundary_focused_loss:
            logger.info("Using boundary-focused loss for fine-tuning")
            self.loss = BoundaryFocalLoss(
                alpha=0.25,
                gamma=2.0,
                boundary_weight=3.0,  # Higher weight on boundary frames
                label_smoothing=0.05
            )
        else:
            self.loss = FocalLoss(
                alpha=0.25,
                gamma=2.0,
                label_smoothing=0.05
            )

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
        x, y, mask = batch  # Now includes mask for valid frames

        # Check if we received a completely empty/invalid batch
        if not mask.any():
            logger.warning(f"Received batch with no valid frames in {tag} step")
            # Return a zero loss tensor that can be backpropagated
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            self.log(f"{tag}_loss", dummy_loss, prog_bar=True, on_epoch=True)
            return dummy_loss

        logger.debug(
            f"{tag} step with batch shapes: x={x.shape}, y={y.shape}, mask={mask.shape}"
        )

        logits = self(x)  # (B,T) - per frame logits

        # Apply loss only on valid (non-padded) frames using the mask
        valid_logits = logits[mask]
        valid_targets = y[mask]

        # Apply loss (FocalLoss handles label smoothing internally)
        loss = self.loss(valid_logits, valid_targets)

        # For metrics, we need to use binary thresholding on the predictions
        preds = torch.sigmoid(valid_logits)

        # For accuracy, compare with original binary targets (not smoothed)
        binary_targets = (valid_targets > 0.5).float()
        acc = ((preds > 0.5) == binary_targets.bool()).float().mean()

        # Log metrics
        self.log(f"{tag}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True)

        # Use stateful metrics with binary targets for consistent evaluation
        if tag == "train":
            self.train_metrics.update(preds, binary_targets)
        else:
            self.val_metrics.update(preds, binary_targets)

        logger.debug(f"{tag} metrics: loss={loss:.4f}, acc={acc:.4f}")
        return loss

    def training_step(self, b, _):
        logger.debug("Executing training step")
        return self._step(b, "train")

    def validation_step(self, b, _):
        logger.debug("Executing validation step")
        self._step(b, "val")

    def configure_optimizers(self):
        """
        Configure optimizer with Cosine Annealing with Warm Restarts
        Base LR: 1e-4, Min LR: 1e-6
        """
        logger.info(f"Configuring optimizer with CosineAnnealingWarmRestarts scheduler")

        # Buat optimizer
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  # Base learning rate (1e-4)
            weight_decay=1e-2,  # L2 regularization
            betas=(0.9, 0.999),  # Default AdamW betas
        )

        # Hitung parameter berdasarkan jumlah epoch total
        total_epochs = self.trainer.max_epochs

        # Pilih T_0 yang sesuai berdasarkan ukuran dataset dan jumlah epoch
        # Untuk dataset ~35k (20k pos), 5-7 epoch adalah titik awal yang baik
        # T_0 = max(5, min(7, total_epochs // 4))

        # Pilih T_mult berdasarkan total epochs yang direncanakan
        # Untuk 32 epoch, T_mult=2 memberikan siklus: 5, 10, 20, ...
        # T_mult = 2

        # To continue from specific epoch wihtout reset
        T_0 = 25
        T_mult = 1

        logger.info(f"CosineAnnealingWarmRestarts config: T_0={T_0}, T_mult={T_mult}")
        logger.info(f"Learning rates: max={self.hparams.lr}, min=1e-6")

        # Setup scheduler dengan warm restarts
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ConstantLR(
                optimizer=opt,
                factor=0.1,  # Use 10% of your base learning rate
                total_iters=0,  # Apply immediately
            ),
            "interval": "epoch",
        }
        # scheduler = {
        #     "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #         optimizer=opt,
        #         T_0=T_0,  # Panjang siklus pertama (epoch)
        #         T_mult=T_mult,  # Faktor multiplikasi untuk panjang siklus
        #         eta_min=1e-6,  # Minimum learning rate
        #         last_epoch=-1,
        #     ),
        #     "interval": "epoch",  # Update per epoch
        #     "frequency": 1,  # Update setiap epoch
        #     "name": "cosine_lr",  # Nama untuk logging
        # }

        return {"optimizer": opt, "lr_scheduler": scheduler}

    # def configure_optimizers(self): FOR TRAINING FROM SCRATCH
    #     opt = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=5e-4,  # Slightly higher initial LR
    #         weight_decay=1e-2,
    #         betas=(0.9, 0.999),
    #     )
        
    #     # Use CosineAnnealingWarmRestarts with special settings
    #     scheduler = {
    #         "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #             optimizer=opt,
    #             T_0=3,          # First restart after 3 epochs
    #             T_mult=2,        # Double period after each restart
    #             eta_min=1e-6,    # Minimum LR
    #         ),
    #         "interval": "epoch",
    #         "name": "cosine_restart_lr",
    #     }
        
    #     return {"optimizer": opt, "lr_scheduler": scheduler}

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute(), prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    # Replace the test_step and on_test_epoch_end methods around line 380-443:

    def test_step(self, batch, batch_idx):
        """Test step for final model evaluation."""
        # Reuse the same logic as _step method
        mel, labels, mask = batch

        # Check if we received a completely empty/invalid batch
        if not mask.any():
            logger.warning(f"Received batch with no valid frames in test step")
            # Return a zero loss tensor that can be backpropagated
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return {
                "test_loss": dummy_loss,
                "test_acc": torch.tensor(0.0, device=self.device),
                "test_probs": torch.tensor([0.5], device=self.device),
                "test_labels": torch.tensor([0], device=self.device),
            }

        # Forward pass
        logits = self.net(mel)  # (B,T) - per frame logits

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
