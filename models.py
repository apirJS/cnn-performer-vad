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

        # Add convolutional layers to capture local patterns
        # Enhanced frontend with BatchNorm
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim // 2),  # Add BatchNorm for stability
            nn.ReLU(),
            nn.Conv1d(dim // 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(dim),  # Add BatchNorm
            nn.ReLU(),
        )

        # Initialize all weights properly
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.proj = nn.Linear(dim, dim)  # Changed input dimension to match conv output
        self.norm = nn.LayerNorm(dim)

        # Add positional encoding
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)

        self.transformer = Performer(
            dim=dim,
            depth=layers,
            heads=heads,
            dim_head=dim // heads,
            causal=False,
            nb_features=512,  # Increased from 256
            feature_redraw_interval=500,
            generalized_attention=False,
            reversible=True,
            ff_chunks=1,
            use_scalenorm=False,
            use_rezero=False,
            ff_glu=True,
            ff_dropout=0.2,  # Increased dropout
            attn_dropout=0.2,  # Increased dropout
        )

        # Change classifier to frame-level prediction (no pooling)
        # More robust classifier
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
        logger.debug(f"Forward pass with input shape: {x.shape}")
        # Process with convolutional layers (transpose for conv1d)
        x = x.transpose(1, 2)  # (B,n_mels,T)
        x = self.conv_layers(x)  # (B,dim,T)
        x = x.transpose(1, 2)  # (B,T,dim)

        x = self.proj(x)  # (B,T,dim)
        x = self.norm(x)

        # Add positional encoding
        seq_len = min(x.shape[1], self.max_seq_len)
        x = x[:, :seq_len] + self.pos_embedding[:, :seq_len]

        x = self.transformer(x)  # (B,T,dim)
        # Apply classifier to each time step
        return self.clf(x).squeeze(-1)  # (B,T) - one prediction per frame


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.

    Parameters:
    - alpha: Weight for positive class (1) examples. Negative class (0)
             examples are weighted by (1-alpha). Default 0.25.
    - gamma: Focusing parameter that reduces the loss contribution from
             easy examples. Default 2.0.
    - reduction: How to reduce the loss ('mean' or 'sum'). Default 'mean'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # Probability of being correct

        # Apply alpha differently to positive and negative examples
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Complete focal loss formula
        F_loss = alpha_weight * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)


class VADLightning(pl.LightningModule):
    """
    Lightning wrapper for frame-level VAD model with BCE loss & comprehensive metrics logging.
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

        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(hp.pos_weight))

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        logger.info("Frame-level VADLightning model initialized")

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

        loss = self.loss(valid_logits, valid_targets)
        preds = torch.sigmoid(valid_logits)
        acc = ((preds > 0.5) == valid_targets.bool()).float().mean()

        # Log metrics
        self.log(f"{tag}_loss", loss, prog_bar=True, on_epoch=True)
        self.log(f"{tag}_acc", acc, prog_bar=True, on_epoch=True)

        # Use stateful metrics with valid frames only
        if tag == "train":
            self.train_f1(preds > 0.5, valid_targets.int())
            self.train_auroc(preds, valid_targets.int())
            self.log("train_f1", self.train_f1, on_epoch=True)
            self.log("train_auroc", self.train_auroc, on_epoch=True)
        else:  # validation
            self.val_f1(preds > 0.5, valid_targets.int())
            self.val_auroc(preds, valid_targets.int())
            self.log("val_f1", self.val_f1, on_epoch=True)
            self.log("val_auroc", self.val_auroc, on_epoch=True)

        logger.debug(f"{tag} metrics: loss={loss:.4f}, acc={acc:.4f}")
        return loss

    def training_step(self, b, _):
        logger.debug("Executing training step")
        return self._step(b, "train")

    def validation_step(self, b, _):
        logger.debug("Executing validation step")
        self._step(b, "val")

    def configure_optimizers(self):
        logger.info(f"Configuring optimizer with lr={self.hparams.lr}")
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )

        # Convert warmup_epochs to percentage of total steps
        total_epochs = self.trainer.max_epochs
        warmup_pct = (
            self.hparams.warmup_epochs / total_epochs if total_epochs > 0 else 0.1
        )

        # OneCycle scheduler - better convergence
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=warmup_pct,  # First 10% for warmup
                div_factor=25,  # Start with lr/25
                final_div_factor=1000,
                anneal_strategy="cos",
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": opt, "lr_scheduler": scheduler}
