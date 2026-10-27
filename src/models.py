"""
Dual-Head Multi-Task Deep Neural Network Architecture
PyTorch implementation of shared trunk with joint regression and classification heads.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class DualHeadMLP(nn.Module):
    """
    Dual-Head Multi-Task Multi-Layer Perceptron.
    Simultaneously predicts:
    1. Continuous CCME Water Quality Index scalar (Head A: 0.0 to 100.0)
    2. Discrete 5-tier WQI regulatory classification logits (Head B: 5 classes)
    """

    def __init__(
        self,
        input_dim: int = 38,
        num_classes: int = 5,
        hidden_dims: Tuple[int, int, int] = (256, 128, 64),
        dropout_rates: Tuple[float, float, float] = (0.20, 0.15, 0.10),
    ):
        super(DualHeadMLP, self).__init__()

        # Shared Feature Extraction Trunk
        h1, h2, h3 = hidden_dims
        d1, d2, d3 = dropout_rates

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.GELU(),
            nn.Dropout(d1),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.GELU(),
            nn.Dropout(d2),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.GELU(),
            nn.Dropout(d3),
        )

        # Head A: Continuous Scalar Regression Head (Smooth CCME 0-100 score)
        self.regression_head = nn.Sequential(
            nn.Linear(h3, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Head B: Discrete Multi-Class Classification Head (5 WQI regulatory grades)
        self.classification_head = nn.Sequential(
            nn.Linear(h3, 32),
            nn.GELU(),
            nn.Linear(32, num_classes),  # Unnormalized logits
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            reg_out: (batch_size, 1) continuous CCME scores
            cls_out: (batch_size, 5) categorical logits
        """
        features = self.trunk(x)
        reg_out = self.regression_head(features)
        cls_out = self.classification_head(features)
        return reg_out, cls_out



class MultiTaskLoss(nn.Module):
    """
    Joint loss function combining Smooth L1 (Huber) regression loss
    and class-weighted Cross-Entropy classification loss.
    """

    def __init__(self, class_weights: torch.Tensor = None, cls_weight_factor: float = 0.5):
        super(MultiTaskLoss, self).__init__()
        self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.cls_weight_factor = cls_weight_factor

    def forward(
        self,
        pred_reg: torch.Tensor,
        pred_cls: torch.Tensor,
        target_reg: torch.Tensor,
        target_cls: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_reg = self.reg_criterion(pred_reg.squeeze(-1), target_reg)
        loss_cls = self.cls_criterion(pred_cls, target_cls)
        total_loss = loss_reg + self.cls_weight_factor * loss_cls

        metrics = {
            "total_loss": float(total_loss.item()),
            "reg_loss": float(loss_reg.item()),
            "cls_loss": float(loss_cls.item()),
        }
        return total_loss, metrics
