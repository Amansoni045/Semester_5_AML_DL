"""
Unit tests for DualHeadMLP PyTorch architecture and MultiTaskLoss.
"""

import pytest
import torch
from src.models import DualHeadMLP, MultiTaskLoss


def test_dual_head_mlp_shapes():
    batch_size = 16
    input_dim = 38
    num_classes = 5

    model = DualHeadMLP(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)

    reg_out, cls_out = model(x)

    assert reg_out.shape == (batch_size, 1), f"Expected reg shape {(batch_size, 1)}, got {reg_out.shape}"
    assert cls_out.shape == (batch_size, num_classes), f"Expected cls shape {(batch_size, num_classes)}, got {cls_out.shape}"


def test_dual_head_mlp_checkpoint_loading():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    ckpt_path = root / "data" / "GeneratedOutput" / "03.ModelTraining" / "deep_learning_mlp_model.pt"

    if ckpt_path.exists():
        model = DualHeadMLP(input_dim=38, num_classes=5)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        x = torch.randn(4, 38)
        reg_out, cls_out = model(x)
        assert reg_out.shape == (4, 1)
        assert cls_out.shape == (4, 5)



def test_multi_task_loss():
    batch_size = 8
    loss_fn = MultiTaskLoss(cls_weight_factor=0.5)

    pred_reg = torch.tensor([[85.0], [70.0], [92.0], [40.0], [60.0], [95.0], [80.0], [50.0]])
    target_reg = torch.tensor([88.0, 68.0, 90.0, 42.0, 62.0, 94.0, 81.0, 48.0])

    pred_cls = torch.randn(batch_size, 5)
    target_cls = torch.tensor([3, 2, 4, 0, 1, 4, 3, 1], dtype=torch.long)

    total_loss, metrics = loss_fn(pred_reg, pred_cls, target_reg, target_cls)

    assert total_loss > 0.0
    assert "total_loss" in metrics
    assert "reg_loss" in metrics
    assert "cls_loss" in metrics
    assert metrics["reg_loss"] >= 0.0
    assert metrics["cls_loss"] >= 0.0
