import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


class TverskyLoss(nn.Module):
    """Tversky loss with asymmetric FP/FN weighting.

    ``alpha`` weights false positives, ``beta`` weights false negatives. For
    thin filaments the foreground is a tiny fraction of a patch, so plain BCE
    (and to a lesser extent Dice) pushes the model to predict background and
    miss filaments. Using ``beta > alpha`` up-weights missed filaments, which
    encourages higher recall.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1.0 - tversky).mean()


class FocalTverskyLoss(TverskyLoss):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ) -> None:
        super().__init__(alpha=alpha, beta=beta, smooth=smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, self.gamma).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, gamma: float = 2.0, smooth: float = 1.0) -> None:
        super().__init__()
        self.focal_weight = focal_weight
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1.0 - probs)
        focal = (torch.pow(1.0 - pt, self.gamma) * bce).mean()

        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice_loss = 1.0 - ((2.0 * intersection + self.smooth) / (union + self.smooth)).mean()
        return self.focal_weight * focal + (1.0 - self.focal_weight) * dice_loss


def build_loss(name: str) -> nn.Module:
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "tversky":
        return TverskyLoss(alpha=0.3, beta=0.7)
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    if name == "dice_focal":
        return DiceFocalLoss(focal_weight=0.45, gamma=2.0)
    raise ValueError(f"Unknown loss: {name}")
