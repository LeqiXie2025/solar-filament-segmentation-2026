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


class IoUBCELoss(nn.Module):
    """Soft Jaccard objective aligned with the official IoU-based PQ."""

    def __init__(self, bce_weight: float = 0.35, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims) - intersection
        iou_loss = 1.0 - ((intersection + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * iou_loss


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


class BoundaryDiceFocalLoss(nn.Module):
    """Dice/focal region loss plus a differentiable thin-boundary objective."""

    def __init__(self, boundary_weight: float = 0.2) -> None:
        super().__init__()
        self.region = DiceFocalLoss(focal_weight=0.45, gamma=2.0)
        self.boundary_weight = boundary_weight

    @staticmethod
    def boundary_map(values: torch.Tensor) -> torch.Tensor:
        dilated = F.max_pool2d(values, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-values, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        region_loss = self.region(logits, targets)
        pred_boundary = self.boundary_map(torch.sigmoid(logits))
        target_boundary = self.boundary_map(targets)
        boundary_loss = F.l1_loss(pred_boundary, target_boundary)
        return region_loss + self.boundary_weight * boundary_loss


def build_loss(name: str) -> nn.Module:
    if name == "dice_bce":
        return DiceBCELoss()
    if name == "iou_bce":
        return IoUBCELoss()
    if name == "tversky":
        return TverskyLoss(alpha=0.3, beta=0.7)
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33)
    if name == "dice_focal":
        return DiceFocalLoss(focal_weight=0.45, gamma=2.0)
    if name == "boundary_dice_focal":
        return BoundaryDiceFocalLoss(boundary_weight=0.2)
    raise ValueError(f"Unknown loss: {name}")
