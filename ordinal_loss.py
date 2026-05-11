"""Distance-aware multiclass losses for the 0-5 sentiment score task."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentiment_scale import NUM_SENTIMENT_CLASSES


@dataclass(frozen=True)
class OrdinalLossConfig:
    """Configuration for the distance-aware multiclass objective."""

    ce_weight: float = 1.0
    distance_weight: float = 0.35
    label_smoothing: float = 0.05
    pseudo_label_smoothing: float = 0.02
    focal_gamma: float = 1.5
    logit_adjustment_weight: float = 0.0


class DistanceAwareOrdinalLoss(nn.Module):
    """Multiclass softmax objective for the 0-5 sentiment score task.

    Components:
    - class-balanced focal CrossEntropy/soft CE for 6 score classes;
    - expected-score SmoothL1 for distance-aware scoring.
    """

    def __init__(
        self,
        *,
        class_weights: torch.Tensor | None = None,
        config: OrdinalLossConfig | None = None,
        num_classes: int = NUM_SENTIMENT_CLASSES,
    ) -> None:
        super().__init__()
        self.config = config or OrdinalLossConfig()
        self.num_classes = num_classes
        self.register_buffer(
            "score_values",
            torch.arange(num_classes, dtype=torch.float32),
            persistent=False,
        )
        if class_weights is None:
            self.register_buffer("class_weights", None, persistent=False)
        else:
            self.register_buffer("class_weights", class_weights.float(), persistent=False)
        self.register_buffer("logit_adjustment", None, persistent=False)

    def set_logit_adjustment(self, class_priors: torch.Tensor | None) -> None:
        if class_priors is None:
            self.logit_adjustment = None
            return
        priors = class_priors.float().clamp_min(1e-8)
        priors = priors / priors.sum().clamp_min(1e-8)
        self.logit_adjustment = priors.log()

    def forward(
        self,
        logits: torch.Tensor | dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        soft_labels: torch.Tensor | None = None,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if isinstance(logits, dict):
            outputs = logits
            logits = outputs["logits"]

        labels = labels.long()
        target_probs = self._build_targets(labels, soft_labels)
        adjusted_logits = self._adjust_logits(logits)

        probs = torch.softmax(adjusted_logits, dim=1)
        if soft_labels is None:
            ce_per_sample = F.cross_entropy(
                adjusted_logits,
                labels,
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
                reduction="none",
                label_smoothing=max(0.0, float(self.config.label_smoothing)),
            )
        else:
            log_probs = F.log_softmax(adjusted_logits, dim=1)
            ce_per_sample = -(target_probs * log_probs).sum(dim=1)
            if self.class_weights is not None:
                class_weight_per_sample = (target_probs * self.class_weights.to(logits.device)).sum(dim=1)
                ce_per_sample = ce_per_sample * class_weight_per_sample
        if self.config.focal_gamma > 0:
            target_confidence = (target_probs * probs).sum(dim=1).clamp(1e-6, 1.0)
            ce_per_sample = ce_per_sample * (1.0 - target_confidence).pow(self.config.focal_gamma)

        score_values = self.score_values.to(logits.device)
        predicted_score = (probs * score_values).sum(dim=1)
        target_score = (target_probs * score_values).sum(dim=1)
        distance_per_sample = F.smooth_l1_loss(predicted_score, target_score, reduction="none")

        loss_per_sample = (
            self.config.ce_weight * ce_per_sample
            + self.config.distance_weight * distance_per_sample
        )

        if sample_weights is not None:
            loss_per_sample = loss_per_sample * sample_weights.to(logits.device).float()
        return loss_per_sample.mean()

    def _adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.logit_adjustment is None or self.config.logit_adjustment_weight <= 0:
            return logits
        adjustment = self.logit_adjustment.to(logits.device)
        return logits - self.config.logit_adjustment_weight * adjustment

    def _build_targets(
        self,
        labels: torch.Tensor,
        soft_labels: torch.Tensor | None,
    ) -> torch.Tensor:
        if soft_labels is not None:
            target = soft_labels.to(labels.device).float()
            smoothing = self.config.pseudo_label_smoothing
        else:
            target = F.one_hot(labels, num_classes=self.num_classes).float()
            smoothing = self.config.label_smoothing

        if smoothing > 0:
            uniform = torch.full_like(target, 1.0 / self.num_classes)
            target = target * (1.0 - smoothing) + uniform * smoothing
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return target


def effective_number_class_weights(
    class_counts: list[int],
    *,
    beta: float = 0.9999,
    min_weight: float = 0.05,
) -> torch.Tensor:
    """Compute class-balanced weights from effective number of samples."""
    counts = torch.tensor([max(0, int(count)) for count in class_counts], dtype=torch.float32)
    present = counts > 0
    weights = torch.zeros_like(counts)
    if not bool(present.any()):
        return torch.ones_like(counts)

    beta = min(max(float(beta), 0.0), 0.999999)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts[present])
    weights[present] = (1.0 - beta) / effective_num.clamp_min(1e-8)
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-8)
    if min_weight > 0:
        weights[present] = weights[present].clamp_min(min_weight)
    return weights
