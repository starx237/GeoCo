"""All factual and counterfactual metrics reported for GeoCo-SAVi.

Mask coordinates use ``(x, y)`` ordering.  Geometry defaults to the common
``[-1, 1]^2`` canvas, making Obj3D and MOVi-C values directly comparable.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Mean per-example RGB peak signal-to-noise ratio."""

    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share a batched shape")
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    value = 10.0 * torch.log10(float(data_range) ** 2 / mse.clamp_min(1e-12))
    return value.mean()


def _batched_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = torch.as_tensor(labels).long()
    if labels.ndim == 2:
        labels = labels[None]
    if labels.ndim < 3:
        raise ValueError("label maps must be (H,W) or (B,...,H,W)")
    return labels.reshape(labels.shape[0], -1)


def _ari_one(
    target: torch.Tensor,
    prediction: torch.Tensor,
    ignore_label: int | None,
) -> torch.Tensor:
    if ignore_label is not None:
        keep = target != int(ignore_label)
        target = target[keep]
        prediction = prediction[keep]
    count = target.numel()
    if count < 2:
        return torch.ones((), device=target.device, dtype=torch.float64)
    _, target_inverse = torch.unique(target, return_inverse=True)
    _, prediction_inverse = torch.unique(prediction, return_inverse=True)
    prediction_count = int(prediction_inverse.max().item()) + 1
    contingency = torch.bincount(
        target_inverse * prediction_count + prediction_inverse
    ).double()
    target_marginal = torch.bincount(target_inverse).double()
    prediction_marginal = torch.bincount(prediction_inverse).double()
    choose2 = lambda value: value * (value - 1.0) * 0.5
    index = choose2(contingency).sum()
    target_index = choose2(target_marginal).sum()
    prediction_index = choose2(prediction_marginal).sum()
    total_pairs = choose2(torch.tensor(float(count), device=target.device))
    expected = target_index * prediction_index / total_pairs.clamp_min(1.0)
    maximum = 0.5 * (target_index + prediction_index)
    denominator = maximum - expected
    if float(denominator.abs()) < 1e-12:
        return torch.ones_like(denominator)
    return (index - expected) / denominator


def adjusted_rand_index(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    ignore_label: int | None = None,
) -> torch.Tensor:
    """Adjusted Rand index, averaged equally over the first dimension."""

    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    return torch.stack(
        [
            _ari_one(target[index], prediction[index], ignore_label)
            for index in range(target.shape[0])
        ]
    ).mean().float()


def fg_ari(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:
    """Foreground ARI; GT background pixels are excluded."""

    return adjusted_rand_index(
        target_labels,
        predicted_labels,
        ignore_label=background_label,
    )


def ari(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
) -> torch.Tensor:
    """Global ARI including GT background."""

    return adjusted_rand_index(target_labels, predicted_labels)


def _instance_iou_matrix(
    target: torch.Tensor,
    prediction: torch.Tensor,
    background_label: int,
) -> torch.Tensor:
    target_ids = torch.unique(target)
    target_ids = target_ids[target_ids != int(background_label)]
    prediction_ids = torch.unique(prediction)
    if target_ids.numel() == 0 or prediction_ids.numel() == 0:
        return torch.empty(
            target_ids.numel(),
            prediction_ids.numel(),
            device=target.device,
            dtype=torch.float32,
        )
    target_masks = target[None] == target_ids[:, None]
    prediction_masks = prediction[None] == prediction_ids[:, None]
    intersection = (
        target_masks[:, None] & prediction_masks[None]
    ).sum(dim=-1).float()
    union = (
        target_masks[:, None] | prediction_masks[None]
    ).sum(dim=-1).float()
    return intersection / union.clamp_min(1.0)


def _valid_mean(values: list[torch.Tensor], device) -> torch.Tensor:
    if not values:
        return torch.full((), float("nan"), device=device)
    return torch.stack(values).mean()


def hungarian_miou(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:
    """Mean GT-instance IoU after one-to-one Hungarian matching."""

    from scipy.optimize import linear_sum_assignment

    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    scores = []
    for index in range(target.shape[0]):
        matrix = _instance_iou_matrix(
            target[index], prediction[index], background_label
        )
        if matrix.shape[0] == 0:
            continue
        if matrix.shape[1] == 0:
            scores.append(matrix.new_zeros(()))
            continue
        rows, columns = linear_sum_assignment(
            -matrix.detach().cpu().numpy()
        )
        matched = matrix[
            torch.as_tensor(rows, device=matrix.device),
            torch.as_tensor(columns, device=matrix.device),
        ].sum()
        scores.append(matched / matrix.shape[0])
    return _valid_mean(scores, target.device)


def mean_best_overlap(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:
    """Mean best IoU over GT foreground instances (mBO)."""

    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    scores = []
    for index in range(target.shape[0]):
        matrix = _instance_iou_matrix(
            target[index], prediction[index], background_label
        )
        if matrix.shape[0] == 0:
            continue
        if matrix.shape[1] == 0:
            scores.append(matrix.new_zeros(()))
        else:
            scores.append(matrix.amax(dim=1).mean())
    return _valid_mean(scores, target.device)


def hard_labels(alpha: torch.Tensor) -> torch.Tensor:
    """Convert full-softmax slot alpha to an exclusive label map."""

    if alpha.ndim >= 5 and alpha.shape[-3] == 1:
        alpha = alpha.squeeze(-3)
    if alpha.ndim < 4:
        raise ValueError("alpha must have shape (...,N,H,W)")
    return alpha.argmax(dim=-3)


def hard_mask_geometry(
    masks: torch.Tensor,
    *,
    normalized: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Centroid, RMS radius, and coverage for arbitrary leading dimensions."""

    masks = torch.as_tensor(masks).float()
    if masks.ndim < 2:
        raise ValueError("masks must end in (H,W)")
    height, width = masks.shape[-2:]
    if normalized:
        y = torch.linspace(-1.0, 1.0, height, device=masks.device)
        x = torch.linspace(-1.0, 1.0, width, device=masks.device)
    else:
        y = torch.arange(height, device=masks.device, dtype=masks.dtype)
        x = torch.arange(width, device=masks.device, dtype=masks.dtype)
    yy, xx = torch.meshgrid(
        y.to(masks.dtype), x.to(masks.dtype), indexing="ij"
    )
    mass = masks.sum(dim=(-2, -1))
    valid = mass > 0.0
    centroid_x = (masks * xx).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    centroid_y = (masks * yy).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    centroid = torch.stack([centroid_x, centroid_y], dim=-1)
    distance = (
        xx - centroid_x[..., None, None]
    ).square() + (
        yy - centroid_y[..., None, None]
    ).square()
    radius = torch.sqrt(
        (masks * distance).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    )
    coverage = mass / float(height * width)
    nan = torch.full_like(radius, float("nan"))
    return {
        "centroid": torch.where(valid[..., None], centroid, float("nan")),
        "radius": torch.where(valid, radius, nan),
        "coverage": torch.where(valid, coverage, nan),
        "mass": mass,
        "valid": valid,
    }


def position_centroid_error(
    position: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Mean distance between explicit p and hard-mask centroid."""

    geometry = hard_mask_geometry(masks, normalized=True)
    position = torch.as_tensor(position, device=masks.device).float()
    if position.shape != geometry["centroid"].shape:
        raise ValueError("position and mask leading dimensions do not match")
    error = (position - geometry["centroid"]).norm(dim=-1)
    return error[geometry["valid"]].mean()


def attention_overlap(
    attention: torch.Tensor,
    *,
    active_slots: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ordered off-diagonal overlap of spatially normalized attention maps."""

    attention = torch.as_tensor(attention).float()
    if attention.ndim < 3:
        raise ValueError("attention must end in (slots,tokens)")
    slots, tokens = attention.shape[-2:]
    flat = attention.reshape(-1, slots, tokens)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if active_slots is None:
        active = torch.ones(
            flat.shape[0], slots, dtype=torch.bool, device=flat.device
        )
    else:
        active = torch.as_tensor(
            active_slots, device=flat.device, dtype=torch.bool
        ).reshape(-1, slots)
        if active.shape[0] != flat.shape[0]:
            raise ValueError("active_slots prefix dimensions do not match")
    values = []
    pairwise = torch.bmm(flat, flat.transpose(1, 2))
    diagonal = torch.eye(slots, device=flat.device, dtype=torch.bool)
    for index in range(flat.shape[0]):
        pair_mask = (
            active[index, :, None] & active[index, None, :] & ~diagonal
        )
        if bool(pair_mask.any()):
            values.append(pairwise[index][pair_mask].mean())
    return _valid_mean(values, flat.device)


def translation_error(
    factual_masks: torch.Tensor,
    edited_masks: torch.Tensor,
    delta_pixels: torch.Tensor,
    *,
    command_canvas_size: int = 64,
) -> torch.Tensor:
    """Normalized centroid-command error in a common square command canvas.

    Masks may be measured at any resolution (for example, MOVi-C's native GT
    resolution). ``delta_pixels`` is expressed on ``command_canvas_size``;
    the default 64-pixel canvas is shared by the reported Obj3D and MOVi-C
    protocols.
    """

    command_canvas_size = int(command_canvas_size)
    if command_canvas_size < 2:
        raise ValueError("command_canvas_size must be at least 2")
    factual = hard_mask_geometry(factual_masks, normalized=True)
    edited = hard_mask_geometry(edited_masks, normalized=True)
    delta = torch.as_tensor(delta_pixels, device=factual_masks.device).float()
    if delta.shape != factual["centroid"].shape:
        raise ValueError("delta_pixels must match mask leading dimensions")
    observed_pixels = (
        edited["centroid"] - factual["centroid"]
    ) * ((command_canvas_size - 1) / 2.0)
    residual = observed_pixels - delta
    valid = factual["valid"] & edited["valid"]
    diagonal = math.sqrt(2.0 * command_canvas_size * command_canvas_size)
    return residual.norm(dim=-1)[valid].mean() / diagonal


def appearance_transplant_centroid_drift(
    factual_masks: torch.Tensor,
    transplanted_masks: torch.Tensor,
) -> torch.Tensor:
    """Normalized centroid drift after changing appearance only."""

    factual = hard_mask_geometry(factual_masks, normalized=True)
    edited = hard_mask_geometry(transplanted_masks, normalized=True)
    valid = factual["valid"] & edited["valid"]
    return (
        edited["centroid"] - factual["centroid"]
    ).norm(dim=-1)[valid].mean()


def fixed_scale_consistency(
    masks: torch.Tensor,
    *,
    appearance_dimension: int = -3,
) -> dict[str, torch.Tensor]:
    """Within-group size variation while p and s remain fixed."""

    geometry = hard_mask_geometry(masks, normalized=True)
    dimension = (
        appearance_dimension
        if appearance_dimension >= 0
        else geometry["radius"].ndim + appearance_dimension + 2
    )
    # For the conventional (groups, appearances, H, W) input, radius has
    # shape (groups, appearances) and the appearance dimension is the last.
    if masks.ndim == 4 and appearance_dimension == -3:
        dimension = -1
    radius_std = torch.nan_to_num(
        geometry["radius"].std(dim=dimension, unbiased=False),
        nan=float("nan"),
    )
    coverage_std = torch.nan_to_num(
        geometry["coverage"].std(dim=dimension, unbiased=False),
        nan=float("nan"),
    )
    return {
        "radius_std": torch.nanmean(radius_std),
        "coverage_std": torch.nanmean(coverage_std),
    }


def fixed_s_radius_std(masks: torch.Tensor) -> torch.Tensor:
    return fixed_scale_consistency(masks)["radius_std"]


def fixed_s_coverage_std(masks: torch.Tensor) -> torch.Tensor:
    return fixed_scale_consistency(masks)["coverage_std"]


def scale_sweep_metrics(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Log-log slopes and monotonicity for a fixed ordered scale sweep.

    ``masks`` has shape ``(groups, scales, H, W)``.  All groups use the same
    strictly increasing positive scale factors.
    """

    masks = torch.as_tensor(masks).float()
    if masks.ndim != 4:
        raise ValueError("masks must have shape (groups,scales,H,W)")
    factors = torch.as_tensor(
        list(scale_factors) if not isinstance(scale_factors, torch.Tensor) else scale_factors,
        device=masks.device,
        dtype=masks.dtype,
    )
    if factors.ndim != 1 or factors.numel() != masks.shape[1]:
        raise ValueError("one scale factor is required per sweep mask")
    if bool((factors <= 0).any()) or bool((factors[1:] <= factors[:-1]).any()):
        raise ValueError("scale factors must be positive and strictly increasing")
    geometry = hard_mask_geometry(masks, normalized=True)
    radius = geometry["radius"]
    coverage = geometry["coverage"]
    x = factors.log()
    centered_x = x - x.mean()
    denominator = centered_x.square().sum().clamp_min(1e-12)

    def slope(values: torch.Tensor) -> torch.Tensor:
        y = values.clamp_min(1e-12).log()
        return (
            (y - y.mean(dim=1, keepdim=True)) * centered_x
        ).sum(dim=1) / denominator

    valid = geometry["valid"].all(dim=1)
    radius_slopes = slope(radius)
    coverage_slopes = slope(coverage)
    radius_monotonic = (radius[:, 1:] >= radius[:, :-1]).all(dim=1)
    coverage_monotonic = (coverage[:, 1:] >= coverage[:, :-1]).all(dim=1)
    return {
        "radius_slope": radius_slopes[valid].mean(),
        "coverage_slope": coverage_slopes[valid].mean(),
        "radius_monotonic": radius_monotonic[valid].float().mean(),
        "coverage_monotonic": coverage_monotonic[valid].float().mean(),
    }


def radius_slope(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["radius_slope"]


def coverage_slope(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["coverage_slope"]


def radius_monotonicity(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["radius_monotonic"]


def coverage_monotonicity(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["coverage_monotonic"]
