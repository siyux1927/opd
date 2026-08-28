"""Per-token advantage functions for on-policy distillation.

Everything is expressed through the token-level density ratio

    u = q(y_n | y_<n, x) / p_theta(y_n | y_<n, x)

with q the teacher and p_theta the student, i.e. ``log_u = teacher_logp - student_logp``.

An f-divergence between student and teacher along student-sampled prefixes is
``E_p[f(u)]``. Standard OPD (Thinking Machines, 2025) stop-gradients the reward and
therefore uses the advantage ``-f(u)``. OPD+ (arXiv:2606.01039) shows this is a biased
gradient estimator and derives the gradient-faithful weight

    w_f(u) = -f(u) + u * f'(u)

Reverse KL is the special case where the correction collapses to the constant -1, which
is why it is the only divergence that happens to work under the stop-gradient convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

LOG2 = math.log(2.0)

DIVERGENCES = ("reverse_kl", "forward_kl", "jsd")
ADVANTAGE_MODES = ("opd", "opd_plus")


def _log1p_exp(log_u: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 + u) given log_u."""
    return F.softplus(log_u)


def _log_half_one_plus_u(log_u: torch.Tensor) -> torch.Tensor:
    """log((1 + u) / 2) given log_u."""
    return _log1p_exp(log_u) - LOG2


def advantage_from_log_ratio(
    log_u: torch.Tensor,
    divergence: str,
    mode: str,
) -> torch.Tensor:
    """Per-token advantage. ``log_u`` must already be clipped by the caller.

    Table 1 of OPD+:

        divergence     OPD (stop-grad): -f(u)                       OPD+: w_f(u)
        forward_kl     -u*ln u                                      u
        reverse_kl     ln u                                         ln u - 1
        jsd            -0.5*[u*ln u - (1+u)*ln((1+u)/2)]             0.5*ln((1+u)/2)
    """
    if divergence not in DIVERGENCES:
        raise ValueError(f"unknown divergence {divergence!r}, expected one of {DIVERGENCES}")
    if mode not in ADVANTAGE_MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {ADVANTAGE_MODES}")

    if divergence == "reverse_kl":
        # f(u) = -ln u  ->  -f(u) = ln u ;  u f'(u) = -1
        return log_u if mode == "opd" else log_u - 1.0

    u = torch.exp(log_u)

    if divergence == "forward_kl":
        # f(u) = u ln u  ->  -f(u) = -u ln u ;  w = u
        return -u * log_u if mode == "opd" else u

    # jsd: f(u) = 0.5 * [u ln u - (1 + u) ln((1 + u) / 2)]
    log_mix = _log_half_one_plus_u(log_u)
    if mode == "opd":
        return -0.5 * (u * log_u - (1.0 + u) * log_mix)
    return 0.5 * log_mix


@dataclass
class RatioStats:
    """Diagnostics on the density ratio, logged every training step.

    ``clip_frac`` is the fraction of response tokens whose log-ratio saturated the clip.
    It is the single most important number to watch for forward KL, whose OPD+ advantage
    ``u`` is unbounded above.
    """

    mean_log_u: float
    reverse_kl: float
    clip_frac: float
    adv_mean: float
    adv_std: float
    adv_absmax: float


def compute_advantages(
    student_logp: torch.Tensor,
    teacher_logp: torch.Tensor,
    mask: torch.Tensor,
    divergence: str,
    mode: str,
    log_ratio_clip: float = 6.0,
) -> tuple[torch.Tensor, RatioStats, torch.Tensor]:
    """Return (advantages, stats, clipped_log_u), all masked to response tokens.

    No z-score normalisation is applied anywhere: rescaling the advantage per batch
    would erase the very difference between ``-f(u)`` and ``w_f(u)`` that this project
    measures.
    """
    raw_log_u = teacher_logp - student_logp
    log_u = raw_log_u.clamp(-log_ratio_clip, log_ratio_clip)

    adv = advantage_from_log_ratio(log_u, divergence, mode) * mask

    n = mask.sum().clamp(min=1.0)
    clipped = ((raw_log_u.abs() > log_ratio_clip).float() * mask).sum() / n
    stats = RatioStats(
        mean_log_u=float((log_u * mask).sum() / n),
        # per-token reverse KL estimate KL[p||q] = log p - log q = -log u
        reverse_kl=float((-log_u * mask).sum() / n),
        clip_frac=float(clipped),
        adv_mean=float(adv.sum() / n),
        adv_std=float(torch.sqrt((((adv - adv.sum() / n) * mask) ** 2).sum() / n)),
        adv_absmax=float((adv.abs() * mask).max()),
    )
    return adv, stats, log_u
