"""Verify the advantage table against autograd, so a sign error cannot slip through.

Run with: python -m pytest tests -q   (CPU only, no GPU or model download needed)
"""

from __future__ import annotations

import math

import torch

from opd.divergences import DIVERGENCES, advantage_from_log_ratio, compute_advantages


def f_of_u(u: torch.Tensor, divergence: str) -> torch.Tensor:
    """The convex generator f, with f(1) = 0, for each divergence."""
    if divergence == "reverse_kl":
        return -torch.log(u)
    if divergence == "forward_kl":
        return u * torch.log(u)
    if divergence == "jsd":
        return 0.5 * (u * torch.log(u) - (1.0 + u) * torch.log((1.0 + u) / 2.0))
    raise ValueError(divergence)


def test_generators_vanish_at_one():
    one = torch.ones(1, dtype=torch.float64)
    for divergence in DIVERGENCES:
        assert torch.allclose(f_of_u(one, divergence), torch.zeros(1, dtype=torch.float64), atol=1e-12)


def test_opd_advantage_is_negative_f():
    log_u = torch.linspace(-4, 4, 200, dtype=torch.float64)
    u = torch.exp(log_u)
    for divergence in DIVERGENCES:
        got = advantage_from_log_ratio(log_u, divergence, "opd")
        assert torch.allclose(got, -f_of_u(u, divergence), atol=1e-9), divergence


def test_opd_plus_matches_autograd_correction():
    """w_f(u) = -f(u) + u f'(u), with f' taken by autograd rather than by hand."""
    log_u = torch.linspace(-4, 4, 200, dtype=torch.float64)
    u = torch.exp(log_u).requires_grad_(True)
    for divergence in DIVERGENCES:
        f = f_of_u(u, divergence)
        (f_prime,) = torch.autograd.grad(f.sum(), u, retain_graph=False)
        expected = -f_of_u(u.detach(), divergence) + u.detach() * f_prime
        got = advantage_from_log_ratio(log_u, divergence, "opd_plus")
        assert torch.allclose(got, expected, atol=1e-8), divergence


def test_reverse_kl_correction_is_a_constant_baseline():
    """The whole point of the paper: for reverse KL the correction is exactly -1."""
    log_u = torch.linspace(-5, 5, 100, dtype=torch.float64)
    delta = advantage_from_log_ratio(log_u, "reverse_kl", "opd_plus") - advantage_from_log_ratio(
        log_u, "reverse_kl", "opd"
    )
    assert torch.allclose(delta, torch.full_like(delta, -1.0), atol=1e-12)


def test_other_divergences_have_non_constant_corrections():
    log_u = torch.linspace(-5, 5, 100, dtype=torch.float64)
    for divergence in ("forward_kl", "jsd"):
        delta = advantage_from_log_ratio(log_u, divergence, "opd_plus") - advantage_from_log_ratio(
            log_u, divergence, "opd"
        )
        assert delta.std() > 1e-3, divergence


def test_forward_kl_stopgrad_advantage_vanishes_where_student_is_wrong():
    """Diagnosis of the collapse: -u ln u -> 0 as u -> 0.

    Tokens the teacher wants but the student almost never emits produce ~no learning
    signal under stop-gradient forward KL, while the corrected advantage u stays a
    strictly positive, monotone push.
    """
    log_u = torch.tensor([-8.0, -6.0, -4.0], dtype=torch.float64)
    opd = advantage_from_log_ratio(log_u, "forward_kl", "opd")
    plus = advantage_from_log_ratio(log_u, "forward_kl", "opd_plus")
    assert (opd.abs() < 0.08).all()
    assert (plus > 0).all()
    assert torch.all(plus[1:] > plus[:-1])


def test_compute_advantages_masks_and_reports_clipping():
    student = torch.tensor([[-1.0, -2.0, -3.0]])
    teacher = torch.tensor([[-1.0, -20.0, -3.0]])  # middle token saturates the clip
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, stats, log_u = compute_advantages(
        student, teacher, mask, divergence="reverse_kl", mode="opd", log_ratio_clip=6.0
    )
    assert adv[0, 2] == 0.0
    assert math.isclose(stats.clip_frac, 0.5, abs_tol=1e-6)
    assert math.isclose(float(log_u[0, 1]), -6.0, abs_tol=1e-6)
    assert math.isclose(stats.reverse_kl, (0.0 + 6.0) / 2, abs_tol=1e-6)


def test_no_implicit_normalisation():
    """Advantages must not be z-scored: that would erase the OPD vs OPD+ difference."""
    student = torch.full((2, 5), -3.0)
    teacher = torch.full((2, 5), -1.0)
    mask = torch.ones(2, 5)
    base, _, _ = compute_advantages(student, teacher, mask, divergence="reverse_kl", mode="opd")
    plus, _, _ = compute_advantages(student, teacher, mask, divergence="reverse_kl", mode="opd_plus")
    assert torch.allclose(base - plus, torch.ones_like(base))
