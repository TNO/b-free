"""
Tests for the estimators.

Two of these guard against specific errors that reached a manuscript draft and
were caught late: the difference statistics measured from the wrong origin, and
the Bayesian prior being the binomial one. Both are cheap to check and neither
announces itself in the numbers.
"""

import numpy as np
import pytest

from bfree import (
    LOG10,
    b_aki,
    b_assembled,
    b_grouped,
    b_ogata,
    bayesian_survival_rounded,
    bfree_survival_rounded,
    difference_statistics_rounded,
    gr_survival,
    magnitude_statistics_rounded,
)

# The worked example in the docstrings of
# seismostats/analysis/bvalue/{classic,utsu}.py at commit 3ab5679, with the
# b-values those docstrings report. An independent implementation of the same
# two estimators.
SEISMOSTATS_MAGNITUDES = np.array([
    2.0, 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3, 2.7, 2.2, 2.4, 2.0,
    2.7, 2.2, 2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5,
])
SEISMOSTATS_MC = 2.0
SEISMOSTATS_DELTA_M = 0.1
SEISMOSTATS_CLASSIC = 1.114920128810535
SEISMOSTATS_UTSU = 1.1088369750721319


def test_matches_seismostats():
    delta = SEISMOSTATS_DELTA_M / 2
    n, t, _ = magnitude_statistics_rounded(SEISMOSTATS_MAGNITUDES, SEISMOSTATS_MC, delta)
    assert b_grouped(t, n, delta) == pytest.approx(SEISMOSTATS_CLASSIC, rel=1e-14)
    assert b_aki(t, n) == pytest.approx(SEISMOSTATS_UTSU, rel=1e-14)


def test_assembled_is_grouped_times_finite_size_factor():
    """b_assembled : b_grouped :: b_ogata : b_aki, with the same (n-1)/n."""
    delta = 0.05
    # T is a sum over n events of an excess that is at least delta, so t has to
    # scale with n; a fixed t makes the mean excess negative at large n and the
    # grouped estimator correctly returns nan.
    for n in (2, 20, 100, 1000):
        t = n * np.array([0.2, 0.5, 1.3])
        assert np.all(np.isfinite(b_grouped(t, n, delta)))
        assert b_assembled(t, n, delta) == pytest.approx(
            (n - 1) / n * b_grouped(t, n, delta), rel=1e-15)
        assert b_ogata(t, n) == pytest.approx((n - 1) / n * b_aki(t, n), rel=1e-15)
    with pytest.raises(ValueError):
        b_assembled(np.array([10.0]), 1, delta)


def test_magnitude_statistics_origin():
    """T is the excess above the bottom of the catalogue, m0 - delta."""
    m = np.array([2.0, 2.2, 2.4])
    n, t, w = magnitude_statistics_rounded(m, 2.0, 0.05)
    assert n == 3
    assert t == pytest.approx(np.sum(m - 1.95))
    assert w == 0 + 2 + 4


def test_magnitude_statistics_rejects_off_rounded():
    with pytest.raises(ValueError, match="lattice"):
        magnitude_statistics_rounded(np.array([2.0, 2.13]), 2.0, 0.05)


def test_magnitude_statistics_rejects_zero_delta():
    with pytest.raises(ValueError, match="delta must be positive"):
        magnitude_statistics_rounded(np.array([2.0, 2.2]), 2.0, 0.0)


def test_difference_statistics_measured_from_the_threshold():
    """Both T and W are excesses above k_min, not above zero.

    The two agree at k_min = 1, which is why measuring from zero survived
    undetected for as long as the threshold was not tunable.
    """
    delta = 0.05
    m = 2.0 + 2 * delta * np.array([0, 3, 1, 6, 2, 2])
    k_min = 2
    n_pos, t, w = difference_statistics_rounded(m, 2.0, delta, k_min=k_min)

    retained = np.array([3, 5])  # bin differences 3-0, 6-1; 2-2 = 0 is dropped
    assert n_pos == len(retained)
    assert w == int(np.sum(retained - k_min))
    assert t == pytest.approx(np.sum((retained - k_min) * 2 * delta + delta))


def test_b_ogata_rejects_n_below_two():
    with pytest.raises(ValueError, match="n >= 2"):
        b_ogata(1.0, 1)


def test_b_ogata_guard_is_vectorised():
    """One out-of-domain element aborts the whole batch; callers must mask."""
    with pytest.raises(ValueError, match="n >= 2"):
        b_ogata(np.array([1.0, 1.0]), np.array([5, 1]))


def test_gr_survival():
    assert gr_survival(2.0, 1.0, 2.0) == pytest.approx(1.0)
    assert gr_survival(3.0, 1.0, 2.0) == pytest.approx(0.1)
    assert gr_survival(4.0, 0.5, 2.0) == pytest.approx(0.1)


def test_bfree_survival_is_one_at_the_reference():
    sf = bfree_survival_rounded(w=40, n=10, k=0)
    assert sf.shape == (1, 1)
    assert sf[0, 0] == pytest.approx(1.0)


def test_bfree_survival_vanishes_above_its_support():
    """Support is [0, w+1); above it the estimator is zero, not negative."""
    sf = bfree_survival_rounded(w=5, n=3, k=[5.5, 6.0, 7.0])
    assert sf[0, 0] > 0.0
    assert sf[0, 1] == 0.0
    assert sf[0, 2] == 0.0


def test_bfree_survival_approaches_the_continuous_form():
    """At large w the lattice estimator tends to (1 - k/w)^(n-1).

    The gamma ratio leaves a relative error of order n(n-1)/2 / (w - k), so w
    has to be large against the distance to the top of the support, not just
    against n; at w = 2e5 the k = 0.9w point is still 1.3e-3 away.
    """
    n, w = 8, 2_000_000
    k = np.array([0.1, 0.5, 0.9]) * w
    sf = bfree_survival_rounded(w, n, k)
    assert sf.shape == (1, len(k))
    assert sf[0] == pytest.approx((1.0 - k / w) ** (n - 1), rel=1e-3)


def test_bfree_survival_broadcasts_over_catalogues():
    sf = bfree_survival_rounded(w=np.array([10, 20, 30]), n=np.array([5, 5, 5]), k=[0, 1, 2])
    assert sf.shape == (3, 3)
    assert np.all(np.diff(sf, axis=1) < 0.0)


def test_bayesian_prior_is_the_negative_binomial_jeffreys():
    """Beta(1/2, 0), not the binomial Beta(1/2, 1/2).

    n0 = 0 and n0 = 1/2 differ by a few percent in the tail, which is small
    enough to look like a modelling choice rather than an error.
    """
    from inspect import signature

    defaults = signature(bayesian_survival_rounded).parameters
    assert defaults["w0"].default == 0.5
    assert defaults["n0"].default == 0.0

    correct = bayesian_survival_rounded(w=40, n=10, k=40)
    binomial = bayesian_survival_rounded(w=40, n=10, k=40, n0=0.5)
    assert correct[0, 0] != pytest.approx(binomial[0, 0], rel=1e-3)


def test_bayesian_survival_is_one_at_the_reference():
    assert bayesian_survival_rounded(w=40, n=10, k=0)[0, 0] == pytest.approx(1.0)


def test_log10_constant():
    assert pytest.approx(np.log(10.0)) == LOG10
