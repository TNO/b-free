"""
Estimators for Gutenberg-Richter earthquake size distributions.

Implements the estimators presented in "b-free: Inferring Gutenberg-Richter
earthquake size distributions without estimating b-values".  Docstrings name
the estimators rather than cite equation numbers, which move as the manuscript
is revised.

Two families are provided:

  * plug-in estimators, which first estimate a b-value and then substitute it
    into the Gutenberg-Richter law;
  * b-free estimators, which estimate the size distribution directly from the
    sufficient statistics.  Each comes in a frequentist and a Bayesian form.

Every quantity that depends on how magnitudes were recorded says which case it
assumes, in its name.  `_rounded` is for magnitudes reported to a bin width;
`_continuous` is for magnitudes reported as real numbers.  Neither is the base
case and neither is a variant of the other: they are different models of the
data, geometric and exponential, and the estimators do not carry across.  A
continuous estimator applied to rounded magnitudes acquires the discretisation
bias that the grouped MLE of Bender exists to remove, and a rounded
estimator needs a bin index, which continuous magnitudes do not have:
`magnitude_statistics_rounded` refuses a catalogue that is off the lattice
rather than rounding it onto one.

That discretisation bias is a separate matter from the plug-in bias, which
comes from substituting an estimated b into a non-linear function and is
present in both cases.

The names say `rounded` because that is a property of the catalogue, which is
what these functions are given.  The lattice is a property of the model: the
manuscript writes the rounded law on a lattice of bin width 2*delta, and the
word is kept here for the grid itself rather than for the data on it.  Binned
is a third name for the same distinction.

Most catalogues are reported rounded, so that case is the one most often needed
in practice, but continuous catalogues are no longer unusual and the package
treats the two alike.

The plug-in estimators carry their authors' names rather than a suffix, because
the case each assumes is part of the method it names: b_aki is the continuous
maximum likelihood estimator and b_grouped the grouped one. b_grouped is named for
the form seismology uses, that of Tinti and Mulargia (1987); the estimator is
Bender's (1983), and both are cited where it is defined.

The reference magnitude m0 is a level chosen by the analyst, not the
completeness magnitude m_c of the catalogue.  Under `_continuous` the catalogue
starts at m0 itself and the requirement is m0 >= m_c.  Under `_rounded`
magnitudes lie on bins of width 2*delta centred on multiples of 2*delta above
m0, so m0 is a bin centre, the bottom of the catalogue is m0 - delta, and the
requirement is m0 - delta >= m_c.  Above that the estimators are unbiased, by
memorylessness, so raising m0 introduces no bias and costs only precision,
n ~ 10^(-b (m0-m_c)).
"""

import numpy as np
from scipy.special import betaln, digamma, gammaln

LOG10 = np.log(10.0)


# ---------------------------------------------------------------------------
# Sufficient statistics
# ---------------------------------------------------------------------------

def magnitude_statistics_rounded(m_obs, m0, delta):
    """
    Sufficient statistics of the magnitude-based formulation, rounded.

    Returns (n, T, W).  T is the excess sum measured from the bottom of the
    catalogue, m0 - delta, and W is the sum of bin indices.  delta = 0 is
    rejected rather than divided by: a bin index is undefined without a lattice,
    so continuous catalogues go through `magnitude_statistics_continuous`.

    *m0* may be an array, one threshold per event, for a catalogue whose
    completeness changes over time.  Each index is then counted from the
    event's own threshold, and because those thresholds sit on the same lattice
    every index is geometric with the same q: the b-free estimators apply
    unchanged to the pooled (n, W).
    """
    if delta <= 0:
        raise ValueError("delta must be positive; use "
                         "magnitude_statistics_continuous for continuous magnitudes")
    m_obs = np.asarray(m_obs, dtype=float)
    t = np.sum(m_obs - (m0 - delta))
    idx = (m_obs - m0) / (2 * delta)
    # A fractional index means the catalogue is not on the lattice m0 + 2*delta*j,
    # which the geometric model requires; rounding it away would hide that.
    if np.any(np.abs(idx - np.round(idx)) > 1e-6):
        raise ValueError("recorded magnitudes are not on the lattice implied by "
                         f"m0={m0} and delta={delta}")
    w = np.sum(np.round(idx).astype(np.int64))
    return len(m_obs), float(t), int(w)

def magnitude_statistics_continuous(m_obs, m0):
    """
    Sufficient statistics of the magnitude-based formulation, continuous.

    Returns (n, T), the plain excess sum above m0.  There is no half-bin
    correction and no bin index: the catalogue starts at m0 itself.
    """
    m_obs = np.asarray(m_obs, dtype=float)
    return len(m_obs), float(np.sum(m_obs - m0))


def difference_statistics_rounded(m_obs, m0, delta, overlapping=False, k_min=1):
    """
    Sufficient statistics of the difference-based (b-positive) formulation,
    rounded.

    Pairs are formed either from consecutive events (overlapping) or from
    even- and odd-indexed events (non-overlapping).  Only differences of at
    least *k_min* bins are retained.

    Returns (n+, T, W), measured from the bottom of the lowest retained
    difference bin, delta_m_th - delta.
    """
    if delta <= 0:
        raise ValueError("delta must be positive; use "
                         "difference_statistics_continuous for continuous magnitudes")
    m_obs = np.asarray(m_obs, dtype=float)
    k_idx = np.round((m_obs - m0) / (2 * delta)).astype(np.int64)
    k_i = k_idx[:-1] if overlapping else k_idx[0::2]
    k_j = k_idx[1:] if overlapping else k_idx[1::2]
    nr = min(len(k_i), len(k_j))
    diffs = k_j[:nr] - k_i[:nr]
    pos = diffs[diffs >= k_min]
    # Excess above the threshold, not above zero: both T and W are measured from
    # k_min, which matters as soon as delta_m_th exceeds one bin.
    t = np.sum((pos - k_min) * 2 * delta + delta)
    return len(pos), float(t), int(np.sum(pos - k_min))


def difference_statistics_continuous(m_obs, dm_th=0.0, overlapping=False):
    """
    Sufficient statistics of the difference-based formulation, continuous.

    Returns (n+, T) for the differences at or above *dm_th*, with T their excess
    sum above it.  Conditioning an exponential difference on exceeding a
    threshold leaves it exponential with the same rate, so none of the half-bin
    bookkeeping the rounded case needs arises here.
    """
    m_obs = np.asarray(m_obs, dtype=float)
    m_i = m_obs[:-1] if overlapping else m_obs[0::2]
    m_j = m_obs[1:] if overlapping else m_obs[1::2]
    nr = min(len(m_i), len(m_j))
    diffs = m_j[:nr] - m_i[:nr]
    pos = diffs[diffs >= dm_th]
    return len(pos), float(np.sum(pos - dm_th))


# ---------------------------------------------------------------------------
# Plug-in b-value estimators
# ---------------------------------------------------------------------------

def b_aki(t, n):
    """Aki-Utsu maximum likelihood estimator."""
    return n / (t * LOG10)


def b_ogata(t, n):
    """Ogata and Yamashina finite-size bias-corrected estimator, for n >= 2.

    At n = 1 the correction annihilates the estimate rather than degrading it,
    so the case is rejected instead of returning zero.
    """
    if np.any(np.asarray(n) < 2):
        raise ValueError("b_ogata requires n >= 2")
    return (n - 1) / (t * LOG10)

def _grouped_mle(t, n, delta):
    """Grouped maximum likelihood estimator of Bender (1983).

    Undefined when every event falls in the lowest bin: the mean excess is then
    zero, or a hair below it in floating point, and the estimator diverges.
    Callers drop the non-finite results rather than substitute a value.
    """
    mean_excess = t / n - delta
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log(1 + 2 * delta / mean_excess) / (LOG10 * 2 * delta)


def b_grouped(t, n, delta):
    """The grouped MLE, without a finite-size correction.

    Derived by Bender (1983), whose own word for the rounded likelihood is
    "grouped"; the closed form used here and generally in seismology is the one
    given by Tinti and Mulargia (1987), whose subject is its confidence
    intervals. Both are cited where it is defined.
    """
    return _grouped_mle(t, n, delta)


def b_assembled(t, n, delta):
    """The grouped MLE carrying Ogata and Yamashina's finite-size factor.

    Not a published estimator. It completes a two-by-two: b_aki and b_grouped
    maximise the exponential and the grouped likelihood respectively, and
    b_ogata and b_assembled apply (n-1)/n to each. The factor is exact only for
    unrounded magnitudes, so on a lattice this is assembled rather than derived.

    Utsu (1966) tabulated the large-catalogue ratio b_grouped/b_aki as a
    correction factor; it carries no n, so it is the ancestor of b_grouped and
    not of this.
    """
    if np.any(np.asarray(n) < 2):
        raise ValueError("b_assembled requires n >= 2")
    return (n - 1) / n * _grouped_mle(t, n, delta)


def gr_survival(m, b, m0):
    """Gutenberg-Richter survival function."""
    return 10.0 ** (-b * (np.asarray(m, dtype=float) - m0))


# ---------------------------------------------------------------------------
# b-free estimators, continuous magnitudes
# ---------------------------------------------------------------------------

def _as_columns(stat, n, coord):
    """Broadcast a catalogue statistic to (n_cat, 1) and a coordinate to (1, n_coord)."""
    stat = np.atleast_1d(stat).astype(float)[:, None]
    coord = np.atleast_1d(coord).astype(float)[None, :]
    n = np.atleast_1d(n).astype(float)
    n = n[:, None] if n.size > 1 else np.full_like(stat, n.item())
    return stat, n, coord


def bfree_survival_continuous(t, n, m):
    """
    Frequentist b-free survival function, continuous magnitudes.

    *t* is the plain excess sum sum(M_i - m0), without the half-bin correction
    that the rounded formulation carries; *m* is the excess magnitude m - m0.
    Support is [0, t], and the estimator is zero above it.  The mask is explicit
    because the exponent vanishes at n = 1, where numpy evaluates the clipped
    base as 0**0 = 1 and would return 1 everywhere; with the mask n = 1 gives the
    indicator 1(m <= t), which is the UMVUE there.
    """
    t, n, m = _as_columns(t, n, m)
    return np.where(m <= t,
                    np.clip(1.0 - m / t, 0.0, None) ** (n - 1),
                    0.0)


def bfree_pdf_continuous(t, n, m):
    """
    Frequentist b-free density, continuous magnitudes, for n >= 2.

    Zero at and above t, taken from the right as in `bfree_pdf_rounded`; at n = 2 the
    density jumps there rather than decaying to it.  At n = 1 the derivative of
    the survival function is a point mass and no ordinary density exists, so the
    case is rejected rather than returned as a negative power of zero.
    """
    if np.any(np.asarray(n) < 2):
        raise ValueError("bfree_pdf_continuous requires n >= 2; at n = 1 the "
                         "derivative of the survival function is a point mass")
    t, n, m = _as_columns(t, n, m)
    return np.where(m < t,
                    (n - 1) / t * np.clip(1.0 - m / t, 0.0, None) ** (n - 2),
                    0.0)


def bayesian_survival_continuous(t, n, m, n0=0.0, t0=0.0):
    """
    Bayesian b-free posterior predictive survival, continuous magnitudes.

    The defaults n0 = t0 = 0 are the Jeffreys prior p(beta) ~ 1/beta, which in
    this parameterisation is the empty hypothetical catalogue, Gamma(0, 0).  The
    rounded counterpart needs w0 = 1/2 instead, which is the asymmetry discussed
    in the manuscript.  Non-zero (n0, t0) give the conjugate case.
    """
    t, n, m = _as_columns(t, n, m)
    return (1.0 + m / (t + t0)) ** (-(n + n0))


def bayesian_pdf_continuous(t, n, m, n0=0.0, t0=0.0):
    """Bayesian b-free posterior predictive density, continuous magnitudes."""
    t, n, m = _as_columns(t, n, m)
    return (n + n0) / (t + t0) * (1.0 + m / (t + t0)) ** (-(n + n0 + 1))


def conjugate_prior_continuous(b_prior, n0_prior):
    """
    Gamma hyperparameters (n0, T0) of the continuous conjugate prior, expressed
    as a hypothetical catalogue of *n0_prior* events with b-value *b_prior*.
    """
    return float(n0_prior), n0_prior * np.log10(np.e) / b_prior


# ---------------------------------------------------------------------------
# b-free estimators, rounded magnitudes on the lattice m0 + 2*delta*j
# ---------------------------------------------------------------------------

def bfree_survival_rounded(w, n, k):
    """
    Frequentist b-free survival function, rounded magnitudes.

    *w* is the catalogue total sum(K_i) and *k* the bin level (m - m0) / 2*delta
    at which to evaluate.  *w* and *n* may be scalars or arrays of equal length;
    the result has shape (n_cat, n_k).

    A real-valued level is supported on [0, w+1): the expression is positive on
    (w, w+1) and is set to zero from w+1 onward, where gammaln returns
    log|Gamma| and so discards the sign.  It is exactly unbiased at integer
    levels only; see `bfree_pdf_rounded` and the paper's supplementary material.
    """
    w, n, k = _as_columns(w, n, k)
    with np.errstate(invalid='ignore'):
        sf = np.exp((gammaln(w - k + n) - gammaln(w - k + 1))
                    - (gammaln(w + n) - gammaln(w + 1)))
    sf[k >= w + 1] = 0.0
    return sf


def bfree_pdf_rounded(w, n, u, delta):
    """
    Frequentist b-free density, rounded magnitudes, from the continued survival.

    The derivative of `bfree_survival_rounded` with respect to magnitude, so *delta* is
    needed to convert from bin levels to magnitude units.  *u* is the continued
    real coordinate, integer at a bin edge and half-integer at a bin centre.
    Unlike the binned mass this is a genuine density and may be evaluated
    between bin edges.  It is unbiased only asymptotically in *n*, not exactly:
    see the paper's supplementary material.

    The survival function is clamped from w+1 onward, where it reaches zero with
    a non-zero slope: it has a kink there rather than a derivative, so the density
    jumps.  It is taken from the right and set to zero, which makes the support
    the half-open interval [0, w+1).  Requires n >= 2, as the continuous density
    does; at n = 1 the rounded estimator is the mass 1(K_1 == k), not a density.
    """
    if np.any(np.asarray(n) < 2):
        raise ValueError("bfree_pdf_rounded requires n >= 2; at n = 1 the rounded "
                         "estimator is the mass 1(K_1 == k), not a density")
    w, n, u = _as_columns(w, n, u)
    with np.errstate(invalid='ignore'):
        sf = np.exp((gammaln(w - u + n) - gammaln(w - u + 1))
                    - (gammaln(w + n) - gammaln(w + 1)))
        pdf = sf * (digamma(w - u + n) - digamma(w - u + 1)) / (2.0 * delta)
    pdf[u >= w + 1] = 0.0
    return pdf


def bayesian_survival_rounded(w, n, k, w0=0.5, n0=0.0):
    """
    Bayesian b-free posterior predictive survival function, rounded magnitudes.

    The defaults w0 = 1/2, n0 = 0 are the Jeffreys prior for the geometric
    likelihood, Beta(1/2, 0).  It is improper; the posterior Beta(W+1/2, n) is
    proper for n >= 1.  Note that Beta(1/2, 1/2) is the Jeffreys prior for the
    binomial design, which is not the one the catalogue realises.  Shapes follow
    `bfree_survival_rounded`.
    """
    w, n, k = _as_columns(w, n, k)
    return np.exp(betaln(w + w0 + k, n + n0) - betaln(w + w0, n + n0))


def bayesian_pdf_rounded(w, n, u, delta, w0=0.5, n0=0.0):
    """
    Bayesian b-free posterior predictive density, rounded magnitudes.

    Derivative of `bayesian_survival_rounded` with respect to magnitude.  With
    a = W + w0 + u and b = n + n0, d/du log B(a, b) = psi(a) - psi(a + b).
    """
    w, n, u = _as_columns(w, n, u)
    a = w + w0 + u
    b = n + n0
    sf = np.exp(betaln(a, b) - betaln(w + w0, b))
    return sf * (digamma(a + b) - digamma(a)) / (2.0 * delta)


def conjugate_prior_rounded(b_prior, n0_prior, delta):
    """
    Beta hyperparameters (W0, n0) of a conjugate prior expressed as a
    hypothetical catalogue of *n0_prior* events with b-value *b_prior*.
    """
    q_prior = 10.0 ** (-2.0 * delta * b_prior)
    return n0_prior * q_prior / (1.0 - q_prior), float(n0_prior)


def survival_to_pdf(sf, bin_width):
    """Convert a survival function sampled at bin edges to a binned PDF."""
    return -np.diff(sf, axis=-1) / bin_width
