# API

Everything public is re-exported from `bfree`, so callers need not track the
module layout:

```python
from bfree import bfree_survival_rounded, magnitude_statistics_rounded
```

Every function whose answer depends on how magnitudes were recorded says which
case it assumes, in its name. `_rounded` is for magnitudes reported to a bin
width $2\delta$, on bins centred on multiples of $2\delta$ above $m_0$;
`_continuous` is for magnitudes reported as real numbers above $m_0$.

Neither is the base case and neither is a variant of the other. They are
different models of the data — geometric and exponential — and the estimators
do not carry across. A continuous estimator applied to rounded magnitudes
acquires the discretisation bias that the grouped MLE of Bender
exists to remove; a rounded estimator needs a bin index, which continuous
magnitudes do not have, and `magnitude_statistics_rounded` refuses a catalogue
that is off the lattice rather than rounding it onto one. That discretisation
bias is a separate matter from the plug-in bias, which comes from substituting
an estimated $b$ into a non-linear function and is present in both cases.

The names say **rounded** because that is a property of the catalogue, which is
what these functions are given. The **lattice** is a property of the model: the
manuscript writes the rounded law on a lattice of bin width $2\delta$, and the
word is kept for the grid itself rather than for the data on it. **Binned** is a
third name for the same distinction.

Most catalogues are reported rounded, so that case is the one most often needed
in practice, but continuous catalogues are no longer unusual and the package
treats the two alike.

The reference $m_0$ is a level chosen by the analyst, not the completeness
magnitude $m_c$. In the continuous case the catalogue starts at $m_0$ itself
and the requirement is $m_0 \geq m_c$. In the rounded case $m_0$ is a bin
centre, so the bottom of the catalogue is $m_0 - \delta$ and the requirement is
$m_0 - \delta \geq m_c$. Above that the estimators are unbiased by
memorylessness, so raising $m_0$ introduces no bias and costs only precision,
$n \sim 10^{-b(m_0 - m_c)}$.

## Sufficient statistics

| Function | Returns |
|---|---|
| `magnitude_statistics_rounded(m_obs, m0, delta)` | `(n, T, W)` — the count, the excess sum measured from the bottom of the catalogue, and the sum of bin indices. Rejects magnitudes off the implied lattice rather than rounding them onto it. |
| `magnitude_statistics_continuous(m_obs, m0)` | `(n, T)` for continuous magnitudes; no half-bin correction and no bin index. |
| `difference_statistics_rounded(m_obs, m0, delta, overlapping=False, k_min=1)` | `(n⁺, T, W)` for the difference-based formulation. Both excesses are measured from the threshold `k_min`, not from zero — these agree at `k_min = 1` and diverge above it. |
| `difference_statistics_continuous(m_obs, dm_th=0.0, overlapping=False)` | The continuous counterpart. |

## Plug-in b-value estimators

| Function | Estimator |
|---|---|
| `b_aki(t, n)` | Aki–Utsu maximum likelihood. |
| `b_ogata(t, n)` | Ogata–Yamashina finite-size bias correction. Requires `n >= 2`; the guard is vectorised, so one out-of-domain element rejects the whole batch and callers must mask. |
| `b_grouped(t, n, delta)` | The grouped MLE. Derived by Bender (1983); the closed form used here, and generally in seismology, is that of Tinti and Mulargia (1987), whose own subject is its confidence intervals. Named for the form a reader will recognise. |
| `b_assembled(t, n, delta)` | `b_grouped` carrying Ogata's `(n-1)/n` factor, so that both the rounding and the finite-size bias are corrected. This combination is not published; the factor is exact only for unrounded magnitudes, so applying it on a lattice is an approximation, tested in `analyse_plugin_estimators.py`. Requires `n >= 2`. |
| `gr_survival(m, b, m0)` | The Gutenberg–Richter survival function. |

## b-free estimators

Each comes in a frequentist and a Bayesian form, and each of those in a
continuous and a rounded form.

| Function | Notes |
|---|---|
| `bfree_survival_rounded(w, n, k)` | Frequentist, rounded. Exactly unbiased at integer bin levels; a real-valued level is the continuation, supported on $[0, w+1)$. |
| `bfree_pdf_rounded(w, n, u, delta)` | The derivative of the above with respect to magnitude. Unbiased only asymptotically in `n`; requires `n >= 2`. |
| `bfree_survival_continuous(t, n, m)` | Frequentist, continuous. Support $[0, t]$. At `n = 1` this is the indicator, which is the UMVUE there. |
| `bfree_pdf_continuous(t, n, m)` | Requires `n >= 2`; at `n = 1` the derivative is a point mass and no density exists. |
| `bayesian_survival_rounded(w, n, k, w0=0.5, n0=0.0)` | Posterior predictive, rounded. The defaults are the Jeffreys prior for the geometric likelihood, $\mathrm{Beta}(1/2, 0)$ — **not** $\mathrm{Beta}(1/2, 1/2)$, which is the Jeffreys prior for a binomial design and not the one a catalogue realises. |
| `bayesian_pdf_rounded(w, n, u, delta, w0=0.5, n0=0.0)` | The density form. |
| `bayesian_survival_continuous(t, n, m, n0=0.0, t0=0.0)` | Continuous. The defaults are $p(\beta) \propto 1/\beta$, the empty hypothetical catalogue. |
| `bayesian_pdf_continuous(t, n, m, n0=0.0, t0=0.0)` | The density form. |

## Priors and conversions

| Function | Purpose |
|---|---|
| `conjugate_prior_rounded(b_prior, n0_prior, delta)` | Beta hyperparameters of the rounded conjugate prior, expressed as a hypothetical catalogue of `n0_prior` events with b-value `b_prior`. |
| `conjugate_prior_continuous(b_prior, n0_prior)` | The Gamma hyperparameters of the continuous counterpart. |
| `survival_to_pdf(sf, bin_width)` | Converts a survival function sampled at bin edges to a binned mass. |

## Results on disk

`bfree.store` reads and writes the result files. Two functions:

| Function | Purpose |
|---|---|
| `save(path, obj)` | Writes a nested result structure to a Zarr store. |
| `load(path)` | Reads one back, with the same types it went in with. |

The mapping is:

| In Python | In the store |
|---|---|
| `dict` | a group |
| `numpy` array | an array |
| `list`, `tuple` | a group of numbered members, marked with its kind |
| `DataFrame` | a group of one array per column |
| number, string, `None` | an attribute on the parent group |

A store holds arrays and JSON, so reading one cannot run code. Each result is
one Zip store, so it is a single file rather than the several thousand a
directory would need; `unzip` gives the directory form if a reader's Zarr
implementation wants that. Dictionary keys are written as strings, so
a tuple key such as `(1.0, 100)` becomes the group name `1_100`; `save` refuses
a dictionary whose keys collide under that rule rather than dropping one, and
`load` restores the original key from the recorded type.

## Figures

`bfree.plotting` holds the shared drawing code and estimator color palette;
`bfree.bssa` holds the journal figure settings; `bfree.provenance` writes the
figure manifest. These support `scripts/make_figures.py` and are documented in
[figures.md](figures.md) rather than here.
