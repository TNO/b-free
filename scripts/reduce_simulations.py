#!/usr/bin/env python
"""
reduce_simulations.py

Reduces a simulation zarr store to the quantities plotted in the manuscript:
for every estimator and every magnitude bin, the mean estimated PDF over all
simulated catalogues and its ratio to the ground-truth PDF, together with the
median and root-mean-square error of that ratio.

The mean alone is not sufficient to compare estimators.  An estimator can be
exactly unbiased in the mean while falling below the truth for the majority of
individual catalogues, if its sampling distribution is skewed - which is the
case in the tail of the distribution.  The median therefore describes the
typical single-catalogue outcome, and the RMSE combines bias and variance into
one decision-theoretic criterion.

Processing is done in chunks, so the full (n_simulations x n_bins) arrays are
never materialised: only running sums and a histogram of the ratio are kept.
The median is read off that histogram, so it is accurate to the histogram
resolution (RATIO_BINS_PER_DECADE) rather than exact.

Usage
-----
    python reduce_simulations.py
    python reduce_simulations.py --input my_sim.zarr --output my_analysis.zarr
    python reduce_simulations.py --help
"""

import argparse
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import zarr

from bfree import (
    LOG10,
    b_aki,
    b_assembled,
    b_grouped,
    b_ogata,
    bayesian_pdf_continuous,
    bayesian_pdf_rounded,
    bayesian_survival_continuous,
    bayesian_survival_rounded,
    bfree_pdf_continuous,
    bfree_pdf_rounded,
    bfree_survival_continuous,
    bfree_survival_rounded,
    conjugate_prior_continuous,
    conjugate_prior_rounded,
)
from bfree.store import save as save_results

MAG_KEYS = ['b_aki', 'b_ogata', 'b_grouped', 'b_assembled',
            'direct_mag', 'direct_mag_jeffreys', 'direct_mag_conjugate']
NOL_KEYS = ['b_pos_aki_nol', 'b_pos_ogata_nol', 'b_pos_grouped_nol',
            'direct_diff_nol', 'direct_diff_nol_jeffreys',
            'direct_diff_nol_conjugate']
OL_KEYS = ['b_pos_aki_ol', 'b_pos_ogata_ol', 'b_pos_grouped_ol',
           'direct_diff_ol', 'direct_diff_ol_jeffreys',
           'direct_diff_ol_conjugate']
ALL_KEYS = MAG_KEYS + NOL_KEYS + OL_KEYS

# Difference catalogues are kept only from two retained differences upwards. The
# densities compared here, and the Ogata plug-in, are undefined at one: the
# comparison is therefore conditional on n_diff >= 2 rather than on n_diff > 0.
# It is a rare exclusion - 283 of 1.2e8 stored catalogues, all at n = 50 - but it
# has to be the same sample for every estimator in the panel.
MIN_N_DIFF = 2

# Histogram of the estimator/truth ratio, used for the median. Log-spaced so
# that the resolution is uniform in relative terms across several decades.
RATIO_MIN, RATIO_MAX = 1e-4, 1e4
RATIO_BINS_PER_DECADE = 250
RATIO_EDGES = np.logspace(
    np.log10(RATIO_MIN), np.log10(RATIO_MAX),
    round(np.log10(RATIO_MAX / RATIO_MIN) * RATIO_BINS_PER_DECADE) + 1)
N_RATIO_BINS = len(RATIO_EDGES) - 1


def _median_from_hist(counts):
    """
    Median of the ratio from its histogram, per magnitude bin.

    counts has shape (n_bins_magnitude, N_RATIO_BINS + 2); the first and last
    columns are the under- and overflow bins and are counted but cannot be
    interpolated into, so a median falling there is reported as the
    corresponding limit.
    """
    centres = np.sqrt(RATIO_EDGES[:-1] * RATIO_EDGES[1:])      # geometric
    out = np.full(counts.shape[0], np.nan)
    for j, row in enumerate(counts):
        total = row.sum()
        if total == 0:
            continue
        cum = np.cumsum(row)
        idx = int(np.searchsorted(cum, 0.5 * total))
        if idx == 0:
            out[j] = RATIO_MIN
        elif idx >= N_RATIO_BINS + 1:
            out[j] = RATIO_MAX
        else:
            out[j] = centres[idx - 1]
    return out


# ---------------------------------------------------------------------------
# Per-combination computation
# ---------------------------------------------------------------------------

def _frac_below_one(counts):
    """
    Fraction of catalogues on which the estimator falls below the ground truth.

    An unbiased estimator with a right-skewed sampling distribution can exceed
    0.5 here, i.e. be too low for most individual catalogues while still having
    the correct mean.
    """
    below = int(np.searchsorted(RATIO_EDGES, 1.0))     # ratio bins fully < 1
    out = np.full(counts.shape[0], np.nan)
    for j, row in enumerate(counts):
        total = row.sum()
        if total:
            out[j] = row[:below].sum() / total
    return out


def compute_combination(zarr_path, b_true, n_events, chunk_size=100_000,
                        mmin=None, mmax=None, b_prior=1.0, n0_prior=100):
    """
    Compute the mean PDF and ratio-to-truth for one (b_true, n_events) combination.

    The zarr store is opened read-only and streamed *chunk_size* rows at a time,
    so peak memory is O(chunk_size) regardless of the number of simulations.

    Parameters
    ----------
    b_prior, n0_prior : float
        Conjugate informative prior expressed as a hypothetical catalogue of
        *n0_prior* events with b-value *b_prior*.  The Jeffreys prior
        (W0 = 1/2, n0 = 0) is always computed in addition.

    Returns
    -------
    dict with keys:
        pdf_mrange        — bin centres
        pdf_ground_truth  — analytic PDF (n_pdf,)
        estimators        — per estimator: {'pdf_mean', 'ratio_mean'}
        metadata
    """
    store = zarr.open(zarr_path, mode='r')
    g = store[f"b{float(b_true)}_n{int(n_events)}"]

    delta = g.attrs['delta']
    mc = g.attrs['mc']
    n_total = g['t_mag'].shape[0]
    bin_width = 2 * delta

    if mmin is None:
        mmin = mc - delta
    if mmax is None:
        mmax = mmin + 5 + bin_width + delta

    edges = np.arange(mmin, mmax, bin_width)
    pdf_mrange = edges[1:] - delta
    u_centres = (pdf_mrange - mmin) / bin_width      # half-integers: bin centres
    n_pdf = len(pdf_mrange)

    # The survival function is evaluated at bin edges, where the rounded
    # estimator is the continuous one exactly rather than a continuation.
    sf_mrange = edges
    k_edges = np.rint((sf_mrange - mmin) / bin_width)
    n_sf = len(sf_mrange)

    beta_true = b_true * LOG10
    pdf_gt = beta_true * np.exp(-beta_true * (pdf_mrange - mmin))
    sf_gt = np.exp(-beta_true * (sf_mrange - mmin))

    w0_jeffreys, n0_jeffreys = 0.5, 0.0        # Beta(1/2, 0), geometric likelihood
    w0_conj, n0_conj = conjugate_prior_rounded(b_prior, n0_prior, delta)

    n_ratio_cols = N_RATIO_BINS + 2
    grids = {
        'pdf': {'gt': pdf_gt, 'n': n_pdf,
                'offsets': (np.arange(n_pdf) * n_ratio_cols)[None, :]},
        'sf': {'gt': sf_gt, 'n': n_sf,
               'offsets': (np.arange(n_sf) * n_ratio_cols)[None, :]},
    }
    total = {q: {k: np.zeros(grids[q]['n']) for k in ALL_KEYS} for q in grids}
    total_sq = {q: {k: np.zeros(grids[q]['n']) for k in ALL_KEYS} for q in grids}
    hist_ratio = {q: {k: np.zeros((grids[q]['n'], n_ratio_cols), dtype=np.int64)
                      for k in ALL_KEYS} for q in grids}
    n_valid_nol = 0
    n_valid_ol = 0

    def _accumulate(quantity, key, matrix):
        """Running sums and ratio histogram for one chunk of catalogues."""
        grid = grids[quantity]
        gt, n_cols = grid['gt'], grid['n']
        total[quantity][key] += matrix.sum(axis=0)
        total_sq[quantity][key] += np.square(matrix).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(gt > 0, matrix / gt, np.nan)
        # searchsorted puts NaN past the last edge, i.e. into the overflow bin
        idx = np.searchsorted(RATIO_EDGES, ratio)
        np.clip(idx, 0, N_RATIO_BINS + 1, out=idx)
        # One bincount over a flattened (magnitude bin, ratio bin) index is far
        # cheaper than one bincount per magnitude bin.
        flat = idx + grid['offsets']
        hist_ratio[quantity][key] += np.bincount(
            flat.ravel(), minlength=n_cols * n_ratio_cols
        ).reshape(n_cols, n_ratio_cols)

    def _add_plugin(key, b_vals):
        """Accumulate a plug-in estimator from its b-values, skipping non-finite ones."""
        good = np.isfinite(b_vals) & (b_vals > 0)
        if not good.any():
            return
        beta = b_vals[good, None] * LOG10
        _accumulate('pdf', key, beta * np.exp(-beta * (pdf_mrange - mmin)))
        _accumulate('sf', key, np.exp(-beta * (sf_mrange - mmin)))

    def _add_direct(key, pdf, sf):
        """Accumulate a b-free estimator, which already supplies both curves."""
        _accumulate('pdf', key, pdf)
        _accumulate('sf', key, sf)

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        t_mag = g['t_mag'][start:end].astype(np.float64)
        w_mag = g['w_mag'][start:end].astype(np.int64)

        _add_plugin('b_aki', b_aki(t_mag, n_events))
        _add_plugin('b_ogata', b_ogata(t_mag, n_events))
        _add_plugin('b_grouped', b_grouped(t_mag, n_events, delta))
        _add_plugin('b_assembled', b_assembled(t_mag, n_events, delta))

        _add_direct('direct_mag',
                    bfree_pdf_rounded(w_mag, n_events, u_centres, delta),
                    bfree_survival_rounded(w_mag, n_events, k_edges))
        _add_direct('direct_mag_jeffreys',
                    bayesian_pdf_rounded(w_mag, n_events, u_centres, delta,
                                 w0_jeffreys, n0_jeffreys),
                    bayesian_survival_rounded(w_mag, n_events, k_edges,
                                      w0_jeffreys, n0_jeffreys))
        _add_direct('direct_mag_conjugate',
                    bayesian_pdf_rounded(w_mag, n_events, u_centres, delta, w0_conj, n0_conj),
                    bayesian_survival_rounded(w_mag, n_events, k_edges, w0_conj, n0_conj))

        for suffix in ('_nol', '_ol'):
            t_diff = g[f't_diff{suffix}'][start:end].astype(np.float64)
            w_diff = g[f'w_diff{suffix}'][start:end].astype(np.int64)
            n_diff = g[f'n_diff{suffix}'][start:end].astype(np.int64)

            valid = n_diff >= MIN_N_DIFF
            if suffix == '_nol':
                n_valid_nol += int(valid.sum())
            else:
                n_valid_ol += int(valid.sum())
            if not valid.any():
                continue

            td, wd, nd = t_diff[valid], w_diff[valid], n_diff[valid]

            _add_plugin(f'b_pos_aki{suffix}', b_aki(td, nd))
            _add_plugin(f'b_pos_ogata{suffix}', b_ogata(td, nd))
            _add_plugin(f'b_pos_grouped{suffix}', b_grouped(td, nd, delta))

            _add_direct(f'direct_diff{suffix}',
                        bfree_pdf_rounded(wd, nd, u_centres, delta),
                        bfree_survival_rounded(wd, nd, k_edges))
            _add_direct(f'direct_diff{suffix}_jeffreys',
                        bayesian_pdf_rounded(wd, nd, u_centres, delta,
                                     w0_jeffreys, n0_jeffreys),
                        bayesian_survival_rounded(wd, nd, k_edges,
                                          w0_jeffreys, n0_jeffreys))
            _add_direct(f'direct_diff{suffix}_conjugate',
                        bayesian_pdf_rounded(wd, nd, u_centres, delta, w0_conj, n0_conj),
                        bayesian_survival_rounded(wd, nd, k_edges, w0_conj, n0_conj))

    estimators = {}
    for key in ALL_KEYS:
        if key in NOL_KEYS:
            n_denom = n_valid_nol
        elif key in OL_KEYS:
            n_denom = n_valid_ol
        else:
            n_denom = n_total
        if n_denom == 0:
            continue

        entry = {}
        for quantity, prefix in (('pdf', 'pdf'), ('sf', 'sf')):
            gt = grids[quantity]['gt']
            mean = total[quantity][key] / n_denom
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio_mean = np.where(gt > 0, mean / gt, np.nan)
                # E[(ratio - 1)^2] = E[ratio^2] - 2 E[ratio] + 1
                ratio_sq_mean = np.where(
                    gt > 0, (total_sq[quantity][key] / n_denom) / np.square(gt), np.nan)
                ratio_rmse = np.sqrt(
                    np.maximum(ratio_sq_mean - 2 * ratio_mean + 1.0, 0.0))
            entry[f'{prefix}_mean'] = mean
            entry[f'{prefix}_ratio_mean'] = ratio_mean
            entry[f'{prefix}_ratio_median'] = _median_from_hist(hist_ratio[quantity][key])
            entry[f'{prefix}_ratio_rmse'] = ratio_rmse
            # fraction of catalogues on which the estimator falls below truth
            entry[f'{prefix}_frac_below_truth'] = _frac_below_one(hist_ratio[quantity][key])
        # Names without a prefix stay on the density, so existing readers work.
        entry['ratio_mean'] = entry['pdf_ratio_mean']
        entry['ratio_median'] = entry['pdf_ratio_median']
        entry['ratio_rmse'] = entry['pdf_ratio_rmse']
        entry['frac_below_truth'] = entry['pdf_frac_below_truth']
        estimators[key] = entry

    return {
        'pdf_mrange': pdf_mrange,
        'pdf_ground_truth': pdf_gt,
        'sf_mrange': sf_mrange,
        'sf_ground_truth': sf_gt,
        'estimators': estimators,
        'metadata': {
            'b_true': b_true,
            'n_events': n_events,
            'mc': mc,
            'delta': delta,
            'n_simulations': n_total,
            'n_valid_nol': n_valid_nol,
            'n_valid_ol': n_valid_ol,
            'b_prior': b_prior,
            'n0_prior': n0_prior,
            'W0_conjugate': w0_conj,
        },
    }


# ---------------------------------------------------------------------------
# Unrounded counterpart
# ---------------------------------------------------------------------------

CONTINUOUS_KEYS = ['b_aki', 'b_ogata', 'direct_mag',
                   'direct_mag_jeffreys', 'direct_mag_conjugate']


def compute_continuous(b_true, n_events, n_catalogs=10_000_000,
                       chunk_size=100_000, m_max=5.0, n_grid=101, seed=None,
                       b_prior=1.0, n0_prior=100):
    """
    The same reduction as `compute_combination`, for unrounded catalogues.

    Magnitudes are drawn continuously and never rounded, so every quantity is a
    genuine density evaluated pointwise: there is no lattice, no bin mass and no
    half-bin offset.  The grouped estimators of Bender have no
    counterpart here, since there is nothing to group.

    Catalogues are generated on the fly rather than read from a store, which is
    cheap enough that no simulation output needs to be kept.
    """
    beta_true = b_true * LOG10
    mrange = np.linspace(0.0, m_max, n_grid)          # excess above m0
    pdf_gt = beta_true * np.exp(-beta_true * mrange)
    sf_gt = np.exp(-beta_true * mrange)

    n_ratio_cols = N_RATIO_BINS + 2
    grids = {'pdf': pdf_gt, 'sf': sf_gt}
    total = {q: {k: np.zeros(n_grid) for k in CONTINUOUS_KEYS} for q in grids}
    total_sq = {q: {k: np.zeros(n_grid) for k in CONTINUOUS_KEYS} for q in grids}
    hist_ratio = {q: {k: np.zeros((n_grid, n_ratio_cols), dtype=np.int64)
                      for k in CONTINUOUS_KEYS} for q in grids}
    bin_offsets = (np.arange(n_grid) * n_ratio_cols)[None, :]
    rng = np.random.default_rng(seed)
    n0_conj, t0_conj = conjugate_prior_continuous(b_prior, n0_prior)

    def _accumulate(quantity, key, matrix):
        """Running sums and ratio histogram for one chunk of catalogues."""
        gt = grids[quantity]
        total[quantity][key] += matrix.sum(axis=0)
        total_sq[quantity][key] += np.square(matrix).sum(axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(gt > 0, matrix / gt, np.nan)
        idx = np.searchsorted(RATIO_EDGES, ratio)
        np.clip(idx, 0, N_RATIO_BINS + 1, out=idx)
        flat = idx + bin_offsets
        hist_ratio[quantity][key] += np.bincount(
            flat.ravel(), minlength=n_grid * n_ratio_cols
        ).reshape(n_grid, n_ratio_cols)

    for start in range(0, n_catalogs, chunk_size):
        n_this = min(chunk_size, n_catalogs - start)
        t = rng.exponential(1.0 / beta_true, (n_this, n_events)).sum(axis=1)

        for key, b_vals in (('b_aki', b_aki(t, n_events)),
                            ('b_ogata', b_ogata(t, n_events))):
            beta = b_vals[:, None] * LOG10
            _accumulate('pdf', key, beta * np.exp(-beta * mrange))
            _accumulate('sf', key, np.exp(-beta * mrange))

        _accumulate('pdf', 'direct_mag', bfree_pdf_continuous(t, n_events, mrange))
        _accumulate('sf', 'direct_mag', bfree_survival_continuous(t, n_events, mrange))
        _accumulate('pdf', 'direct_mag_jeffreys',
                    bayesian_pdf_continuous(t, n_events, mrange))
        _accumulate('sf', 'direct_mag_jeffreys',
                    bayesian_survival_continuous(t, n_events, mrange))
        _accumulate('pdf', 'direct_mag_conjugate',
                    bayesian_pdf_continuous(t, n_events, mrange,
                                            n0=n0_conj, t0=t0_conj))
        _accumulate('sf', 'direct_mag_conjugate',
                    bayesian_survival_continuous(t, n_events, mrange,
                                                 n0=n0_conj, t0=t0_conj))

    estimators = {}
    for key in CONTINUOUS_KEYS:
        entry = {}
        for quantity, prefix in (('pdf', 'pdf'), ('sf', 'sf')):
            gt = grids[quantity]
            mean = total[quantity][key] / n_catalogs
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio_mean = np.where(gt > 0, mean / gt, np.nan)
                ratio_sq_mean = np.where(
                    gt > 0, (total_sq[quantity][key] / n_catalogs) / np.square(gt), np.nan)
                ratio_rmse = np.sqrt(
                    np.maximum(ratio_sq_mean - 2 * ratio_mean + 1.0, 0.0))
            entry[f'{prefix}_mean'] = mean
            entry[f'{prefix}_ratio_mean'] = ratio_mean
            entry[f'{prefix}_ratio_median'] = _median_from_hist(hist_ratio[quantity][key])
            entry[f'{prefix}_ratio_rmse'] = ratio_rmse
            entry[f'{prefix}_frac_below_truth'] = _frac_below_one(hist_ratio[quantity][key])
        entry['ratio_mean'] = entry['pdf_ratio_mean']
        entry['ratio_median'] = entry['pdf_ratio_median']
        entry['ratio_rmse'] = entry['pdf_ratio_rmse']
        entry['frac_below_truth'] = entry['pdf_frac_below_truth']
        estimators[key] = entry

    return {
        'pdf_mrange': mrange,
        'pdf_ground_truth': pdf_gt,
        'sf_mrange': mrange,
        'sf_ground_truth': sf_gt,
        'estimators': estimators,
        'metadata': {
            'b_true': b_true,
            'n_events': n_events,
            'n_simulations': n_catalogs,
            'rounded': False,
            'b_prior': b_prior,
            'n0_prior': float(n0_prior),
            'T0_conjugate': t0_conj,
        },
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _fmt(s):
    """Seconds as h:mm:ss, or '??:??' when the estimate is not yet finite."""
    if not np.isfinite(float(s)):
        return "??:??"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _analysis_worker(args):
    """Unpacking wrapper so Pool.imap_unordered can call compute_combination."""
    zarr_path, b_true, n_events, chunk_size, b_prior, n0_prior = args
    result = compute_combination(zarr_path, b_true, n_events, chunk_size,
                                 b_prior=b_prior, n0_prior=n0_prior)
    return (b_true, n_events), result


def run(input_path, output_path, chunk_size, n_workers=1,
        b_prior=1.0, n0_prior=100):
    """
    Compute the analysis for all combinations in *input_path* (zarr store).

    Each combination is processed by an independent worker process that opens
    the zarr store read-only — zarr supports concurrent reads safely.
    """
    store = zarr.open(input_path, mode='r')
    keys = sorted(
        [(float(g.attrs['b_true']), int(g.attrs['n_events']))
         for _, g in store.groups()],
        key=lambda x: (x[0], x[1]),
    )
    n_combos = len(keys)
    n_sim = next(store.group_values()).attrs['n_catalogs']
    print(
        f"Analysing {input_path}\n"
        f"  {n_combos} combinations, {n_sim:,} simulations each\n"
        f"  chunk size = {chunk_size:,}, workers = {n_workers}\n"
        f"  conjugate prior: b_prior={b_prior}, n0_prior={n0_prior}\n"
    )

    tasks = [(input_path, b, n, chunk_size, b_prior, n0_prior) for b, n in keys]

    analysis = {}
    t_start = time.perf_counter()
    completed = 0

    with Pool(processes=n_workers) as pool:
        for (b_true, n_events), result in pool.imap_unordered(
                _analysis_worker, tasks):
            analysis[(b_true, n_events)] = result
            completed += 1
            elapsed = time.perf_counter() - t_start
            rate = completed / max(elapsed, 1e-6)
            eta = (n_combos - completed) / rate
            print(f"  [{completed:2d}/{n_combos}]"
                  f"  b_true={b_true}, n_events={n_events}"
                  f"  done  (elapsed {_fmt(elapsed)}, ETA {_fmt(eta)})")

    print(f"\nSaving to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, analysis)

    print(f"Done.  Total time: {_fmt(time.perf_counter() - t_start)}")


def run_continuous(output_path, b_values, n_events_values, n_catalogs,
                   chunk_size, seed=None, b_prior=1.0, n0_prior=100):
    """Compute the unrounded analysis for every (b, n) combination requested."""
    keys = [(b, n) for b in b_values for n in n_events_values]
    print(
        f"Analysing unrounded catalogues\n"
        f"  {len(keys)} combinations, {n_catalogs:,} simulations each\n"
        f"  chunk size = {chunk_size:,}\n"
    )

    analysis = {}
    t_start = time.perf_counter()
    for i, (b_true, n_events) in enumerate(keys, 1):
        analysis[(b_true, n_events)] = compute_continuous(
            b_true, n_events, n_catalogs=n_catalogs,
            chunk_size=chunk_size, seed=seed,
            b_prior=b_prior, n0_prior=n0_prior)
        elapsed = time.perf_counter() - t_start
        eta = (len(keys) - i) * elapsed / i
        print(f"  [{i:2d}/{len(keys)}]  b_true={b_true}, n_events={n_events}"
              f"  done  (elapsed {_fmt(elapsed)}, ETA {_fmt(eta)})")

    print(f"\nSaving to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, analysis)
    print(f"Done.  Total time: {_fmt(time.perf_counter() - t_start)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-compute analysis statistics from a simulation zarr store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', default='results/simulation_results.zarr', metavar='FILE')
    parser.add_argument('--output', default='results/analysis_results.zarr.zip', metavar='FILE')
    parser.add_argument(
        '--chunk_size', type=lambda x: int(float(x)), default=100_000, metavar='N',
        help='Zarr rows read per processing chunk. Smaller means less memory.',
    )
    parser.add_argument(
        '--workers', type=int, default=max(1, cpu_count() - 1), metavar='W',
        help='Number of parallel worker processes (one per combination).',
    )
    parser.add_argument(
        '--b_prior', type=float, default=1.0, metavar='B',
        help='Prior b-value for the conjugate informative prior.',
    )
    parser.add_argument(
        '--n0_prior', type=float, default=100.0, metavar='N',
        help='Pseudo-count strength for the conjugate informative prior '
             '(equivalent number of prior observations).',
    )
    parser.add_argument(
        '--continuous', action='store_true',
        help='Analyse unrounded catalogues, generated on the fly, instead of '
             'reducing the zarr store.',
    )
    parser.add_argument(
        '--n_catalogs', type=lambda x: int(float(x)), default=10_000_000, metavar='N',
        help='Number of catalogues per combination, --continuous only.',
    )
    parser.add_argument(
        '--b_values', type=float, nargs='+', default=[1.0], metavar='B',
        help='b-values to analyse, --continuous only.',
    )
    parser.add_argument(
        '--n_events', type=int, nargs='+', default=[100], metavar='N',
        help='Catalogue sizes to analyse, --continuous only.',
    )
    parser.add_argument(
        '--seed', type=int, default=None, metavar='S',
        help='Random seed, --continuous only.  Required there: the unrounded '
             'catalogues are generated on the fly rather than read from the '
             'zarr store, so the seed is the only record of what was analysed.',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.continuous:
        if args.seed is None:
            raise SystemExit('--continuous requires --seed; the catalogues are '
                             'generated on the fly and are not otherwise stored')
        run_continuous(args.output, args.b_values, args.n_events,
                       args.n_catalogs, args.chunk_size, seed=args.seed,
                       b_prior=args.b_prior, n0_prior=args.n0_prior)
    else:
        run(args.input, args.output, args.chunk_size, args.workers,
            b_prior=args.b_prior, n0_prior=args.n0_prior)
