#!/usr/bin/env python
"""
run_simulations.py

Runs the full synthetic pipeline in one command:
  1. Generates synthetic earthquake catalogues in parallel -> results/simulation_results.zarr
  2. Reduces them to the plotted summary statistics       -> results/analysis_results.zarr.zip

Only the sufficient statistics of each catalogue are stored; the individual
magnitudes are discarded.

Usage
-----
    python run_simulations.py                           # default settings
    python run_simulations.py --nr_simulations 1e7 --workers 32
    python run_simulations.py --skip-analysis           # simulations only
    python run_simulations.py --help                    # show all options
"""

import argparse
import time
from collections import deque
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import zarr

# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def generate_catalogs(n_catalogs, n_events, b_true, mc=0, delta=0.05, k_min=1):
    """
    Generate *n_catalogs* synthetic catalogues and compute sufficient statistics
    for the magnitude framework and for both difference frameworks.

    All catalogues are produced in a single vectorised pass — no Python loop over
    individual catalogues.  Each catalogue draws *n_events* independent exponential
    samples, so there is no sharing of random numbers across catalogues or within
    a single catalogue.

    Returns a dict of arrays with keys:
        t_mag, w_mag,
        n_diff_nol, t_diff_nol, w_diff_nol,   (non-overlapping pairs)
        n_diff_ol,  t_diff_ol,  w_diff_ol      (overlapping pairs)
    """
    beta_true = b_true * np.log(10)
    bin_width = 2 * delta
    m_physical_start = mc - delta
    rng = np.random.default_rng()

    # --- Draw all samples at once: shape (n_catalogs, n_events) ---
    # Each row is one independent catalog; each column is one independent event.
    m_continuous = rng.exponential(1.0 / beta_true,
                                   (n_catalogs, n_events)) + m_physical_start

    # Round to bin centres
    m_obs = np.round((m_continuous - mc) / bin_width) * bin_width + mc

    # --- Magnitude framework ---
    t_mag = (m_obs - (mc - delta)).sum(axis=1)                           # (n_catalogs,)
    k_indices = np.round((m_obs - mc) / bin_width).astype(np.int64)     # (n_catalogs, n_events)
    w_mag = k_indices.sum(axis=1)                                        # (n_catalogs,)

    # --- Non-overlapping pairs (even-indexed vs odd-indexed events) ---
    k_i_nol = k_indices[:, 0::2]
    k_j_nol = k_indices[:, 1::2]
    nr_nol = min(k_i_nol.shape[1], k_j_nol.shape[1])
    diffs_nol = k_j_nol[:, :nr_nol] - k_i_nol[:, :nr_nol]
    mask_nol = diffs_nol >= k_min
    n_diff_nol = mask_nol.sum(axis=1)
    # Excess above the threshold; identical to (diffs*bin_width - delta) at
    # k_min = 1, and correct above it.
    t_diff_nol = np.where(mask_nol, (diffs_nol - k_min) * bin_width + delta, 0.0).sum(axis=1)
    w_diff_nol = np.where(mask_nol, diffs_nol - k_min, 0).sum(axis=1)

    # --- Overlapping pairs (consecutive events) ---
    diffs_ol = k_indices[:, 1:] - k_indices[:, :-1]
    mask_ol = diffs_ol >= k_min
    n_diff_ol = mask_ol.sum(axis=1)
    t_diff_ol = np.where(mask_ol, (diffs_ol - k_min) * bin_width + delta, 0.0).sum(axis=1)
    w_diff_ol = np.where(mask_ol, diffs_ol - k_min, 0).sum(axis=1)

    return {
        "t_mag":      t_mag,
        "w_mag":      w_mag,
        "n_diff_nol": n_diff_nol,
        "t_diff_nol": t_diff_nol,
        "w_diff_nol": w_diff_nol,
        "n_diff_ol":  n_diff_ol,
        "t_diff_ol":  t_diff_ol,
        "w_diff_ol":  w_diff_ol,
    }


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Unpacking wrapper so Pool.imap can call generate_catalogs."""
    chunk_size, n_events, b_true, mc, delta = args
    return (b_true, n_events), generate_catalogs(
        n_catalogs=chunk_size,
        n_events=n_events,
        b_true=b_true,
        mc=mc,
        delta=delta,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run(nr_simulations, b_true_range, cat_sizes, chunk_size, n_workers, output,
        mc=0.0, delta=0.05):
    """Build the task list, run in parallel, merge, and save."""

    b_true_range = np.asarray(b_true_range)
    cat_sizes = np.asarray(cat_sizes, dtype=int)

    # Auto-size chunks so each chunk uses ~100 MB of array memory.
    # With the vectorised kernel the dominant allocation is roughly
    # 3 × n_catalogs × n_events × 8 bytes (float64 + int64 intermediates).
    if chunk_size is None:
        chunk_size = max(1_000, int(1e8 / (24 * int(cat_sizes.max()))))

    # Build flat list of chunks: one entry per (b_true, n_events, chunk_idx)
    tasks = []
    for b_true in b_true_range:
        for n_events in cat_sizes:
            remaining = nr_simulations
            while remaining > 0:
                this_chunk = min(chunk_size, remaining)
                tasks.append((this_chunk, n_events, b_true, mc, delta))
                remaining -= this_chunk

    total_chunks = len(tasks)
    combos = [(b, n) for b in b_true_range for n in cat_sizes]

    print(
        f"Simulations : {nr_simulations:,} per (b_true, cat_size) combination\n"
        f"Combinations: {len(combos)}\n"
        f"Workers     : {n_workers}\n"
        f"Chunk size  : {chunk_size:,}\n"
        f"Total chunks: {total_chunks:,}\n"
        f"Output      : {output}\n"
    )

    _ZARR_COLS = [
        't_mag', 'w_mag',
        'n_diff_nol', 't_diff_nol', 'w_diff_nol',
        'n_diff_ol',  't_diff_ol',  'w_diff_ol',
    ]

    # Accumulate chunks as raw numpy arrays — lighter than DataFrames and
    # avoids any zarr I/O on the hot path so workers are never throttled.
    partial: dict[tuple, list[dict]] = {key: [] for key in combos}

    completed = 0
    t_start = time.perf_counter()
    TS_WINDOW = max(50, n_workers * 4)
    completion_ts: deque[float] = deque(maxlen=TS_WINDOW)
    ETA_ALPHA = 0.05
    smoothed_eta: float | None = None

    with Pool(processes=n_workers) as pool:
        for key, chunk in pool.imap_unordered(_worker, tasks):
            partial[key].append(chunk)
            completed += 1
            completion_ts.append(time.perf_counter())

            remaining_chunks = total_chunks - completed

            if len(completion_ts) >= 2:
                window_span = completion_ts[-1] - completion_ts[0]
                rate = (len(completion_ts) - 1) / max(window_span, 1e-6)
                raw_eta = remaining_chunks / rate
                smoothed_eta = (
                    raw_eta if smoothed_eta is None
                    else ETA_ALPHA * raw_eta + (1 - ETA_ALPHA) * smoothed_eta
                )
            eta_s = smoothed_eta if smoothed_eta is not None else float("inf")

            elapsed_str = _fmt_seconds(completion_ts[-1] - t_start)
            eta_str = _fmt_seconds(eta_s)
            pct = 100.0 * completed / total_chunks
            print(
                f"\r  [{completed:>{len(str(total_chunks))}}/{total_chunks}]"
                f"  {pct:5.1f}%"
                f"  elapsed {elapsed_str}"
                f"  ETA {eta_str}   ",
                end="",
                flush=True,
            )

    print()  # newline after progress line

    # Bulk-write to zarr — one concatenated array per column per combination.
    # Doing this after the pool closes keeps all disk I/O off the hot path.
    print(f"Writing zarr store to {output} …")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(output, mode='w')
    for key in combos:
        b_val, n_val = key
        g = store.require_group(f"b{float(b_val)}_n{int(n_val)}")
        chunks = partial.pop(key)   # free memory as we go
        for col in _ZARR_COLS:
            data = np.concatenate([c[col] for c in chunks])
            g.zeros(name=col, shape=data.shape, dtype=data.dtype,
                    chunks=(100_000,), overwrite=True)[:] = data
        g.attrs.update({
            'b_true': float(b_val), 'mc': mc, 'delta': delta,
            'n_events': int(n_val), 'n_catalogs': nr_simulations,
        })

    total_elapsed = _fmt_seconds(time.perf_counter() - t_start)
    print(f"Simulations done.  Total time: {total_elapsed}")


def _fmt_seconds(s: float) -> str:
    """Format a duration in seconds as H:MM:SS or MM:SS."""
    if not np.isfinite(s):
        return "??:??"
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: simulate catalogs then compute analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nr_simulations", type=lambda x: int(float(x)), default=10_000_000,
        metavar="N",
        help="Number of synthetic catalogs per (b_true, cat_size) combination. "
             "Scientific notation accepted, e.g. 1e7.",
    )
    parser.add_argument(
        "--b_true_range", type=float, nargs="+",
        default=[0.8, 1.0, 1.2],
        metavar="B",
        help="Space-separated list of true b-values to simulate.",
    )
    parser.add_argument(
        "--cat_sizes", type=int, nargs="+",
        default=[50, 100, 500, 1000],
        metavar="N",
        help="Space-separated list of catalog sizes (number of events).",
    )
    parser.add_argument(
        "--chunk_size", type=lambda x: int(float(x)), default=None,
        metavar="C",
        help="Number of catalogs per parallel work unit.  Omit to let the "
             "script choose automatically based on max(cat_sizes) so that "
             "each chunk uses ~100 MB of array memory.",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, cpu_count() - 1),
        metavar="W",
        help="Number of parallel worker processes. "
             "Defaults to (CPU count - 1).",
    )
    parser.add_argument(
        "--output", type=str, default="results/simulation_results.zarr",
        metavar="FILE",
        help="Path for the simulation results zarr store.",
    )
    parser.add_argument(
        "--analysis-output", type=str, default="results/analysis_results.zarr.zip",
        metavar="FILE",
        help="Path for the reduced analysis results.",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Save simulation results and exit without running the analysis step.",
    )
    parser.add_argument(
        "--mc", type=float, default=0.0,
        help="Completeness magnitude (centre of first bin).",
    )
    parser.add_argument(
        "--delta", type=float, default=0.05,
        help="Half-bin width (rounding precision).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        nr_simulations=args.nr_simulations,
        b_true_range=args.b_true_range,
        cat_sizes=args.cat_sizes,
        chunk_size=args.chunk_size,
        n_workers=args.workers,
        output=args.output,
        mc=args.mc,
        delta=args.delta,
    )
    if not args.skip_analysis:
        import reduce_simulations
        print()
        reduce_simulations.run(
            input_path=args.output,
            output_path=args.analysis_output,
            chunk_size=100_000,
            n_workers=args.workers,
        )
