#!/usr/bin/env python
"""
analyse_seismostats.py

Checks against SeismoStats (Mirwald et al., 2026), a community Python package
for statistical seismology, and against the pairing of magnitude differences it
uses for its b-more-positive estimator.

  --identity     bfree.b_grouped and bfree.b_aki reproduce the published values of
                 SeismoStats' ClassicBValueEstimator and UtsuBValueEstimator.
                 The two packages implement the same two estimators, so this is
                 a regression check against an independent implementation
                 rather than an experiment. Exit status is 1 on mismatch, so it
                 can gate a build. The same check runs under pytest.

  --next-larger  Bias and spread of the grouped b-value estimate under the three
                 pairings of magnitude differences, measured on the same
                 catalogues, together with the factor by which the Shi and Bolt
                 (1982) standard deviation understates the truth for the
                 next-larger pairing.

The three pairings, at a threshold of one bin:

  alternate     m_2i - m_2i-1        independent, half the data
  successive    m_i+1 - m_i          each magnitude enters two differences
  next-larger   m_nu(i) - m_i        nu(i) the first later event at least dmc
                                     larger, so one large event partners a whole
                                     run of its predecessors

The first two are Tinti and Gasperini (2024) equations 23 and 22; the third is
the pairing of Lippiello and Petrillo (2024), which they do not treat.

Simulation is exact: under the Gutenberg-Richter law with bins of width
2*delta, the bin index of an event is geometric on {0,1,2,...} with ratio
q = 10^(-2 delta b), so catalogues are drawn as geometric variates directly.

Usage:
    python scripts/analyse_seismostats.py --identity
    python scripts/analyse_seismostats.py --next-larger [--n-rep N]
"""
import argparse
import math
import sys

import numpy as np

import bfree

LOG10 = np.log(10.0)

# Magnitudes, mc and bin size of the worked example in the docstrings of
# seismostats/analysis/bvalue/{classic,utsu}.py, with the b-values those
# docstrings report. Taken from the repository at commit 3ab5679.
SEISMOSTATS_MAGNITUDES = np.array([
    2.0, 2.5, 2.1, 2.2, 2.5, 2.2, 2.6, 2.3, 2.7, 2.2, 2.4, 2.0,
    2.7, 2.2, 2.3, 2.1, 2.4, 2.6, 2.2, 2.2, 2.7, 2.4, 2.2, 2.5,
])
SEISMOSTATS_MC = 2.0
SEISMOSTATS_DELTA_M = 0.1
SEISMOSTATS_CLASSIC = 1.114920128810535
SEISMOSTATS_UTSU = 1.1088369750721319


def run_identity(rel_tol=1e-14):
    """Compare bfree.b_grouped and bfree.b_aki with the values SeismoStats publishes."""
    delta = SEISMOSTATS_DELTA_M / 2
    n, t, _ = bfree.magnitude_statistics_rounded(SEISMOSTATS_MAGNITUDES, SEISMOSTATS_MC, delta)

    cases = (
        ("ClassicBValueEstimator", "bfree.b_grouped",
         SEISMOSTATS_CLASSIC, bfree.b_grouped(t, n, delta)),
        ("UtsuBValueEstimator", "bfree.b_aki",
         SEISMOSTATS_UTSU, bfree.b_aki(t, n)),
    )

    print(f"SeismoStats worked example: {n} magnitudes, mc = {SEISMOSTATS_MC}, "
          f"bin size {SEISMOSTATS_DELTA_M}\n")
    print(f"{'SeismoStats':<24}{'ours':<16}{'published':>20}{'ours':>20}{'rel. diff':>12}")
    failed = []
    for their_name, our_name, theirs, ours in cases:
        rel = abs(ours - theirs) / abs(theirs)
        print(f"{their_name:<24}{our_name:<16}{theirs:>20.16f}{ours:>20.16f}{rel:>12.1e}")
        if not math.isclose(ours, theirs, rel_tol=rel_tol):
            failed.append(their_name)

    if failed:
        print(f"\nMISMATCH against {', '.join(failed)} at rel_tol={rel_tol:g}.")
        print("bfree.py and SeismoStats implement the same two estimators; a "
              "difference here means one of them has changed.")
        return 1
    print(f"\nboth estimators agree with SeismoStats to rel_tol={rel_tol:g}")
    return 0


def next_larger_gaps(k, tau):
    """Bin gaps to the first later event at least *tau* bins larger.

    Events with no such successor contribute no difference, which is what
    SeismoStats' find_next_larger does with its zero sentinel.
    """
    out = []
    for i in range(len(k) - 1):
        for j in range(i + 1, len(k)):
            gap = k[j] - k[i]
            if gap >= tau:
                out.append(gap)
                break
    return np.asarray(out, dtype=np.int64)


def _shi_bolt(differences, b):
    """Shi and Bolt (1982), as SeismoStats computes it: population sd over sqrt(N-1)."""
    spread = np.sqrt(np.mean((differences - np.mean(differences)) ** 2))
    return LOG10 * b ** 2 * spread / np.sqrt(len(differences) - 1)


def run_next_larger(b_true, delta, n_rep, seed, tau=1):
    """Bias and spread of the b-value estimate under the three pairings."""
    rng = np.random.default_rng(seed)
    two_delta = 2 * delta
    q = 10.0 ** (-two_delta * b_true)

    def grouped(gaps):
        """The grouped MLE on difference gaps, measured from the lowest retained class."""
        mean_excess = np.mean(gaps - tau) * two_delta
        return np.log(1 + two_delta / mean_excess) / (LOG10 * two_delta)

    print(f"b = {b_true}, 2 delta = {two_delta}, dmc = {tau} bin, "
          f"{n_rep:,} complete catalogues per cell")
    print("quantity is the grouped b-value estimate, not the b-free estimator\n")
    print(f"{'n_cat':>7}{'pairing':>14}{'pairs':>9}{'mean':>9}{'bias %':>9}"
          f"{'sd':>9}{'RMSE':>9}{'ShiBolt/sd':>12}")

    for n in (200, 1000):
        k = rng.geometric(1 - q, size=(n_rep, n)) - 1
        magnitudes = k * two_delta          # a lattice with its bottom bin at 0
        rows = []
        for label in ("alternate", "successive", "next-larger"):
            estimates, counts, shi_bolt = [], [], []
            for i in range(n_rep):
                if label == "next-larger":
                    gaps = next_larger_gaps(k[i], tau)
                    if len(gaps) < 2:
                        continue
                    b_hat, r = grouped(gaps), len(gaps)
                else:
                    r, t, _ = bfree.difference_statistics_rounded(
                        magnitudes[i], 0.0, delta,
                        overlapping=(label == "successive"), k_min=tau)
                    if r < 2:
                        continue
                    b_hat = bfree.b_grouped(t, r, delta)
                    gaps = None
                if not np.isfinite(b_hat):
                    continue
                estimates.append(b_hat)
                counts.append(r)
                # Shi and Bolt needs the differences themselves, which only the
                # next-larger branch has to hand; it is the only case in doubt.
                if gaps is not None and i < 1500:
                    shi_bolt.append(_shi_bolt(gaps * two_delta, b_hat))
            estimates = np.asarray(estimates)
            rows.append((label, float(np.mean(counts)), estimates,
                         float(np.mean(shi_bolt)) if shi_bolt else None))

        for label, pairs, estimates, shi_bolt in rows:
            sd = estimates.std(ddof=1)
            rmse = float(np.sqrt(np.mean((estimates - b_true) ** 2)))
            ratio = f"{shi_bolt / sd:>12.3f}" if shi_bolt else f"{'':>12}"
            print(f"{n:>7}{label:>14}{pairs:>9.1f}{estimates.mean():>9.4f}"
                  f"{100 * (estimates.mean() / b_true - 1):>9.2f}{sd:>9.4f}{rmse:>9.4f}{ratio}")

        successive, next_larger = rows[1], rows[2]
        print(f"{'':7}next-larger uses "
              f"{next_larger[1] / successive[1]:.2f}x the pairs of successive and has "
              f"{next_larger[2].std(ddof=1) / successive[2].std(ddof=1):.2f}x its spread\n")

    print("Shi and Bolt treats the differences as an independent sample, so the "
          "column above is 1 / sqrt(variance inflation).")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Checks against SeismoStats and against the next-larger pairing.")
    parser.add_argument("--identity", action="store_true",
                        help="check bfree.b_grouped and bfree.b_aki against the "
                             "values SeismoStats publishes; exit 1 on mismatch")
    parser.add_argument("--next-larger", action="store_true",
                        help="bias and spread of the b-value estimate under the "
                             "three pairings of magnitude differences")
    parser.add_argument("--b-true", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.05,
                        help="half the bin size; 0.05 gives bins of 0.1")
    parser.add_argument("--n-rep", type=int, default=40_000,
                        help="catalogues per cell for --next-larger")
    parser.add_argument("--seed", type=int, default=31415)
    return parser.parse_args()


def main():
    args = _parse_args()
    if not (args.identity or args.next_larger):
        print("nothing to do: pass --identity or --next-larger")
        return 1
    status = 0
    if args.identity:
        status |= run_identity()
    if args.next_larger:
        if args.identity:
            print()
        run_next_larger(args.b_true, args.delta, args.n_rep, args.seed)
    return status


if __name__ == "__main__":
    sys.exit(main())
