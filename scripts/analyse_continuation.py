#!/usr/bin/env python
"""
analyse_continuation.py

Measures the bias of the continued rounded b-free estimators, which are the
rounded estimators with the integer bin index replaced by a real coordinate

    u = (m - m0* + delta) / 2*delta,

so that integer u is a bin edge and half-integer u a bin centre.

Expectations are evaluated by deterministic summation over the negative
binomial law of W, not by simulation: the biases at the catalogue sizes the
manuscript uses are ~1e-13, which no feasible Monte Carlo run can resolve.

The summand is `bfree_survival_rounded` and `bfree_pdf_rounded` themselves, so the table this
script prints describes the estimators as implemented rather than a separate
transcription of them.  That matters here, because the two differ in the one
place the support is subtle: the continued survival function is positive on
(W, W+1) and zero only from W+1 onward, so a term at w = ceil(u) - 1 enters the
expectation whenever u is not an integer.  Truncating the sum at w >= ceil(u)
instead reports a bias of the wrong sign.
"""
import argparse

import numpy as np
from scipy.stats import nbinom

from bfree import bfree_pdf_rounded, bfree_survival_rounded
from bfree.store import load as load_results

LOG10 = np.log(10.0)

# Largest bin coordinate each analysis evaluates: the synthetic density grid runs
# to m - m0 = 5 at 2*delta = 0.1, and the ESHM20 curves stop one magnitude unit
# above m_250 at 2*delta = 0.2.
SYNTHETIC_U_MAX = 51.0
ESHM20_U_MAX = 30.0


def expectations(u, n, b, two_delta, tail_tol=1e-18):
    """
    E[S*(u)] and E[f*(u)] under W ~ NegBin(n, 1-q), with their targets.

    The sum runs over every w with non-negligible probability; the support of
    the estimator is imposed by `bfree_survival_rounded` and `bfree_pdf_rounded`, which return
    zero from w+1 onward.
    """
    delta = two_delta / 2.0
    q = 10.0 ** (-two_delta * b)
    w_max = int(nbinom.isf(tail_tol, n, 1 - q)) + 1
    w = np.arange(w_max + 1)
    pmf = nbinom.pmf(w, n, 1 - q)
    tail = float(nbinom.sf(w_max, n, 1 - q))
    if tail > tail_tol:
        raise ValueError(f"truncated {tail:.3e} of the mass at w_max={w_max}")

    u = np.atleast_1d(float(u))
    e_sf = float(pmf @ bfree_survival_rounded(w, n, u)[:, 0])
    e_pdf = float(pmf @ bfree_pdf_rounded(w, n, u, delta)[:, 0])
    return e_sf, e_pdf, float(q ** u[0]), float(-np.log(q) * q ** u[0] / (2 * delta))


def relative_bias(u, n, b, two_delta):
    """Relative bias of the continued survival function and density at level *u*."""
    e_sf, e_pdf, sf_true, pdf_true = expectations(u, n, b, two_delta)
    return e_sf / sf_true - 1.0, e_pdf / pdf_true - 1.0


def bias_table(n_values, b, two_delta, u_edge, u_centre):
    """Rows of (n, sf and pdf bias at a bin edge, the same at a bin centre).

    The edge is where the rounded estimator is exact, so a non-zero entry there
    is numerical noise; the centre is where the continuation does its work.
    """
    return [(n, *relative_bias(u_edge, n, b, two_delta),
             *relative_bias(u_centre, n, b, two_delta)) for n in n_values]


def worst_over_range(n, b, two_delta, u_max, step=0.05):
    """Largest relative departure of either object over 0 < u < u_max.

    A finite mesh samples the interval rather than proving a bound over it, and
    *u_max* is the largest coordinate the corresponding analysis evaluates: the
    departure keeps growing beyond it, where the survival function itself is
    below 1e-15 and the gamma ratios lose relative precision.
    """
    worst = (0.0, 0.0, np.nan)
    for u in np.arange(step, u_max, step):
        rb_sf, rb_pdf = relative_bias(u, n, b, two_delta)
        if max(abs(rb_sf), abs(rb_pdf)) > max(abs(worst[0]), abs(worst[1])):
            worst = (rb_sf, rb_pdf, u)
    return worst


def eshm20_settings(path="results/eshm20_results.zarr.zip"):
    """(b, bin width, n) of the qualifying ESHM20 zones.

    The zones are scanned as they are rather than through a stand-in, since the
    point of the scan is to bound the departure at the settings actually used.
    Missing results are an error rather than an empty list: silently scanning
    twelve synthetic settings and reporting it as full coverage is exactly the
    kind of claim this script exists to avoid.
    """
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found; run analyse_eshm20.py first, or pass "
            "--no-eshm20 to scan only the synthetic settings")
    zones = load_results(path)["zones"]
    return [(float(r["b_aki"]), float(2 * r["delta"]), int(r["n"]), ESHM20_U_MAX)
            for _, r in zones.iterrows()]


def fmt(x, digits=1):
    """Relative bias in the manuscript's format: exact zero stays a bare 0."""
    if x == 0.0:
        return "$0$"
    mant, exp = f"{abs(x):.{digits}e}".split("e")
    sign = "-" if x < 0 else "+"
    return f"${sign}{mant}\\times10^{{{int(exp)}}}$"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b", type=float, default=1.0)
    p.add_argument("--bin-width", type=float, default=0.1, dest="two_delta")
    p.add_argument("--n", type=int, nargs="+", default=[2, 3, 5, 8, 12, 20])
    p.add_argument("--u-edge", type=float, default=2.0)
    p.add_argument("--u-centre", type=float, default=2.5)
    p.add_argument("--scan", action="store_true",
                   help="also report the largest departure over 0 < u < 60 at "
                        "the catalogue sizes and bin widths the manuscript uses")
    p.add_argument("--no-eshm20", action="store_true",
                   help="scan only the synthetic settings, without the zones")
    args = p.parse_args()

    rows = bias_table(args.n, args.b, args.two_delta, args.u_edge, args.u_centre)

    print(f"b = {args.b}, 2 delta = {args.two_delta}, "
          f"edge u = {args.u_edge}, centre u = {args.u_centre}\n")
    print(f"{'n':>4} {'edge S*':>12} {'edge f*':>12} {'centre S*':>12} {'centre f*':>12}")
    for n, s_e, f_e, s_c, f_c in rows:
        print(f"{n:>4} {s_e:>12.3e} {f_e:>12.3e} {s_c:>12.3e} {f_c:>12.3e}")

    print("\nLaTeX table body:\n")
    for n, s_e, f_e, s_c, f_c in rows:
        print(f"{n:<3}& {fmt(s_e):<22}& {fmt(f_e):<22}& "
              f"{fmt(s_c):<22}& {fmt(f_c):<22}\\\\")

    if args.scan:
        settings = [(b, 0.1, n, SYNTHETIC_U_MAX) for b in (0.8, 1.0, 1.2)
                    for n in (50, 100, 500, 1000)]
        if not args.no_eshm20:
            settings += eshm20_settings()
        print(f"\nLargest departure over the evaluated coordinate range, "
              f"{len(settings)} settings:\n")
        worst = 0.0
        for b, two_delta, n, u_max in settings:
            rb_sf, rb_pdf, u = worst_over_range(n, b, two_delta, u_max)
            worst = max(worst, abs(rb_sf), abs(rb_pdf))
            print(f"  b={b:.4f} 2delta={two_delta} n={n:>4} u<{u_max:4.1f}: "
                  f"S* {rb_sf:+.2e}, f* {rb_pdf:+.2e} (at u={u:.2f})")
        print(f"\n  largest over the scan: {worst:.2e}")


if __name__ == "__main__":
    main()
