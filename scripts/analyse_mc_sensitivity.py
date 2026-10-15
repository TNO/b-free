#!/usr/bin/env python
"""
analyse_mc_sensitivity.py

Quantifies how the two b-free formulations respond to an imperfectly known
completeness magnitude.  The two formulations are exposed to it in different
ways, and are therefore tuned against different quantities:

  magnitude-based : requires a reference magnitude m_0, which the analyst
                    chooses.  The requirement on it is one-sided, and applies to
                    the bottom of the catalogue rather than to m_0 itself: bins
                    here are labelled by their lower edge, so that bottom is m_0.
                    (In the manuscript's ESHM20 convention bins are labelled by
                    their centre, and the requirement reads m_0 - delta >= m_c.)
                    Any bottom at or above m_c is unbiased, because the
                    geometric (and in the limit exponential) law is memoryless,
                    so the conditional distribution above it carries the same b.
                    Cutting below m_c admits incomplete data and biases the
                    estimated size distribution; cutting above costs precision
                    rather than introducing bias, as n falls by 10^(-b (m_0-m_c)).

  difference-based: requires no completeness threshold at all, which is the
                    point of the b-positive construction.  It instead requires
                    a minimum retained difference delta_m_th, large enough that
                    the larger event of a retained pair is in the fully-detected
                    range whenever the smaller one is observed.  In these
                    simulations that holds once delta_m_th is comparable to the
                    width of the detection roll-off, but this is an observation
                    here rather than a procedure for real catalogues.

So the magnitude-based formulation has a threshold that must clear m_c, and the
difference-based formulation has a threshold that has to be tuned.  Neither is
free of the completeness problem; they meet it in different places.

Offsets are therefore swept on both sides of m_c: the bias is one-sided, and a
sweep restricted to negative offsets would suggest otherwise.

Catalogues are simulated with a detection roll-off below the completeness
magnitude, so that events below it are generated and then thinned rather than
truncated away.

Usage
-----
    python analyse_mc_sensitivity.py
    python analyse_mc_sensitivity.py --sweep-threshold
"""

import argparse

import numpy as np

from bfree import (
    LOG10,
    bfree_survival_rounded,
    bssa,
    difference_statistics_rounded,
    magnitude_statistics_rounded,
    plotting,
)
from bfree.bssa import LINE_BASE
from bfree.plotting import FORMATS, legend_below_axes, line_kwargs, save_figure, size

MC_TRUE = 0.0            # magnitude at which detection is treated as complete
ROLLOFF = 0.5            # width of the detection roll-off below MC_TRUE
ROLLOFF_SHAPE = "linear"  # "linear" or "normal" (Ogata-Katsura style)
DELTA = 0.05             # half bin width; catalogues rounded to 2*DELTA = 0.1
B_TRUE = 1.0
OFFSETS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)   # assumed m_0 minus MC_TRUE
EVAL_STEPS = (10, 20, 30)             # bins above threshold at which to report

# Minimum positive difference for the b-positive formulation, in bins:
# delta_m_th = K_MIN * 2 * DELTA.  Must be at least the roll-off width; the
# default matches ROLLOFF.
K_MIN = 5


def detection_probability(m, shape=ROLLOFF_SHAPE, width=ROLLOFF, mc=MC_TRUE):
    """
    Probability that an earthquake of magnitude m is detected.

    Completeness is not a step: detection rises over a finite magnitude range.
    Which magnitude is then called "complete" is a convention, and both the
    shape and the width of the roll-off are modelling assumptions - see the
    --shape option.

    linear : rises from 0 at mc - width to 1 at mc.
    normal : cumulative normal centred at mc - width/2 with sigma = width/4,
             reaching about 98% detection at mc.
    """
    if shape == "linear":
        return np.clip((m - (mc - width)) / width, 0.0, 1.0)
    if shape == "normal":
        from scipy.stats import norm
        return norm.cdf(m, loc=mc - width / 2.0, scale=width / 4.0)
    raise ValueError(f"unknown roll-off shape: {shape!r}")


def simulate(n_catalogs, n_target, rng, shape=ROLLOFF_SHAPE, width=ROLLOFF):
    """
    Catalogues with a detection roll-off below MC_TRUE.

    Returns a list of rounded magnitude arrays.  Events are drawn from well
    below MC_TRUE so that the roll-off region is populated.
    """
    beta = B_TRUE * LOG10
    lowest = MC_TRUE - 2.0 * width
    # Draw enough events that the expected number above MC_TRUE is n_target
    # regardless of the roll-off width, so catalogue size does not confound the
    # comparison between widths.
    n_draw = int(n_target * 10.0 ** (B_TRUE * (MC_TRUE - lowest)) * 1.2)
    out = []
    for _ in range(n_catalogs):
        x = rng.exponential(1.0 / beta, size=n_draw) + lowest
        m = np.floor(x / (2 * DELTA)) * (2 * DELTA)
        keep = rng.random(m.shape) < detection_probability(m, shape, width)
        out.append(m[keep])
    return out


def count_comparison(n_catalogs, n_target, seed, shape, widths=(0.2, 0.4, 0.6, 0.8)):
    """
    Number of events each formulation actually uses, as the roll-off widens.

    The magnitude-based formulation uses every event above m_c.  The
    difference-based formulation starts from a larger catalogue, since it also
    uses the events below m_c, but then keeps only non-overlapping pairs whose
    difference is positive and at least delta_m_th.  Tuning requires
    delta_m_th ~ w, so the retained fraction shrinks as the roll-off widens.
    The ceiling is 1/(2(1+q)) with q = 10^(-2*b*delta), reached only for step
    detection; any real roll-off falls below it.
    """
    print(f"\nSample size actually used ({shape} roll-off, "
          r"delta_m_th tuned to w)")
    print(f"{'roll-off w':>11}{'n above m_c':>13}{'n detected':>12}"
          f"{'n_+ used':>11}{'n_+/n':>9}{'predicted':>11}")
    for w in widths:
        cats = simulate(n_catalogs, n_target, np.random.default_rng(seed),
                        shape=shape, width=w)
        k = max(1, round(w / (2 * DELTA)))
        n_all = np.mean([len(m) for m in cats])
        n_c = np.mean([np.sum(m >= MC_TRUE - 1e-9) for m in cats])
        n_d = np.mean([difference_statistics_rounded(m, MC_TRUE, DELTA,
                                             overlapping=False, k_min=k)[0]
                       for m in cats])
        # n_+ = N_detected * 1/2 (pairing) * 1/2 (positive) * 10^(-b delta_m_th)
        predicted = 0.25 * n_all * 10.0 ** (-B_TRUE * k * 2 * DELTA)
        print(f"{w:>11.1f}{n_c:>13.0f}{n_all:>12.0f}{n_d:>11.0f}"
              f"{n_d / n_c:>9.3f}{predicted / n_c:>11.3f}")
    print("The difference-based formulation always uses fewer events, and")
    print("relatively fewer as the roll-off - and hence delta_m_th - grows.\n")


def fit_detection(m):
    """
    Joint maximum-likelihood fit of b and a cumulative-normal detection curve
    to a whole catalogue, incomplete part included, after Ogata and Katsura
    (1993).

    Assuming Gutenberg-Richter throughout, the density of *detected* magnitudes
    is proportional to beta exp(-beta m) Phi((m - mu) / sigma), which is
    normalisable over the whole real line with Z = exp(-beta mu + beta^2
    sigma^2 / 2).  Returns (b, mu, sigma).
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    n = len(m)
    total = m.sum()

    def nll(p):
        """Negative log-likelihood of the Ogata-Katsura detected-magnitude model."""
        log_beta, mu, log_sigma = p
        beta, sigma = np.exp(log_beta), np.exp(log_sigma)
        return -(n * log_beta - beta * total
                 + norm.logcdf((m - mu) / sigma).sum()
                 + n * (beta * mu - 0.5 * beta ** 2 * sigma ** 2))

    res = minimize(nll, [np.log(B_TRUE * LOG10), 0.0, np.log(0.15)],
                   method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
    log_beta, mu, log_sigma = res.x
    return np.exp(log_beta) / LOG10, mu, np.exp(log_sigma)


def detection_inference(n_catalogs, n_target, seed, widths=(0.2, 0.4, 0.6, 0.8)):
    """
    Can the detection curve be inferred from the catalogue itself?

    Fits both roll-off shapes with the cumulative-normal model, so the second
    block is deliberately misspecified.  In the normal parameterisation the
    roll-off width corresponds to 4 sigma.
    """
    print("\nInferring the detection curve from the catalogue "
          "(Ogata-Katsura joint fit)")
    print(f"{'shape':>8}{'w':>6}{'b_hat':>9}{'sd(b)':>8}{'mu_hat':>9}"
          f"{'sigma_hat':>11}{'4*sigma':>9}")
    for shape in ("normal", "linear"):
        for w in widths:
            rng = np.random.default_rng(seed)
            cats = simulate(n_catalogs, n_target, rng, shape=shape, width=w)
            fits = np.array([fit_detection(m + DELTA) for m in cats])
            b, mu, sigma = fits.mean(axis=0)
            print(f"{shape:>8}{w:>6.1f}{b:>9.3f}{fits[:, 0].std():>8.3f}"
                  f"{mu:>9.3f}{sigma:>11.3f}{4 * sigma:>9.2f}")
    print("With the shape correct, 4*sigma recovers the width and b is "
          "unbiased.")
    print("With it misspecified, the width is underestimated and b is biased "
          "low.\n")


def predict_bias(shape=ROLLOFF_SHAPE, width=ROLLOFF, offsets=OFFSETS,
                 step=EVAL_STEPS[-1], n_bins=2000):
    """
    Predict the magnitude-based bias from the detection curve alone.

    Weight the true bin probabilities by the detection probability, renormalise,
    and read off the large-n limit of the estimator, q_hat = Kbar / (1 + Kbar).
    No catalogues are simulated, so agreement with the simulated ratios shows
    that the bias is residual incompleteness and nothing else.

    This is what explains the residual bias at m_0 = m_c for a roll-off that
    only asymptotes to complete detection: there is then no offset at which the
    catalogue is strictly complete, and m_c is a convention rather than a
    property of the data.
    """
    q = 10.0 ** (-2 * DELTA * B_TRUE)
    k = np.arange(n_bins)
    print(f"\nBias predicted from the detection curve alone "
          f"({shape} roll-off of width {width}, ratio at k={step})")
    print(f"true q = {q:.6f}, true mean steps = {q / (1 - q):.4f}")
    print(f"{'m_0 - m_c':>11}{'p at cut':>11}{'missing':>11}{'mean W/n':>11}"
          f"{'q_hat':>10}{'predicted':>11}")
    for off in offsets:
        p = detection_probability(MC_TRUE + off + 2 * DELTA * k, shape, width)
        w = p * (1 - q) * q ** k
        w /= w.sum()
        kbar = float((k * w).sum())
        qhat = kbar / (1.0 + kbar)
        print(f"{off:>11.1f}{p[0]:>11.5f}{1 - p[0]:>11.2e}{kbar:>11.4f}"
              f"{qhat:>10.5f}{(qhat / q) ** step:>11.4f}")
    print("Compare with the simulated ratios below.\n")


def evaluate(catalogs, assumed_m0, steps, k_min=K_MIN):
    """
    Ratio to truth for both formulations.

    The magnitude-based formulation is applied above *assumed_m0*.  The
    difference-based formulation is applied to the whole detected catalogue,
    with no completeness cut, since it does not use one.
    """
    q = 10.0 ** (-2 * DELTA * B_TRUE)
    truth = q ** np.asarray(steps, dtype=float)

    mag, diff, n_mag, n_plus = [], [], [], []
    for m in catalogs:
        if len(m) < 50:
            continue
        kept = m[m >= assumed_m0 - 1e-9]
        if len(kept) >= 50:
            n, _, w = magnitude_statistics_rounded(kept, assumed_m0, DELTA)
            mag.append(bfree_survival_rounded(w, n, np.asarray(steps))[0])
            n_mag.append(n)

        # No m_0 anywhere here: the reference merely fixes the bin origin and
        # cancels when the differences are taken.
        n_d, _, w_d = difference_statistics_rounded(m, MC_TRUE, DELTA,
                                            overlapping=False, k_min=k_min)
        if n_d > 10:
            diff.append(bfree_survival_rounded(w_d, n_d, np.asarray(steps))[0])
            n_plus.append(n_d)

    if not mag or not diff:
        raise RuntimeError(f"no usable catalogues at m_0 = {assumed_m0}")
    return (np.mean(mag, axis=0) / truth, np.mean(diff, axis=0) / truth,
            float(np.mean(n_mag)), float(np.mean(n_plus)))


def run(n_catalogs, n_target, figure_dir, formats, seed=0, k_min=K_MIN,
        sweep_threshold=False, shape=ROLLOFF_SHAPE, counts=False,
        fit_curve=False, predict=False, with_differences=False):
    """Measure how each formulation responds to the placement of the reference
    magnitude relative to completeness.

    The requirement is one-sided: bias appears only below the completeness
    magnitude, while above it the catalogue simply shrinks.
    """
    rng = np.random.default_rng(seed)
    print(f"Simulating {n_catalogs} catalogues "
          f"(m_c = {MC_TRUE}, {shape} roll-off of width {ROLLOFF}) ...")
    catalogs = simulate(n_catalogs, n_target, rng, shape=shape)

    if predict:
        predict_bias(shape=shape)

    if sweep_threshold:
        print("\nb-positive difference threshold against roll-off width")
        print(f"({shape} roll-off; ratio at k={EVAL_STEPS[-1]}; "
              "no completeness cut is applied)")
        kmins = (1, 2, 4, 6, 8)
        print(f"{'roll-off w':>11}" + "".join(
            f"{f'dm={k * 2 * DELTA:.1f}':>9}" for k in kmins))
        for w in (0.2, 0.4, 0.6, 0.8):
            cats_w = simulate(n_catalogs, n_target,
                              np.random.default_rng(seed), shape=shape, width=w)
            cells = ""
            for k in kmins:
                _, r_diff, _, _ = evaluate(cats_w, MC_TRUE, EVAL_STEPS, k_min=k)
                cells += f"{r_diff[-1]:>9.3f}"
            print(f"{w:>11.1f}{cells}")
        print("\nThe bias vanishes once dm_th becomes comparable to the "
              "roll-off width.\n")

    if counts:
        count_comparison(n_catalogs, n_target, seed, shape)

    if fit_curve:
        detection_inference(n_catalogs, n_target, seed)

    results = {}
    print(f"\n{'assumed m_0':>12}{'m_0 - m_c':>11}"
          + "".join(f"{f'mag k={k}':>12}" for k in EVAL_STEPS)
          + "".join(f"{f'diff k={k}':>12}" for k in EVAL_STEPS)
          + f"{'n (mag)':>10}{'n_+ (diff)':>12}")
    for off in OFFSETS:
        assumed = MC_TRUE + off
        r_mag, r_diff, n_m, n_plus = evaluate(catalogs, assumed, EVAL_STEPS,
                                              k_min=k_min)
        results[off] = (r_mag, r_diff)
        print(f"{assumed:>12.1f}{off:>11.1f}"
              + "".join(f"{v:>12.3f}" for v in r_mag)
              + "".join(f"{v:>12.3f}" for v in r_diff)
              + f"{n_m:>10.0f}{n_plus:>12.0f}")
    print("\nThe requirement on m_0 is one-sided: the bias appears only below "
          "m_c,\nwhile above it n falls by 10^(-b (m_0 - m_c)) and nothing "
          "else changes.\n")

    _plot(results, figure_dir, formats, k_min, with_differences)
    return results


def _plot(results, figure_dir, formats, k_min=K_MIN, with_differences=False):
    """Draw the completeness-offset sensitivity figure."""
    import matplotlib.pyplot as plt

    offsets = sorted(results)
    x = np.arange(len(offsets))
    fig, axs = plt.subplots(
        1, len(EVAL_STEPS), sharey=True,
        figsize=size(0.42, original=(4.2 * len(EVAL_STEPS), 4.0)),
    )
    for si, ax in enumerate(axs):
        mag = [results[o][0][si] for o in offsets]
        diff = [results[o][1][si] for o in offsets]
        # Only offsets below m_c admit incomplete data; the rest is a free choice.
        edge = float(np.interp(0.0, offsets, x))
        ax.axvspan(x[0] - 0.5, edge, color="0.9", zorder=0)
        ax.axhline(1.0, color="0.4", lw=LINE_BASE, ls="--", zorder=1)
        # Both curves are b-free; it is the formulation that differs, which is
        # the distinction the framework styles carry elsewhere. The markers are
        # set here rather than taken from the framework, because the
        # magnitude-based one carries none by default and both are wanted.
        ax.plot(x, mag, zorder=3, label="magnitude-based",
                **line_kwargs("bfree", "mag", lw=LINE_BASE,
                              marker="o", markersize=3, markevery=1))
        if with_differences:
            ax.plot(x, diff, zorder=3,
                    label=rf"difference-based ($\Delta m_{{\rm th}}={k_min * 2 * DELTA:.1f}$)",
                    **line_kwargs("bfree", "diff_nol", lw=LINE_BASE,
                                  marker="s", markersize=3, markevery=1))
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        # Every other offset: seven labels collide at the printed panel width.
        ax.set_xticks(x[::2])
        ax.set_xticklabels([f"{offsets[i]:+.1f}" for i in range(0, len(offsets), 2)])
        ax.set_xlabel(r"$m_0 - m_c$")
        ax.set_title(rf"$m - m_0 = {EVAL_STEPS[si] * 2 * DELTA:.1f}$")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel(r"$\hat{S}(m)\,/\,S_{\mathrm{true}}(m)$")
    axs[0].annotate("incomplete\ndata admitted", xy=(0.02, 0.97),
                    xycoords="axes fraction", va="top", color="0.35")

    if with_differences:
        legend_below_axes(fig, axs)
    # Simulated, so there is no input file to hash.
    for path in save_figure(fig, "reference_magnitude_sensitivity", figure_dir, formats):
        print(f"saved {path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Sensitivity of the b-free estimators to the placement of "
                    "m_0 relative to the completeness magnitude.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-catalogs", type=int, default=3000)
    parser.add_argument("--n-target", type=int, default=400,
                        help="Approximate catalogue size above the true m_c.")
    # No default: the destination belongs to the caller, not to this script.
    parser.add_argument("--figure-dir", required=True,
                        help="directory to write the figure to")
    parser.add_argument("--format", nargs="+", default=["png"], choices=FORMATS,
                        help="one or more of: " + ", ".join(FORMATS))
    parser.add_argument("--style", choices=("original", "screen", "bssa"),
                        default="bssa",
                        help="as in make_figures.py; original draws at the "
                             "pre-page size, bssa at the printed page width")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k-min", type=int, default=K_MIN,
                        help="Minimum positive difference in bins; "
                             "delta_m_th = k_min * 2 * delta.")
    parser.add_argument("--sweep-threshold", action="store_true",
                        help="Also report the difference-threshold sweep.")
    parser.add_argument("--shape", default=ROLLOFF_SHAPE,
                        choices=("linear", "normal"),
                        help="Shape of the detection roll-off.")
    parser.add_argument("--counts", action="store_true",
                        help="Also report the sample size each formulation uses.")
    parser.add_argument("--fit-detection", action="store_true",
                        help="Also try to infer the detection curve from the "
                             "catalogue itself.")
    parser.add_argument("--predict-bias", action="store_true",
                        help="Also predict the magnitude-based bias from the "
                             "detection curve alone, without simulating.")
    # Off by default: the manuscript defers the difference-based comparison.
    parser.add_argument("--with-differences", action="store_true",
                        help="Also draw the difference-based series.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.style == "bssa":
        bssa.apply()
        plotting.TITLES_IN_FIGURES = False
    elif args.style == "original":
        plotting.apply_sans_fonts()
        plotting.ORIGINAL_SIZES = True
    run(args.n_catalogs, args.n_target, args.figure_dir, args.format, args.seed,
        k_min=args.k_min, sweep_threshold=args.sweep_threshold,
        shape=args.shape, counts=args.counts, fit_curve=args.fit_detection,
        predict=args.predict_bias, with_differences=args.with_differences)
