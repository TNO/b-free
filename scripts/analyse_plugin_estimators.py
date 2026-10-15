#!/usr/bin/env python
"""
analyse_plugin_estimators.py

Checks the combined plug-in b-value estimator, which is this manuscript's own
rather than a published one: the grouped maximum likelihood estimator of Tinti
and Mulargia (1987), multiplied by the finite-size factor (n-1)/n that Ogata and
Yamashina (1986) derived for the unrounded case.

The combination is worth making only if it improves on the grouped estimator it
is built from, so this script measures the bias and RMSE of four estimators over
the same simulated catalogues:

  Aki-Utsu   n / (T ln10)             corrects for neither effect
  Ogata      (n-1) / (T ln10)         finite size only
  Tinti      grouped MLE              rounding only
  Combined   (n-1)/n * grouped MLE    both

Simulation is exact rather than approximate.  Under the Gutenberg-Richter law
with bins of width 2*delta, the bin index k of an event is geometric on
{0,1,2,...} with ratio q = 10^(-2 delta b), so catalogues are drawn directly as
geometric variates and no continuous magnitudes are needed.  With T defined as
in the manuscript, T = 2 delta sum(k) + n delta, so that T/n - delta = 2 delta kbar
is the mean excess above m_0, which is what the grouped MLE takes as its
argument.

Note that the combined estimator returns an infinite value when every event
falls in the lowest bin, so its expectation is strictly undefined; that has
probability (1-q)^n, which is below 1e-13 for the smallest catalogue considered
here and never occurred.  The degenerate fraction is reported so this stays
visible rather than assumed.
"""
import argparse

import numpy as np

LOG10 = np.log(10.0)

def estimates(k, delta):
    """The four plug-in estimators for catalogues given as bin indices."""
    n = k.shape[1]
    kbar = k.mean(axis=1)
    t = 2 * delta * k.sum(axis=1) + n * delta

    with np.errstate(divide="ignore"):
        grouped = np.log(1 + 1 / kbar) / (LOG10 * 2 * delta)
    return {
        "Aki-Utsu": n / (t * LOG10),
        "Ogata": (n - 1) / (t * LOG10),
        "Grouped": grouped,
        "Assembled": (n - 1) / n * grouped,
    }


def summarise(x, b_true):
    finite = np.isfinite(x)
    xf = x[finite]
    return {
        "mean": xf.mean(),
        "sem": xf.std(ddof=1) / np.sqrt(xf.size),
        "rmse": np.sqrt(np.mean((xf - b_true) ** 2)),
        "degenerate": 1.0 - finite.mean(),
    }


def run(b_values, sizes, delta, n_rep, seed):
    rng = np.random.default_rng(seed)
    print(f"2*delta = {2 * delta}, {n_rep} catalogues per cell\n")
    for b_true in b_values:
        q = 10.0 ** (-2 * delta * b_true)
        print(f"=== b = {b_true} (q = {q:.5f}) ===")
        print(f"{'n':>6}  {'estimator':<9} {'mean':>8} {'bias %':>8} "
              f"{'+-':>7} {'RMSE':>8} {'degen':>9}")
        for n in sizes:
            k = rng.geometric(1 - q, size=(n_rep, n)) - 1
            for name, x in estimates(k, delta).items():
                s = summarise(x, b_true)
                bias_pct = 100 * (s["mean"] - b_true) / b_true
                sem_pct = 100 * s["sem"] / b_true
                print(f"{n:>6}  {name:<9} {s['mean']:>8.4f} {bias_pct:>8.2f} "
                      f"{sem_pct:>7.2f} {s['rmse']:>8.4f} "
                      f"{s['degenerate']:>9.1e}")
            print()


def run_size_distribution(zarr_path, b_true, n_events, target):
    """What each b-value estimator does to the size distribution it is plugged into.

    Reuses the simulated catalogues of the manuscript rather than fresh draws, so
    the numbers sit on the same footing as Figure 1.  The comparison is on the
    survival function, evaluated at a bin edge, where the lattice estimator is
    the continuous one exactly and no continuation is involved.
    """
    import zarr

    from bfree import bfree_survival_rounded

    store = zarr.open(zarr_path, mode="r")
    g = store[f"b{float(b_true)}_n{int(n_events)}"]
    delta, mc = g.attrs["delta"], g.attrs["mc"]
    bin_width = 2 * delta
    mmin = mc - delta

    edges = np.arange(mmin, mmin + 5 + bin_width + delta, bin_width)
    beta_true = b_true * LOG10
    sf_true = np.exp(-beta_true * (edges - mmin))
    i = int(np.argmin(np.abs(edges - target)))

    t = g["t_mag"][:].astype(np.float64)
    est = estimates_from_t(t, n_events, delta)

    def report(name, ratio, bias_pct):
        print(f"{name:<10} {bias_pct:>9} {ratio.mean():>8.3f} "
              f"{np.median(ratio):>8.3f} "
              f"{np.sqrt(np.mean((ratio - 1) ** 2)):>8.3f}")

    print(f"b = {b_true}, n = {n_events}, at m - m0 = {edges[i]:.2f}, "
          f"{len(t)} catalogues\n")
    print(f"{'estimator':<10} {'b bias %':>9} {'mean':>8} {'median':>8} "
          f"{'RMSE':>8}")
    for name, b_vals in est.items():
        good = np.isfinite(b_vals) & (b_vals > 0)
        beta = b_vals[good, None] * LOG10
        ratio = np.exp(-beta * (edges - mmin))[:, i] / sf_true[i]
        bias_pct = f"{100 * (b_vals[good].mean() - b_true) / b_true:.2f}"
        report(name, ratio, bias_pct)

    # The b-free row estimates no b-value, so it has no entry in the first column.
    w_mag = g["w_mag"][:].astype(np.int64)
    k_edge = round((edges[i] - mmin) / bin_width)
    ratio = bfree_survival_rounded(w_mag, n_events, np.array([k_edge]))[:, 0] / sf_true[i]
    report("b-free", ratio, "---")


def estimates_from_t(t, n, delta):
    """The same four estimators, from the stored sufficient statistic T."""
    kbar = (t / n - delta) / (2 * delta)
    with np.errstate(divide="ignore"):
        grouped = np.log(1 + 1 / kbar) / (LOG10 * 2 * delta)
    return {
        "Aki-Utsu": n / (t * LOG10),
        "Ogata": (n - 1) / (t * LOG10),
        "Grouped": grouped,
        "Assembled": (n - 1) / n * grouped,
    }


def run_prior_sensitivity(b_true, n_events, delta, target,
                          zarr_analysis="results/analysis_results.zarr.zip"):
    """What each prior does to the Bayesian estimator's repeated-sampling bias.

    The manuscript's informative-prior example is centred on the value the
    catalogues were drawn from, which flatters it: the bias it shows is the one
    an analyst gets only by having guessed right.  This reports the same
    quantity for priors centred elsewhere.

    Exact rather than simulated: the estimator is a function of W alone, so the
    expectation is a sum over the negative binomial law of W.  Everything is
    measured from the bottom of the catalogue, m0* - delta, so the evaluation
    point m0* + target sits at excess target + delta and the truth has to be
    taken there too -- the two coordinates differ by half a bin, which is a
    factor 10^(b delta).  The Jeffreys and truth-centred rows are checked
    against the simulated artifact for exactly that reason.
    """
    from pathlib import Path

    from scipy.stats import nbinom

    from bfree import bayesian_pdf_rounded, conjugate_prior_rounded
    from bfree.store import load as load_analysis

    two_delta = 2 * delta
    q = 10.0 ** (-two_delta * b_true)
    w = np.arange(0, int(nbinom.isf(1e-15, n_events, 1 - q)) + 1)
    pmf = nbinom.pmf(w, n_events, 1 - q)
    excess = target + delta                      # above m0* - delta, not above m0*
    u = np.array([excess / two_delta])
    truth = b_true * LOG10 * 10.0 ** (-b_true * excess)

    def mean_ratio(w0, n0):
        return float(pmf @ bayesian_pdf_rounded(w, n_events, u, delta,
                                                w0=w0, n0=n0)[:, 0]) / truth

    print(f"b_true = {b_true}, n = {n_events}, at m - m0* = {target}, "
          f"2 delta = {two_delta}\n")
    print(f"{'prior':<32}{'mean ratio':>12}{'departure':>12}")
    rows = {"jeffreys": mean_ratio(0.5, 0.0)}
    print(f"{'Jeffreys (reference)':<32}{rows['jeffreys']:>12.3f}"
          f"{abs(rows['jeffreys'] - 1):>12.3f}")
    for b_prior in (0.8, 1.0, 1.2):
        for n0_prior in (10, 100):
            w0, n0 = conjugate_prior_rounded(b_prior, n0_prior, delta)
            mr = mean_ratio(w0, n0)
            if (b_prior, n0_prior) == (b_true, 100):
                rows["conjugate"] = mr
            flag = "  <- centred on the truth" if b_prior == b_true else ""
            print(f"{f'conjugate b={b_prior}, n0={n0_prior}':<32}"
                  f"{mr:>12.3f}{abs(mr - 1):>12.3f}{flag}")

    if not Path(zarr_analysis).exists():
        raise SystemExit(f"{zarr_analysis} not found; the regression check against "
                         "the simulated result cannot be run. Produce it with "
                         "reduce_simulations.py, or pass --zarr-analysis.")
    cell = load_analysis(zarr_analysis).get((b_true, n_events))
    if cell is None:
        raise SystemExit(f"{zarr_analysis} has no cell for b={b_true}, "
                         f"n={n_events}; cannot check these values.")
    grid = cell["pdf_mrange"]
    j = int(np.argmin(np.abs(grid - target)))
    sim = {"jeffreys": float(np.asarray(cell["estimators"]["direct_mag_jeffreys"]
                                        ["ratio_mean"])[j]),
           "conjugate": float(np.asarray(cell["estimators"]["direct_mag_conjugate"]
                                         ["ratio_mean"])[j])}
    print(f"\nAgainst {cell['metadata']['n_simulations']:,} simulated catalogues:")
    worst = 0.0
    for name in ("jeffreys", "conjugate"):
        d = abs(rows[name] - sim[name])
        worst = max(worst, d)
        print(f"  {name:<10} exact {rows[name]:.4f}  simulated {sim[name]:.4f}"
              f"  difference {d:.1e}")
    # A half-bin slip in the evaluation coordinate would land at 10^(b delta).
    if worst > 1e-3:
        raise SystemExit(f"  MISMATCH: worst difference {worst:.1e} exceeds 1e-3. "
                         f"A half-bin slip would show as a factor "
                         f"{10 ** (b_true * delta):.4f}; check the evaluation "
                         "coordinate.")
    print("  agree to Monte Carlo precision")


def run_conditioning(b_true, delta, target, n_rep, seed):
    """Which random counts may be conditioned on, and at what cost.

    Both formulations condition on a count that nature does not fix: over a
    fixed observation window the catalogue size is Poisson, and the number of
    retained differences is a binomial thinning of it.  The question is not
    whether a count is random but whether its law involves b.  This reports
    three things: that the estimator survives a Poisson catalogue size, that it
    survives a random retained count too, and that the retained count -- unlike
    the catalogue size -- can estimate b on its own, which is why the
    minimum-variance claim is conditional.
    """
    from bfree import bfree_pdf_rounded

    rng = np.random.default_rng(seed)
    two_delta = 2 * delta
    q = 10.0 ** (-two_delta * b_true)
    u = np.array([target / two_delta])            # excess above the bottom
    truth = b_true * LOG10 * 10.0 ** (-b_true * target)

    def ratio(counts_and_sums):
        vals = [bfree_pdf_rounded(w, n, u, delta)[0] for n, w in counts_and_sums if n >= 2]
        return float(np.mean(vals)) / truth, len(vals)

    print(f"b = {b_true}, 2 delta = {two_delta}, evaluated {target} above the "
          f"catalogue bottom, {n_rep:,} catalogues each\n")

    print("Magnitudes, catalogue size Poisson (rate a free parameter):")
    for lam in (30, 100, 400):
        draws = []
        for _ in range(n_rep):
            n = rng.poisson(lam)
            if n < 2:
                continue
            draws.append((n, int((rng.geometric(1 - q, size=n) - 1).sum())))
        mr, used = ratio(draws)
        print(f"   Lambda = {lam:>4}   mean/truth = {mr:.4f}   ({used:,} usable)")

    print("\nNon-overlapping differences, catalogue size fixed, retained count "
          "random:")
    for h in (0, 5):
        for n_mag in (200, 800):
            draws, kept = [], []
            for _ in range(n_rep):
                j = rng.geometric(1 - q, size=n_mag) - 1
                d = j[1::2] - j[0::2]
                s = d[d >= h] - h
                kept.append(len(s))
                draws.append((len(s), int(s.sum())))
            mr, used = ratio(draws)
            print(f"   h = {h}, n_mag = {n_mag:>4}   mean/truth = {mr:.4f}   "
                  f"retention {np.mean(kept) / (n_mag // 2):.4f} vs "
                  f"q^h/(1+q) = {q**h / (1 + q):.4f}")

    print("\nIs W given the retained count still negative binomial? "
          "(the model the derivation assumes, with R in place of n)")
    from scipy.stats import nbinom
    for overlapping, counts in ((False, (10, 14, 18)), (True, (22, 26, 30))):
        got = {}
        for _ in range(600_000):
            j = rng.geometric(1 - q, size=60) - 1
            d = j[1:] - j[:-1] if overlapping else j[1::2] - j[0::2]
            s = d[d >= 1] - 1
            got.setdefault(len(s), []).append(int(s.sum()))
        label = "successive" if overlapping else "non-overlapping"
        for c in counts:
            obs = np.asarray(got.get(c, []))
            if len(obs) < 15_000:
                continue
            hi = int(nbinom.ppf(0.9995, c, 1 - q))
            ref = nbinom.pmf(np.arange(hi + 1), c, 1 - q)

            def tv(x, hi=hi, ref=ref):
                emp = np.bincount(np.clip(x, 0, hi), minlength=hi + 1)[:hi + 1]
                return 0.5 * np.abs(emp / len(x) - ref).sum()

            # Same-size draw from the exact law, to separate real departure from noise.
            floor = tv(nbinom.rvs(c, 1 - q, size=len(obs), random_state=rng))
            print(f"   {label:<16} R={c:>3}  mean W {obs.mean():7.2f} vs "
                  f"{c * q / (1 - q):7.2f}   distance {tv(obs):.4f} "
                  f"against noise floor {floor:.4f}")

    print("\nWhat each count knows about b on its own:")
    grid = np.linspace(0.30, 0.999, 40000)
    for h in (0, 5):
        est = []
        for _ in range(2000):
            j = rng.geometric(1 - q, size=4000) - 1
            p = np.mean((j[1::2] - j[0::2]) >= h)
            est.append(-np.log10(grid[np.argmin(np.abs(grid**h / (1 + grid) - p))])
                       / two_delta)
        print(f"   retained count, h = {h}: b_hat = {np.mean(est):.4f} "
              f"+- {np.std(est):.4f}   (informative)")
    for b_alt in (0.6, 1.0, 1.5):
        x = rng.exponential(1.0, size=(200_000, 2)) / (b_alt * LOG10)
        print(f"   continuous ascents, no threshold, b = {b_alt}: "
              f"P = {np.mean(x[:, 1] > x[:, 0]):.4f}   (free of b)")
    print("   catalogue size: Poisson(nu tau), law free of b by construction")


def run_pairing(b_true, delta, target, n_rep, seed):
    """What the arbitrary choice of non-overlapping pairing costs.

    Exact unbiasedness for differences needs independent pairs, which means
    pairing the catalogue up and discarding the pairs not formed.  Which pairing
    is never specified.  This reports how far two equally defensible pairings of
    the same catalogue disagree, and how much of that is recovered by averaging
    over them -- the Rao-Blackwell step the manuscript defers.  Successive
    (overlapping) differences are included because they turn out to approximate
    that average.
    """
    from bfree import bfree_pdf_rounded

    rng = np.random.default_rng(seed)
    two_delta = 2 * delta
    q = 10.0 ** (-two_delta * b_true)
    u = np.array([target / two_delta])
    truth = b_true * LOG10 * 10.0 ** (-b_true * target)
    h = 1                                   # the threshold used throughout

    def evaluate(excess):
        return (bfree_pdf_rounded(int(excess.sum()), len(excess), u, delta)[0]
                if len(excess) >= 2 else np.nan)

    def pairing(j, offset):
        lo, hi = j[offset::2], j[offset + 1::2]
        k = min(len(lo), len(hi))
        d = hi[:k] - lo[:k]
        return evaluate(d[d >= h] - h)

    def successive(j):
        d = j[1:] - j[:-1]
        return evaluate(d[d >= h] - h)

    print(f"b = {b_true}, 2 delta = {two_delta}, threshold {h} bin, evaluated "
          f"{target} above the catalogue bottom, {n_rep:,} catalogues\n")
    for n in (50, 200, 1000):
        a, b_, o = [], [], []
        for _ in range(n_rep):
            j = rng.geometric(1 - q, size=n) - 1
            a.append(pairing(j, 0))
            b_.append(pairing(j, 1))
            o.append(successive(j))
        a, b_, o = (np.asarray(x) / truth for x in (a, b_, o))
        ok = ~(np.isnan(a) | np.isnan(b_) | np.isnan(o))
        a, b_, o = a[ok], b_[ok], o[ok]
        avg = 0.5 * (a + b_)

        def summary(x):
            return x.mean(), float(np.sqrt(((x - 1) ** 2).mean()))

        print(f"--- n = {n}  ({ok.sum():,} catalogues with all three defined) ---")
        print(f"{'':<26}{'mean':>8}{'RMSE':>9}")
        for name, x in (("one pairing", a), ("average of two pairings", avg),
                        ("successive (overlapping)", o)):
            mu, rmse = summary(x)
            print(f"  {name:<24}{mu:>8.4f}{rmse:>9.3f}")
        print(f"  correlation between the two pairings: "
              f"{np.corrcoef(a, b_)[0, 1]:+.3f}")
        print(f"  RMSE recovered by averaging two pairings: "
              f"{100 * (1 - summary(avg)[1] / summary(a)[1]):.0f}%\n")


def run_overlap_bias(delta, n_rep, seed,
                     zarr_analysis="results/analysis_results.zarr.zip"):
    """A formula for the bias of the successive-difference estimator.

    Successive differences telescope, so the retained excesses carry one
    aggregate constraint the independent case does not.  The estimator behaves as
    though it held slightly more than the R observations it was handed, and that
    single constant deficit accounts for the whole bias: it grows linearly in the
    evaluation index and decays as 1/n.  Fits the constant against the stored
    simulation, then checks that one global value corrects the estimator.
    """
    from pathlib import Path

    from bfree import bfree_survival_rounded
    from bfree.store import load as load_analysis

    two_delta = 2 * delta
    print("Model:  relative bias  ~  D * (1 - q^2)/q * k/n,   k in bins\n"
          "equivalently D * (1 - q) * k / R: the estimator acts as if it held\n"
          "R + D observations rather than R.\n")

    if not Path(zarr_analysis).exists():
        raise SystemExit(f"{zarr_analysis} not found; needed to fit the constant.")
    analysis = load_analysis(zarr_analysis)

    fitted = []
    print(f"{'b':>5}{'q':>9}{'D':>9}{'intercept':>11}")
    for b_true in (0.8, 1.0, 1.2):
        q = 10.0 ** (-two_delta * b_true)
        ks, ys = [], []
        for n in (500, 1000):          # the 1/n regime; smaller n carries more
            cell = analysis.get((b_true, n))
            if cell is None:
                continue
            grid = np.asarray(cell["pdf_mrange"])
            for m in (1, 2, 3, 4, 5):
                j = int(np.argmin(np.abs(grid - m)))
                bias = float(np.asarray(cell["estimators"]["direct_diff_ol"]
                                        ["ratio_mean"])[j]) - 1.0
                ks.append(m / two_delta)
                ys.append(bias * n)
        slope, cept = np.polyfit(np.asarray(ks), np.asarray(ys), 1)
        d = slope * q / (1 - q ** 2)
        fitted.append(d)
        print(f"{b_true:>5}{q:>9.4f}{d:>9.4f}{cept:>11.3f}")
    d_hat = float(np.mean(fitted))
    print(f"\nD = {d_hat:.4f} +- {np.std(fitted):.4f} "
          f"({100 * np.std(fitted) / d_hat:.1f}% across b)\n")

    print(f"Correcting with that one value, {n_rep:,} catalogues:")
    print(f"{'b':>5}{'n':>6}{'k':>5}{'uncorrected':>13}{'corrected':>11}")
    rng = np.random.default_rng(seed)
    for b_true in (0.8, 1.0, 1.2):
        q = 10.0 ** (-two_delta * b_true)
        for n in (100, 400):
            w, counts = [], []
            for _ in range(n_rep):
                j = rng.geometric(1 - q, size=n) - 1
                d = j[1:] - j[:-1]
                s = d[d >= 1] - 1
                if len(s) >= 1:
                    w.append(int(s.sum()))
                    counts.append(len(s))
            w, counts = np.asarray(w), np.asarray(counts)
            for k in (20.0, 50.0):
                truth = q ** k
                raw = bfree_survival_rounded(w, counts,
                                             np.array([k]))[:, 0].mean() / truth - 1
                fix = bfree_survival_rounded(w, counts + d_hat,
                                             np.array([k]))[:, 0].mean() / truth - 1
                print(f"{b_true:>5}{n:>6}{k:>5.0f}{raw:>13.4f}{fix:>11.4f}")


def run_figure(figure_dir, b_true, sizes, delta, n_rep, seed,
               zarr_path, sd_n, sd_target, formats):
    """Draw the two-by-two: which bias each correction removes, and what it costs.

    Panels (a) and (b) carry what the supplement's estimator table reported,
    bias and RMSE against catalog size, so the figure replaces it rather than
    sitting beside it. Panel (c) is the reversal. Colour encodes the likelihood,
    line style the finite-size factor, so each estimator reads as two choices.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt

    from bfree.plotting import (
        COLORS,
        apply_sans_fonts,
        panel_label,
        save_figure,
        size,
    )

    style = {"Aki-Utsu": (COLORS["aki"], "solid"),
             "Ogata": (COLORS["ogata"], "dashed"),
             "Grouped": (COLORS["grouped"], "solid"),
             "Assembled": (COLORS["assembled"], "dashed")}

    rng = np.random.default_rng(seed)
    q = 10.0 ** (-2 * delta * b_true)
    bias = {name: [] for name in style}
    rmse = {name: [] for name in style}
    for n in sizes:
        k = rng.geometric(1 - q, size=(n_rep, n)) - 1
        for name, x in estimates(k, delta).items():
            s = summarise(x, b_true)
            bias[name].append(100 * (s["mean"] - b_true) / b_true)
            rmse[name].append(s["rmse"])

    sf_err = _size_distribution_errors(zarr_path, b_true, sd_n, sd_target)

    apply_sans_fonts()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=size(0.34), layout="constrained")

    for name, (colour, ls) in style.items():
        ax1.plot(sizes, bias[name], color=colour, linestyle=ls, marker="o",
                 markersize=3.0, label=name)
    ax1.axhline(0.0, color="0.4", linewidth=0.8, zorder=0)
    ax1.set_xscale("log")
    ax1.set_xlabel("catalog size $n$")
    # text.usetex is on, so a bare % would comment out the rest of the label.
    ax1.set_ylabel(r"bias in $b$ (\% of $b$)")
    ax1.legend(frameon=False)
    panel_label(ax1, "a")

    # Absolute RMSE is dominated by the 1/sqrt(n) trend, which hides the
    # differences between estimators; the ratio to Aki-Utsu shows them.
    for name, (colour, ls) in style.items():
        ratio = np.asarray(rmse[name]) / np.asarray(rmse["Aki-Utsu"])
        ax2.plot(sizes, ratio, color=colour, linestyle=ls, marker="o",
                 markersize=3.0)
    ax2.axhline(1.0, color="0.4", linewidth=0.8, zorder=0)
    ax2.set_xscale("log")
    ax2.set_xlabel("catalog size $n$")
    ax2.set_ylabel("RMSE of $b$, relative to Aki-Utsu")
    panel_label(ax2, "b")

    for name, (colour, _ls) in style.items():
        b_bias, ratio = sf_err[name]
        ax3.plot([b_bias], [ratio], color=colour, marker="o", markersize=5,
                 linestyle="none")
        right = b_bias > 0.5 * (max(v[0] for v in sf_err.values())
                                + min(v[0] for v in sf_err.values()))
        ax3.annotate(name, (b_bias, ratio), textcoords="offset points",
                     xytext=(-5 if right else 5, -2), color=colour,
                     ha="right" if right else "left")
    ax3.axhline(1.0, color="0.4", linewidth=0.8, zorder=0)
    ax3.axvline(0.0, color="0.4", linewidth=0.8, zorder=0)
    ax3.margins(x=0.22)
    ax3.set_xlabel(r"bias in $b$ (\% of $b$)")
    ax3.set_ylabel("mean estimated / true survival")
    panel_label(ax3, "c")


    out = Path(figure_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in save_figure(fig, "plugin_corrections", out, formats,
                            inputs=(zarr_path,)):
        print(f"saved {path}")
    plt.close(fig)


def _size_distribution_errors(zarr_path, b_true, n_events, target):
    """Signed b-value bias and survival-ratio mean for each estimator, paired."""
    import zarr

    store = zarr.open(zarr_path, mode="r")
    g = store[f"b{float(b_true)}_n{int(n_events)}"]
    delta, mc = g.attrs["delta"], g.attrs["mc"]
    mmin = mc - delta
    edges = np.arange(mmin, mmin + 5 + 2 * delta + delta, 2 * delta)
    i = int(np.argmin(np.abs(edges - target)))
    sf_true = np.exp(-b_true * LOG10 * (edges[i] - mmin))

    t = g["t_mag"][:].astype(np.float64)
    out = {}
    for name, b_vals in estimates_from_t(t, n_events, delta).items():
        good = np.isfinite(b_vals) & (b_vals > 0)
        ratio = np.exp(-b_vals[good] * LOG10 * (edges[i] - mmin)) / sf_true
        out[name] = (100 * (b_vals[good].mean() - b_true) / b_true, ratio.mean())
    return out


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Bias and RMSE of the combined plug-in estimator against the "
                    "published estimators it is built from.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--b-values", type=float, nargs="+",
                        default=[0.8, 1.0, 1.2])
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[20, 50, 100, 200, 500, 1000])
    parser.add_argument("--delta", type=float, default=0.05,
                        help="Half the bin width; catalogues rounded to 2*delta.")
    parser.add_argument("--n-rep", type=int, default=400_000,
                        help="Catalogues simulated per (b, n) cell.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--size-distribution", action="store_true",
                        help="Instead of the b-value comparison, report what "
                             "each estimator does to the size distribution it "
                             "is plugged into.")
    parser.add_argument("--zarr", default="results/simulation_results.zarr",
                        help="Simulated catalogues, for --size-distribution.")
    parser.add_argument("--sd-b", type=float, default=1.0)
    parser.add_argument("--sd-n", type=int, default=100)
    parser.add_argument("--sd-target", type=float, default=4.0,
                        help="Magnitude above m_0 at which the ratio is taken.")
    parser.add_argument("--prior-sensitivity", action="store_true",
                        help="Instead of either, report the Bayesian estimator's "
                             "departure under priors centred away from the truth.")
    parser.add_argument("--conditioning", action="store_true",
                        help="Instead of any of the above, check which random "
                             "counts may be conditioned on without loss.")
    parser.add_argument("--pairing", action="store_true",
                        help="Instead of any of the above, report what the "
                             "arbitrary non-overlapping pairing costs.")
    parser.add_argument("--overlap-bias", action="store_true",
                        help="Instead of any of the above, fit and check a "
                             "formula for the successive-difference bias.")
    parser.add_argument("--figure-dir", default=None, metavar="DIR",
                        help="Instead of printing, draw the two-by-two figure "
                             "into DIR.")
    parser.add_argument("--format", nargs="+", default=["png", "pdf"],
                        help="Figure formats for --figure-dir.")
    parser.add_argument("--style", choices=("original", "screen", "bssa"),
                        default="screen",
                        help="Figure style for --figure-dir.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.figure_dir:
        from bfree import bssa, plotting
        if args.style == "bssa":
            bssa.apply()
            plotting.TITLES_IN_FIGURES = False
        elif args.style == "original":
            plotting.apply_sans_fonts()
            plotting.ORIGINAL_SIZES = True
        run_figure(args.figure_dir, args.sd_b, args.sizes, args.delta,
                   args.n_rep, args.seed, args.zarr, args.sd_n, args.sd_target,
                   args.format)
    elif args.size_distribution:
        run_size_distribution(args.zarr, args.sd_b, args.sd_n, args.sd_target)
    elif args.prior_sensitivity:
        run_prior_sensitivity(args.sd_b, args.sd_n, args.delta, args.sd_target)
    elif args.conditioning:
        run_conditioning(args.sd_b, args.delta, 2.0, 150_000, args.seed)
    elif args.pairing:
        run_pairing(args.sd_b, args.delta, 4.0, 30_000, args.seed)
    elif args.overlap_bias:
        run_overlap_bias(args.delta, 200_000, args.seed)
    else:
        run(args.b_values, args.sizes, args.delta, args.n_rep, args.seed)
