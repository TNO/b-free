"""
Produce the manuscript figures.

Each figure is a function registered with the data files it reads, so the
manifest records exactly what each one used.

    python scripts/make_figures.py --figure-dir DIR --results-dir DIR --format png pdf
    python scripts/make_figures.py ... --only eshm20_ratio_by_catalog_size eshm20_example_zones
    python scripts/make_figures.py ... --only frequentist_by_formulation bayesian_by_formulation
    python scripts/make_figures.py --list

No directory or format has a default: the figures belong to another repository,
and a wrong path should be an error rather than a stray tree or a stale read.
The default paper set excludes the cross-formulation comparisons and extended
catalog-size sweep; request them explicitly with `--only` for research use.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

# Selected before any figure is made, so the script does not depend on a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from bfree import bssa, plotting
from bfree.bssa import LINE_BASE, LINE_THICK, LINE_THIN
from bfree.plotting import (
    FORMATS,
    FRAMEWORKS,
    KEYS,
    RATIO_LABEL_PREDICTED,
    XLABEL_ROUNDED,
    estimator_colour,
    estimator_matrix_pair,
    estimator_panels,
    legend_below_axes,
    legend_below_groups,
    legend_handle,
    line_kwargs,
    panel_title,
    save_figure,
    size,
)
from bfree.store import load as load_results

ANALYSIS = "analysis_results.zarr.zip"
CONTINUOUS = "analysis_results_continuous.zarr.zip"
ESHM20 = "eshm20_results.zarr.zip"
HAZARD = "hazard_results.zarr.zip"
HAZARD_FIXED_RP = "hazard_fixed_rp_results.zarr.zip"
RESEARCH_ONLY = frozenset((
    "frequentist_by_formulation", "bayesian_by_formulation", "error_by_b_and_catalog_size", "hazard_fixed_rp_amplitude"
))

# The case shown in the main text.
B_MAIN, N_MAIN = 1.0, 100

# The curves are survival functions, not b-values, so the legend names the
# method that produced each. A b symbol would say the panel shows an estimate
# of b, and for b-free there is no b to show.
LABEL_AKI = "Aki-Utsu plug-in"
LABEL_BFREE = "Frequentist b-free"
LABEL_BAYES = "Bayesian b-free (Jeffreys)"
# The 250-year return period is this study's reading level, not an ESHM20 one;
# only the a-value and b-value behind the magnitude come from ESHM20.
LABEL_M250 = r"250-year magnitude ($m_{250}$)"
ESTIMATOR_LABELS = {
    "aki": LABEL_AKI,
    "ogata": "Ogata plug-in",
    "grouped": "Grouped plug-in",  # Bender (1983) calls the rounded-magnitude likelihood "grouped"
    "bfree": LABEL_BFREE,
}
M_BFREE, M_BAYES = "o", "^"
IMT_ORDER = ["PGA", "SA(0.2)", "SA(1.0)"]
IMT_LABELS = {"PGA": "PGA", "SA(0.2)": "$S_a$(0.2 s)", "SA(1.0)": "$S_a$(1.0 s)"}
IMT_LINE_STYLES = {"PGA": "-", "SA(0.2)": "--", "SA(1.0)": ":"}
# BSSA asks for 10 to 12 point text within figures, so annotations sit at the floor.
ANNOTATION_PT = 10.0
HAZ_RP = 2475


class Results:
    """The result stores in the results directory, read once each, on demand."""

    def __init__(self, results_dir):
        self.dir = Path(results_dir)
        self._cache = {}

    def __getitem__(self, name):
        if name not in self._cache:
            path = self.dir / name
            if not path.exists():
                raise SystemExit(f"required input is missing: {path}")
            self._cache[name] = load_results(path)
        return self._cache[name]

    def bayes_labels(self):
        """Legend labels for the Bayesian estimators, naming the conjugate prior.

        The prior's strength is read from the stored metadata rather than
        repeated here, so the label cannot disagree with what was computed.
        """
        meta = self[ANALYSIS][(B_MAIN, N_MAIN)]["metadata"]
        conjugate = (
            "Bayesian b-free (conjugate, "
            f"$b_0={meta['b_prior']:g}$, $n_0={meta['n0_prior']:.0f}$)"
        )
        return {"jeffreys": LABEL_BAYES, "conjugate": conjugate}


REGISTRY = {}


def figure(name, inputs):
    """Register a figure builder under *name*, declaring the data it reads.

    *inputs* are file names in the results directory. They are hashed into the
    manifest, so a figure must declare what it actually opens and nothing more:
    an input listed here but never read would put a false entry in the record.
    """
    def register(func):
        REGISTRY[name] = (func, inputs)
        return func
    return register


# --------------------------------------------------------------------------
# Synthetic experiments
# --------------------------------------------------------------------------

def _freq_matrix(r, quantity):
    """The frequentist estimators, continuous above and rounded below."""
    return estimator_matrix_pair(
        r[CONTINUOUS][(B_MAIN, N_MAIN)],
        r[ANALYSIS][(B_MAIN, N_MAIN)],
        ["aki", "ogata", "bfree"],
        ["grouped", "aki", "ogata", "bfree"],
        ESTIMATOR_LABELS,
        quantity=quantity,
    )


@figure("frequentist_survival", (ANALYSIS, CONTINUOUS))
def frequentist_survival(r):
    """Frequentist estimators against the truth, as survival functions.

    The plug-in bias appears in the continuous row too, so it is not a rounding
    artefact."""
    return _freq_matrix(r, "sf")


def _bayes_matrix(r, quantity):
    """The Bayesian estimators, continuous above and rounded below."""
    labels = r.bayes_labels()
    return estimator_matrix_pair(
        r[CONTINUOUS][(B_MAIN, N_MAIN)],
        r[ANALYSIS][(B_MAIN, N_MAIN)],
        ["jeffreys", "conjugate"],
        ["jeffreys", "conjugate"],
        labels,
        quantity=quantity,
        key="Posterior predictive",
        ratio_label=RATIO_LABEL_PREDICTED,
        predictive=True,
    )


@figure("bayesian_survival", (ANALYSIS, CONTINUOUS))
def bayesian_survival(r):
    """Bayesian posterior-predictive estimators under both priors, as survival
    functions."""
    return _bayes_matrix(r, "sf")


@figure("frequentist_by_formulation", (ANALYSIS,))
def frequentist_by_formulation(r):
    """All three formulations for the frequentist estimators, with spread.

    Magnitudes, non-overlapping differences and overlapping differences, with
    mean, median and RMSE."""
    return estimator_panels(
        r[ANALYSIS][(B_MAIN, N_MAIN)],
        ["grouped", "aki", "ogata", "bfree"],
        ESTIMATOR_LABELS,
        rounded=True,
        frameworks=("mag", "diff_nol", "diff_ol"),
        quantity="sf",
        show_spread=True,
    )


@figure("bayesian_by_formulation", (ANALYSIS,))
def bayesian_by_formulation(r):
    """All three formulations for the Bayesian estimators, with spread."""
    return estimator_panels(
        r[ANALYSIS][(B_MAIN, N_MAIN)],
        ["jeffreys", "conjugate"],
        r.bayes_labels(),
        rounded=True,
        frameworks=("mag", "diff_nol", "diff_ol"),
        quantity="sf",
        show_spread=True,
        key="Posterior predictive",
        ratio_label=RATIO_LABEL_PREDICTED,
        predictive=True,
    )


@figure("bayesian_priors", ())
def bayesian_priors(_r):
    """What the b-free priors assume, continuous first to match the exposition.

    Takes no input: the priors are analytic, so nothing is read from results.
    """
    from scipy.special import betaln
    from scipy.stats import beta as beta_dist
    from scipy.stats import gamma as gamma_dist

    from bfree import conjugate_prior_rounded

    LN10, LOG10E = np.log(10.0), np.log10(np.e)
    STEP = 0.1  # 2*delta for the rounded panels
    B_0 = 1.0
    CONJ = [(10, "#E69F00"), (100, "#009E73")]
    C_JEFF = estimator_colour("jeffreys")
    CONT, ROUNDED = 0, 1  # continuous on top, matching the exposition
    JLAB = "Jeffreys (improper)"

    fig, axs = plt.subplots(2, 3, figsize=size(0.62, original=(15, 8.5)))
    q = np.linspace(1e-4, 1 - 1e-4, 4000)
    b = np.linspace(1e-3, 6.0, 4000)
    bet = np.linspace(1e-3, 6.0 * LN10, 4000)
    x = np.linspace(0, 4.0, 400)
    qb = 10.0 ** (-STEP * b)

    # Reference priors are improper, so they are drawn unnormalised, each scaled to
    # 0.5 at b = 1 so that the two formulations can be compared directly.
    kern_q = q**-0.5 / (1 - q)  # Beta(1/2, 0), geometric likelihood
    kern_b = qb**0.5 / (1 - qb)  # the same prior, in b
    kern_b1 = (10.0**-0.5) / (1 - 10.0**-STEP)
    axs[ROUNDED, 0].plot(
        q, kern_q / kern_q[len(q) // 2] * 2, color=C_JEFF, lw=LINE_BASE, label=JLAB
    )
    axs[ROUNDED, 1].plot(b, kern_b / kern_b1 * 0.5, color=C_JEFF, lw=LINE_BASE, label=JLAB)
    axs[ROUNDED, 2].plot([], [], color=C_JEFF, lw=LINE_BASE, label=JLAB)
    axs[CONT, 0].plot(
        bet, (1.0 / bet) / (1.0 / LN10) * 0.5, color=C_JEFF, lw=LINE_BASE, label=JLAB
    )
    axs[CONT, 1].plot(b, 0.5 / b, color=C_JEFF, lw=LINE_BASE, label=JLAB)
    axs[CONT, 2].plot([], [], color=C_JEFF, lw=LINE_BASE, label=JLAB)

    u = np.linspace(0, 4.0 / STEP, 400)
    for n0, colour in CONJ:
        lab = f"conjugate, $b_0={B_0:.0f}$, $n_0={n0}$"
        W0, N0 = conjugate_prior_rounded(B_0, n0, STEP / 2)
        axs[ROUNDED, 0].plot(
            q, beta_dist.pdf(q, W0, N0), color=colour, lw=LINE_BASE, label=lab
        )
        axs[ROUNDED, 1].plot(
            b,
            beta_dist.pdf(qb, W0, N0) * STEP * LN10 * qb,
            color=colour,
            lw=LINE_BASE,
            label=lab,
        )
        axs[ROUNDED, 2].plot(
            STEP * u - STEP / 2,
            np.exp(betaln(W0 + u, N0) - betaln(W0, N0)),
            color=colour,
            lw=LINE_BASE,
            label=lab,
        )
        T0 = n0 * LOG10E / B_0
        axs[CONT, 0].plot(
            bet,
            gamma_dist.pdf(bet, n0, scale=1.0 / T0),
            color=colour,
            lw=LINE_BASE,
            label=lab,
        )
        axs[CONT, 1].plot(
            b,
            gamma_dist.pdf(b, n0, scale=1.0 / (T0 * LN10)),
            color=colour,
            lw=LINE_BASE,
            label=lab,
        )
        axs[CONT, 2].plot(x, (1 + x / T0) ** (-n0), color=colour, lw=LINE_BASE, label=lab)

    axs[CONT, 0].set(
        xlabel="Exponential rate $\\beta$",
        ylabel="Prior density",
        ylim=(0, 2.5 / LN10),
    )
    axs[CONT, 1].set(
        xlabel=r"$\mathrm{b}$-value",
        ylabel="Prior density",
        ylim=(0, 2.5),
    )
    axs[CONT, 2].set(
        xlabel="$m - m_0$",
        ylabel="$P(M \\geq m)$",
        yscale="log",
        ylim=(1e-5, 1.5),
    )
    axs[ROUNDED, 0].set(
        xlabel="Bin-survival probability $q$",
        ylabel="Prior density",
        ylim=(0, 12),
    )
    axs[ROUNDED, 1].set(
        xlabel=r"$\mathrm{b}$-value",
        ylabel="Prior density",
        ylim=(0, 2.5),
    )
    axs[ROUNDED, 2].set(
        xlabel=r"$m-m_0^{*}$",
        ylabel="$P(M \\geq m)$",
        yscale="log",
        ylim=(1e-5, 1.5),
    )
    titles = [
        "Continuous: prior on $\\beta$",
        "Continuous: the same priors in $b$",
        "Continuous: prior predictive",
        "Rounded: prior on $q$",
        "Rounded: the same priors in $b$",
        "Rounded: prior predictive",
    ]
    for ax, letter, text in zip(axs.ravel(), "abcdef", titles, strict=True):
        panel_title(ax, letter, text)
    for ax in axs.ravel():
        ax.grid(alpha=0.3)
    for ax in axs[:, 1]:
        ax.axvline(1.0, color="0.5", ls=":", lw=0.5, zorder=0)

    legend_below_axes(fig, axs)
    return fig


@figure("error_by_b_and_catalog_size", (ANALYSIS,))
def error_by_b_and_catalog_size(r):
    """Ratio of estimated to true survival function over every simulated
    catalogue size and true b-value."""
    analysis = r[ANALYSIS]
    b_true_range = sorted({k[0] for k in analysis})
    cat_sizes = sorted({k[1] for k in analysis})
    estimators = ["grouped", "aki", "ogata", "bfree"]

    fig, axs = plt.subplots(
        nrows=len(b_true_range),
        ncols=len(cat_sizes),
        figsize=size(
            0.86 * len(b_true_range) / len(cat_sizes),
            original=(3.5 * len(cat_sizes), 3.0 * len(b_true_range)),
        ),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, b_true in enumerate(b_true_range):
        for j, n in enumerate(cat_sizes):
            ax = axs[i, j]
            entry = analysis.get((b_true, n))
            if entry is None:
                ax.set_visible(False)
                continue
            mrange = np.asarray(entry["sf_mrange"])
            ax.axhline(1, color="k", lw=0.5, alpha=0.4)
            for estimator in estimators:
                for framework, _ls, _lbl in FRAMEWORKS:
                    e = entry["estimators"].get(KEYS[(estimator, framework)])
                    if e is None:
                        continue
                    ax.plot(
                        mrange,
                        e["sf_ratio_mean"],
                        **line_kwargs(estimator, framework, lw=0.6, alpha=0.75),
                    )

            ax.set(yscale="log", xlim=[0, 5])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.grid(True, alpha=0.25, which="both")
            if i == 0:
                ax.set_title(f"$n={n}$")
            if j == 0:
                ax.set_ylabel(f"$b={b_true}$\nRatio $\\hat{{S}}/S^{{\\rm true}}$")
            if i == len(b_true_range) - 1:
                ax.set_xlabel(XLABEL_ROUNDED)
            plotting.panel_label(ax, chr(ord("a") + i * len(cat_sizes) + j))

    handles = [legend_handle(e, ESTIMATOR_LABELS[e]) for e in estimators]
    framework_handles = [
        plt.Line2D(
            [0], [0], color=plotting.NEUTRAL, lw=0.6,
            ls=ls, label=label,
        )
        for name, ls, label in FRAMEWORKS
    ]
    plotting.figure_title(
        fig,
        "Rounded catalogues: mean estimated survival function / true survival function",
    )
    legend_below_groups(fig, [
        ("Estimator, by colour", handles, 1),
        ("Input data, by line style", framework_handles, 1),
    ])
    return fig


# --------------------------------------------------------------------------
# ESHM20 application
# --------------------------------------------------------------------------

@figure("eshm20_ratio_by_catalog_size", (ESHM20,))
def eshm20_ratio_by_catalog_size(r):
    """Every qualifying ESHM20 zone: ratio to the Aki-Utsu plug-in against
    catalogue size."""
    zones = r[ESHM20]["zones"]
    highlights = r[ESHM20]["highlights"]

    fig, ax = plt.subplots(figsize=size(0.45, original=(10, 5)))

    ax.scatter(
        zones["n"],
        zones["ratio_bfree"],
        s=30,
        alpha=0.7,
        color=estimator_colour("bfree"),
        label=LABEL_BFREE,
        zorder=3,
    )
    ax.scatter(
        zones["n"],
        zones["ratio_bayes"],
        s=30,
        alpha=0.7,
        color=estimator_colour("jeffreys"),
        marker="^",
        label=LABEL_BAYES,
        zorder=3,
    )
    ax.axhline(1, color="0.4", lw=1, ls="--", zorder=1)

    for label, zid in highlights.items():
        row = zones[zones["zone_id"] == zid].iloc[0]
        # The Bayesian ratio exceeds the frequentist one in every zone, so
        # labelling one above its marker and the other below keeps a zone's two
        # labels apart even where the markers themselves nearly coincide.
        for column, marker, color, dy in [
            ("ratio_bfree", M_BFREE, estimator_colour("bfree"), -16),
            ("ratio_bayes", M_BAYES, estimator_colour("jeffreys"), 8),
        ]:
            ax.scatter(
                row["n"],
                row[column],
                s=120,
                color=color,
                edgecolors="k",
                lw=0.8,
                zorder=5,
                marker=marker,
            )
            ax.annotate(
                label,
                xy=(row["n"], row[column]),
                xytext=(5, dy),
                textcoords="offset points",
                fontweight="bold",
            )

    # The lower label hangs a fixed number of points below the lowest marker, so
    # the room for it has to be made after the limits are known.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.11 * (hi - lo), hi)

    ax.set_xlabel("Number of events in last completeness period ($n$)")
    ax.set_ylabel(f"Ratio to {LABEL_AKI}")
    plotting.figure_title(
        fig,
        r"Earthquake size distribution at $m_{250}$: all qualifying ESHM20 zones",
    )
    ax.grid(True, alpha=0.3)

    legend_below_axes(fig, [ax])
    return fig


@figure("eshm20_example_zones", (ESHM20,))
def eshm20_example_zones(r):
    """Three ESHM20 zones in detail: a small catalogue, a large one, and the
    largest difference.

    Shows the observed exceedance levels against the three fitted survival
    functions, and their ratios below."""
    highlights = r[ESHM20]["highlights"]
    curves = r[ESHM20]["highlight_curves"]
    XLIM = (3.3, 7)

    # The survival row carries the annotation boxes as well as the curves, so it
    # is given the taller share of the figure to keep the two apart.
    fig, axs = plt.subplots(
        2, 3, figsize=size(0.80, original=(15, 10)), squeeze=False,
        sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.4, 1]},
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    ratio_values = []

    for col, (label, zid) in enumerate(highlights.items()):
        c = curves[label]
        # zone_curves() returns the survival functions against magnitude threshold,
        # which is the same axis the observed exceedance levels live on.
        m_thresh, n_obs, d = c["m_thresh"], c["n"], c["delta"]

        ax_sf = axs[0, col]
        # Defined on every bin from m0 up, not only the occupied ones: an empty bin
        # repeats the previous level, so the steps tile the axis without gaps.
        k_obs = np.round((np.asarray(c["magnitudes"]) - c["mc"]) / (2 * d)).astype(int)
        k_all = np.arange(k_obs.max() + 1)
        surv_obs = np.array([(k_obs >= k).sum() / n_obs for k in k_all])
        obs_edges = c["mc"] + 2 * d * k_all - d
        ax_sf.plot(obs_edges, surv_obs, "o", color="k", ms=2.5, label="Observed", zorder=5)
        ax_sf.plot(
            m_thresh,
            c["surv_aki"],
            label=LABEL_AKI,
            **line_kwargs("aki", lw=LINE_BASE),
        )
        ax_sf.plot(m_thresh, c["surv_bfree"], label=LABEL_BFREE,
                   **line_kwargs("bfree", lw=LINE_BASE))
        ax_sf.plot(m_thresh, c["surv_bayes"], label=LABEL_BAYES,
                   **line_kwargs("jeffreys", lw=LINE_BASE))
        ax_sf.axvline(
            c["m_250"], color="0.4", ls=":", lw=LINE_THIN,
            label=LABEL_M250,
        )
        ax_sf.set(yscale="log", xlim=XLIM, xlabel="Magnitude $m$")
        # The rows are far enough apart to read separately, so the shared axis
        # keeps its tick labels here rather than only under the bottom row.
        ax_sf.tick_params(labelbottom=True)
        ax_sf.text(
            0.05, 0.05,
            f"$n={n_obs}$\n$b_{{\\mathrm{{Aki-Utsu}}}}={c['b_aki']:.2f}$"
            f"\n$m_{{250}}={c['m_250']:.1f}$",
            transform=ax_sf.transAxes, ha="left", va="bottom",
            fontsize=ANNOTATION_PT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            zorder=6,
        )
        if col == 0:
            ax_sf.set_ylabel("$P(M \\geq m)$")
        # Inside the frame rather than above it, which frees the title row.
        # Clear of the top spine, so the patch behind it leaves no gap in the frame.
        ax_sf.text(
            0.97, 0.92, zid, transform=ax_sf.transAxes, ha="right", va="top",
            fontsize=ANNOTATION_PT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                  "pad": 1.0},
            zorder=6,
        )
        if not plotting.TITLES_IN_FIGURES:
            bssa.panel_label(ax_sf, "abc"[col])
        ax_sf.grid(True, which="both", alpha=0.3)

        ax_r = axs[1, col]
        r_bfree = c["surv_bfree"] / c["surv_aki"]
        r_bayes = c["surv_bayes"] / c["surv_aki"]
        ax_r.plot(m_thresh, r_bfree, label=LABEL_BFREE, **line_kwargs("bfree", lw=LINE_BASE))
        ax_r.plot(m_thresh, r_bayes, label=LABEL_BAYES, **line_kwargs("jeffreys", lw=LINE_BASE))
        ax_r.axhline(1, color="0.4", ls="--", lw=1)
        ax_r.axvline(c["m_250"], color="0.4", ls=":", lw=LINE_THIN)
        ax_r.set(xlim=XLIM, xlabel="Magnitude $m$", ylabel=f"Ratio to {LABEL_AKI}")
        if col != 0:
            ax_r.set_ylabel("")
        if not plotting.TITLES_IN_FIGURES:
            bssa.panel_label(ax_r, "def"[col])
        ax_r.grid(True, alpha=0.3)
        for arr in (r_bfree, r_bayes):
            ratio_values.extend(arr[np.isfinite(arr) & (arr > 0)].tolist())

    # Shared log-scale limits across the ratio panels, with 15% padding in log space.
    lo, hi = max(min(ratio_values), 1e-6), max(ratio_values)
    pad = (np.log10(hi) - np.log10(lo)) * 0.15
    ylo, yhi = 10 ** (np.log10(lo) - pad), 10 ** (np.log10(hi) + pad)
    for ax_r in axs[1]:
        ax_r.set(yscale="log", ylim=(ylo, yhi))
        ax_r.set_yticks([t for t in (0.2, 0.5, 1, 2, 3) if ylo <= t <= yhi])
        ax_r.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_r.yaxis.set_minor_locator(mticker.NullLocator())

    data_handles = [
        plt.Line2D([], [], color="k", marker="o", ls="none", markersize=3,
                   label="Observed"),
    ]
    estimator_handles = [
        legend_handle("aki", LABEL_AKI),
        legend_handle("bfree", LABEL_BFREE),
        legend_handle("jeffreys", LABEL_BAYES),
    ]
    reference_handles = [
        plt.Line2D([], [], color="0.4", ls=":", lw=LINE_THIN, label=LABEL_M250),
    ]
    plt.Line2D([], [], color="0.4", ls="--", lw=1, ),
    
    legend_below_groups(fig, [
        ("Data", data_handles, 1),
        ("Estimator", estimator_handles, 1),
        ("Reference", reference_handles, 1),
    ])
    return fig


@figure("hazard_propagation", (HAZARD,))
def hazard_propagation(r):
    """Hazard-domain ratio against catalogue size, one panel per intensity
    measure, on shared axes so the three can be read against each other."""
    records = r[HAZARD]["records"]

    fig, axs = plt.subplots(
        3, 1, figsize=size(0.95, original=(8, 11)), sharex=True, sharey=True
    )

    for row, (ax, imt) in enumerate(zip(axs, IMT_ORDER, strict=True)):
        sel = [x for x in records if x["imt"] == imt and x[HAZ_RP] is not None]
        n = [x["n"] for x in sel]
        for key, colour, marker, label in (
            ("bfree", estimator_colour("bfree"), M_BFREE, LABEL_BFREE),
            ("bayes", estimator_colour("jeffreys"), M_BAYES, LABEL_BAYES),
        ):
            ax.scatter(n, [x[HAZ_RP][key] for x in sel], s=30, alpha=0.7,
                       color=colour, marker=marker, label=label, zorder=3)
        ax.axhline(1, color="0.4", lw=1, ls="--", zorder=1)
        # Naming the measure inside the frame keeps each panel readable on its
        # own, which a bare panel letter on a shared axis does not.
        ax.text(0.98, 0.92, IMT_LABELS[imt], transform=ax.transAxes,
                ha="right", va="top", fontsize=ANNOTATION_PT,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                      "pad": 1.0}, zorder=6)
        ax.grid(True, alpha=0.3)
        if not plotting.TITLES_IN_FIGURES:
            bssa.panel_label(ax, "abc"[row])

    axs[-1].set_xlabel("Number of events in last completeness period ($n$)")
    axs[1].set_ylabel(f"Ratio to {LABEL_AKI}")

    plotting.figure_title(
        fig,
        "Ground motion exceedance rate ratios at a 2475-year return period, "
        "by intensity measure",
    )
    
    plt.Line2D([], [], color=estimator_colour("bfree"), marker=M_BFREE,
                ls="none", markersize=5, label=LABEL_BFREE),
    plt.Line2D([], [], color=estimator_colour("jeffreys"), marker=M_BAYES,
                ls="none", markersize=5, label=LABEL_BAYES),
    plt.Line2D([], [], color="0.4", ls="--", lw=1, ),

    legend_below_axes(fig, [ax])
    return fig


@figure("return_period_sensitivity", (ESHM20, HAZARD))
def return_period_sensitivity(r):
    """Sensitivity of the comparison to the return period at which it is made."""
    sens = r[ESHM20]["rp_sensitivity"]
    hazard = r[HAZARD]
    fig, axs = plt.subplots(1, 2, figsize=size(0.62, original=(13, 5)))

    # Left: magnitude domain. The choice of m_RP largely sets the size of the difference.
    rp_m = sens["return_periods"]
    # Above ~300 yr, m_RP exceeds Mmax in a growing number of zones, so the median
    # there is taken over a shrinking and increasingly high-Mmax subset.
    n_zones = np.sum(~np.isnan(sens["ratio_bfree"]), axis=0)
    full = n_zones >= 0.75 * n_zones.max()
    for key, colour, label in (
        ("ratio_bfree", estimator_colour("bfree"), LABEL_BFREE),
        ("ratio_bayes", estimator_colour("jeffreys"), LABEL_BAYES),
    ):
        arr = sens[key]
        med = np.nanmedian(arr, axis=0)
        axs[0].plot(rp_m[full], med[full], color=colour, lw=LINE_THICK, label=label)
        axs[0].plot(rp_m, med, color=colour, lw=1, alpha=0.35)
        lo, hi = np.nanpercentile(arr, [25, 75], axis=0)
        axs[0].fill_between(rp_m[full], lo[full], hi[full], color=colour, alpha=0.15, lw=0)
    axs[0].axvline(250, color="0.3", ls=":", lw=LINE_THIN, label="$m_{250}$, as reported")
    axs[0].axvline(
        rp_m[full][-1],
        color="0.6",
        ls="-",
        lw=1,
        label="$m_{RP}>M_{max}$ for >25% of zones",
    )
    axs[0].set(
        xscale="log",
        yscale="log",
        xlabel="Magnitude return period (years)",
        ylabel=f"Ratio to {LABEL_AKI}",
    )
    panel_title(axs[0], "a", "Magnitude domain\ndepends on the magnitude chosen")
    axs[0].set_yticks([0.5, 1.0, 2.0])
    axs[0].set_xlim(rp_m.min(), 3000)

    # Right: hazard domain. m_250 enters nowhere in the integral.
    rp_h = hazard["records"][0]["curve"]["return_period"]
    for imt in IMT_ORDER:
        sel = [x for x in hazard["records"] if x["imt"] == imt]
        for name, estimator in (("bfree", "bfree"), ("bayes", "jeffreys")):
            med = np.nanmedian(np.array([x["curve"][name] for x in sel]), axis=0)
            axs[1].plot(
                rp_h,
                med,
                color=estimator_colour(estimator),
                ls=IMT_LINE_STYLES[imt],
                lw=LINE_BASE,
            )
    for rp in (475, 2475):
        axs[1].axvline(rp, color="0.3", ls=":", lw=LINE_THIN)
    axs[1].set(
        xscale="log",
        xlabel="Ground motion return period (years)",
        ylabel=f"Exceedance rate ratio to {LABEL_AKI}",
    )
    panel_title(axs[1], "b", "Hazard domain\nindependent of the magnitude chosen")

    for ax in axs:
        ax.axhline(1.0, color="0.4", ls="--", lw=1)
        ax.grid(True, which="both", alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mticker.NullLocator())

    estimator_handles = [
        legend_handle("bfree", LABEL_BFREE),
        legend_handle("jeffreys", LABEL_BAYES),
    ]
    imt_handles = [
        plt.Line2D([], [], color=plotting.NEUTRAL, ls=IMT_LINE_STYLES[imt],
                   lw=LINE_BASE, label=IMT_LABELS[imt])
        for imt in IMT_ORDER
    ]
    reference_handles = [
        plt.Line2D([], [], color="0.4", ls="--", lw=1, ),
        plt.Line2D([], [], color="0.3", ls=":", lw=LINE_THIN,
                   label="$m_{250}$, as reported"),
        plt.Line2D([], [], color="0.6", ls="-", lw=1,
                   label="$m_{RP}>M_{max}$ for >25% of zones"),
    ]
    legend_below_groups(fig, [
        ("Estimator, by colour", estimator_handles, 1),
        ("Intensity measure, by line style", imt_handles, 1),
        ("Reference", reference_handles, 1),
    ])
    return fig


@figure("hazard_fixed_rp_amplitude", (HAZARD_FIXED_RP,))
def hazard_fixed_rp_amplitude(r):
    """Percent amplitude change at fixed return periods, relative to the Aki-Utsu plug-in."""
    records = r[HAZARD_FIXED_RP]["records"]
    fig, ax = plt.subplots(figsize=size(0.62, original=(11, 5)))

    return_periods = records[0]["return_periods"]
    available_imts = list(dict.fromkeys(record["imt"] for record in records))
    imts = [imt for imt in IMT_ORDER if imt in available_imts]
    imts.extend(imt for imt in available_imts if imt not in imts)
    positions = []
    labels = []
    for imt_index, imt in enumerate(imts):
        for rp_index, rp_entry in enumerate(return_periods):
            pos = imt_index * len(return_periods) + rp_index
            positions.append(pos)
            labels.append(
                f"{IMT_LABELS.get(imt, imt)}\n{rp_entry['return_period']} yr"
            )
            group = [
                record for record in records
                if record["imt"] == imt
            ]
            for name, colour, marker, offset, label in (
                ("bfree", estimator_colour("bfree"), M_BFREE, -0.12, LABEL_BFREE),
                ("bayes", estimator_colour("jeffreys"), M_BAYES, 0.12, LABEL_BAYES),
            ):
                values = np.array([
                    record["return_periods"][rp_index]["ratios"][name]
                    for record in group
                    if record["return_periods"][rp_index]["ratios"][name] is not None
                ])
                percent = 100 * (values - 1)
                jitter = np.linspace(-0.035, 0.035, len(percent))
                ax.scatter(
                    pos + offset + jitter, percent, color=colour, marker=marker,
                    s=22, alpha=0.35, edgecolors="none",
                    label=label,
                )
                if len(percent):
                    median = float(np.median(percent))
                    ax.plot(
                        [pos + offset - 0.07, pos + offset + 0.07],
                        [median, median], color=colour, lw=LINE_THICK,
                    )

    ax.axhline(0, color="0.3", lw=1, ls="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Amplitude change relative to {LABEL_AKI} (%)")
    ax.set_xlim(-0.5, len(positions) - 0.5)
    ax.grid(alpha=0.3, axis="y")
    plotting.figure_title(
        fig,
        "Ground-motion amplitude change at fixed return periods",
    )
    legend_below_axes(fig, [ax])
    return fig


def report_numbers(r):
    """The ESHM20 figures quoted in the text."""
    eshm20 = r[ESHM20]
    zones, highlights = eshm20["zones"], eshm20["highlights"]
    for k, v in eshm20["summary"].items():
        print(f"{k:26s} {v}")
    print()
    print(f"Qualifying zones                     : {len(zones)}")
    print(f"Excluded because m_250 > lowest Mmax : {len(eshm20['skipped_mmax'])}")
    print(f"Median ratio, frequentist b-free     : {zones['ratio_bfree'].median():.2f}")
    print(f"Median ratio, Bayesian b-free        : {zones['ratio_bayes'].median():.2f}")
    print(
        f"Zones with Bayesian above Aki-Utsu   : {(zones['ratio_bayes'] > 1).sum()}"
        f" / {len(zones)}"
    )
    print(
        f"Zones with frequentist below Aki-Utsu: {(zones['ratio_bfree'] < 1).sum()}"
        f" / {len(zones)}"
    )
    print()
    for label, zid in highlights.items():
        row = zones[zones["zone_id"] == zid].iloc[0]
        print(
            f"Zone {label} ({zid}): n={row['n']}, m_250={row['m_250']:.1f}, "
            f"ratio_bfree={row['ratio_bfree']:.2f}, ratio_bayes={row['ratio_bayes']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--figure-dir", help="where to write; must already exist")
    parser.add_argument("--results-dir", help="where the result stores are")
    parser.add_argument(
        "--format", nargs="+", choices=FORMATS,
        help="one or more of: " + ", ".join(FORMATS)
              + "; BSSA wants eps or high-resolution pdf",
    )
    parser.add_argument(
        "--style", choices=("original", "screen", "bssa"),
        help="original draws at the wide sizes used before the figures were "
             "fitted to the page, which is what reads on a projector; screen "
             "keeps the on-figure titles at matplotlib's default width; bssa "
             "draws at the printed page width with the journal's type sizes "
             "and line weights, and drops the titles",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="report where each figure breaks the BSSA guidelines",
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME",
        help="produce only these figures; explicitly request research-only figures here",
    )
    parser.add_argument("--list", action="store_true", help="list the figure names and exit")
    parser.add_argument(
        "--numbers", action="store_true",
        help="also print the ESHM20 figures quoted in the text",
    )
    args = parser.parse_args()

    if args.list:
        for name, (_func, inputs) in REGISTRY.items():
            marker = " [research-only]" if name in RESEARCH_ONLY else ""
            print(f"{name:<28} {', '.join(inputs)}{marker}")
        return 0

    missing = [f"--{n.replace('_', '-')}" for n in ("figure_dir", "results_dir", "format", "style")
               if getattr(args, n) is None]
    if missing:
        parser.error("required: " + ", ".join(missing))

    if args.style == "bssa":
        bssa.apply()
        plotting.TITLES_IN_FIGURES = False
        print(
            f"BSSA style: Helvetica-compatible sans text and math, "
            f"{bssa.FULL_WIDTH_IN} in wide, "
            f"{bssa.RASTER_DPI} dpi"
        )
    elif args.style == "original":
        plotting.apply_sans_fonts()
        plotting.ORIGINAL_SIZES = True
        print("original style: the wide sizes used before the figures were "
              "fitted to the page")

    figure_dir, results_dir = Path(args.figure_dir), Path(args.results_dir)
    for label, path in (("--figure-dir", figure_dir), ("--results-dir", results_dir)):
        if not path.is_dir():
            raise SystemExit(f"{label} is not a directory: {path}")

    names = args.only if args.only is not None else [
        name for name in REGISTRY if name not in RESEARCH_ONLY
    ]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(unknown)}")

    results = Results(results_dir)
    violations = 0
    for name in names:
        func, inputs = REGISTRY[name]
        fig = func(results)
        if args.check:
            problems = bssa.check(fig)
            violations += len(problems)
            print(f"{name}: " + ("ok" if not problems else ""))
            for p in problems:
                print(f"    {p}")
        for path in save_figure(fig, name, figure_dir, args.format,
                                [results_dir / f for f in inputs]):
            print(f"saved {path}")

    if args.numbers:
        print()
        report_numbers(results)
    if args.check and violations:
        print(f"\n{violations} guideline violation(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
