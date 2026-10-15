"""
Shared plotting machinery for the manuscript figures.

Colour encodes the estimator and line style the statistic it is built on, so a
reader who has learned one figure has learned them all. The label vocabulary is
defined once here rather than per figure, so captions can be checked against it.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from bfree.bssa import (
    LINE_BASE,
    LINE_THICK,
    LINE_THIN,
    SANS_LATEX_PREAMBLE,
    panel_label,
)
from bfree.provenance import record_figure

FORMATS = ("png", "pdf", "eps")


def page_width():
    """The width figures are drawn at, in inches, from the active style."""
    return plt.rcParams["figure.figsize"][0]


def apply_sans_fonts():
    """Use Helvetica-compatible sans text and math in LaTeX-rendered figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
        "text.usetex": True,
        "text.latex.preamble": SANS_LATEX_PREAMBLE,
    })


# Draw at the sizes used before the figures were fitted to the page. Those are
# wide enough to read on a screen or a projector, which the 6.5 inch page
# versions are not, so they stay available for talks and for a submission that
# does not use the journal series.
ORIGINAL_SIZES = False


def size(aspect, original=None):
    """A figure size with the given height-to-width ratio, at the page width.

    Figures are drawn at the size they will be printed. Drawing large and
    letting the journal reduce the result is what puts type below the legible
    floor, because a point is an absolute unit and reduction scales it.

    *original* is the size this figure was drawn at before that change, used
    verbatim under the original style. It is given at the call site rather than
    in a table so that the two sizes for a figure are read together.
    """
    if ORIGINAL_SIZES and original is not None:
        return original
    w = page_width()
    return (w, w * aspect)


# BSSA asks that a figure not carry text belonging in its caption, which is
# what these titles are. They stay available for screen use, where having the
# model and sample size on the figure is worth more than the rule.
TITLES_IN_FIGURES = True


def figure_title(fig, text):
    """Set a figure-level title, unless the active style suppresses titles."""
    if TITLES_IN_FIGURES:
        fig.suptitle(text)


def panel_title(ax, letter, text):
    """Add a panel label and a title only if the axes does not have one.

    BSSA asks that a figure not carry text belonging in its caption, and that
    panel labels sit outside the axes at the top left rather than above it. In
    screen styles, preserve builder-supplied titles instead of overlapping them.
    """
    has_title = any(ax.get_title(loc=loc) for loc in ("left", "center", "right"))
    if TITLES_IN_FIGURES and not has_title:
        ax.set_title(text)
    panel_label(ax, letter)


def legend_below(fig, handles, labels, ncol=None, title=None):
    """Put one legend under the figure, clear of the axes.

    A legend inside an axes covers the curves it describes, and which corner is
    free changes with the data. tight_layout does not account for a figure-level
    legend, so the band it needs is measured from the type size and reserved.
    """
    ncol = ncol or min(len(labels), 4)
    rows = -(-len(labels) // ncol) + (1 if title else 0)
    band = (rows * 1.8 * plt.rcParams["font.size"] / 72 + 0.15) / fig.get_figheight()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, title=title,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, min(band, 0.4), 1, 1))


def legend_below_axes(fig, axes, ncol=None, title=None):
    """One legend under the figure, gathering the labelled artists in *axes*.

    Labels repeated across panels are drawn once.
    """
    handles, labels = [], []
    for ax in np.ravel(np.asarray(axes, dtype=object)):
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend_below(fig, handles, labels, ncol=ncol, title=title)


def legend_below_groups(fig, groups):
    """One titled legend per encoding channel, side by side under the figure.

    Where colour names the estimator and line style names the formulation, a
    single list of swatches does not say which is which, and no swatch is a
    curve that appears in the plot. Splitting the key by channel and titling each
    group clarifies which swatch belongs to which encoding.

    *groups* is a sequence of (title, handles, ncol).
    """
    rows = max(-(-len(h) // n) for _, h, n in groups) + 1
    band = (rows * 1.5 * plt.rcParams["font.size"] / 72 + 0.15) / fig.get_figheight()
    band = min(band, 0.45)
    for i, (title, handles, ncol) in enumerate(groups):
        # Hung from the top of the band, not the bottom, so blocks of different
        # length still line their titles up.
        legend = fig.legend(
            handles, [h.get_label() for h in handles],
            loc="upper center", ncol=ncol, title=title, frameon=False,
            bbox_to_anchor=((i + 0.5) / len(groups), band),
        )
        legend._legend_box.align = "left"
        fig.add_artist(legend)
    fig.tight_layout(rect=(0, band, 1, 1))


# A distinct, colour-vision-safe palette. The hues encode estimators; line style
# continues to encode the data formulation.
COLORS = {
    "aki": "#0072B2",
    "ogata": "#D55E00",
    "grouped": "#009E73",
    "assembled": "#8B4513",
    "bfree": "#CC79A7",
    "jeffreys": "#56B4E9",
    "conjugate": "#E69F00",
}

FRAMEWORKS = [
    ("mag", "solid", "Magnitudes"),
    ("diff_nol", "dashed", "Differences (non-overlapping)"),
    ("diff_ol", "dotted", "Differences (overlapping)"),
]
KEYS = {
    ("aki", "mag"): "b_aki",
    ("aki", "diff_nol"): "b_pos_aki_nol",
    ("aki", "diff_ol"): "b_pos_aki_ol",
    ("ogata", "mag"): "b_ogata",
    ("ogata", "diff_nol"): "b_pos_ogata_nol",
    ("ogata", "diff_ol"): "b_pos_ogata_ol",
    ("grouped", "mag"): "b_grouped",
    ("grouped", "diff_nol"): "b_pos_grouped_nol",
    ("grouped", "diff_ol"): "b_pos_grouped_ol",
    # The (n-1)/n factor assumes independent observations, so the assembled
    # estimator is carried for magnitudes only, not for overlapping differences.
    ("assembled", "mag"): "b_assembled",
    ("bfree", "mag"): "direct_mag",
    ("bfree", "diff_nol"): "direct_diff_nol",
    ("bfree", "diff_ol"): "direct_diff_ol",
    ("jeffreys", "mag"): "direct_mag_jeffreys",
    ("jeffreys", "diff_nol"): "direct_diff_nol_jeffreys",
    ("jeffreys", "diff_ol"): "direct_diff_ol_jeffreys",
    ("conjugate", "mag"): "direct_mag_conjugate",
    ("conjugate", "diff_nol"): "direct_diff_nol_conjugate",
    ("conjugate", "diff_ol"): "direct_diff_ol_conjugate",
}

# For key swatches that stand for a line style rather than an estimator.
NEUTRAL = "0.25"


def estimator_colour(estimator):
    """The colour for one estimator."""
    return COLORS[estimator]


def line_kwargs(estimator, framework="mag", **overrides):
    """Style for one estimator drawn under one formulation.

    Keeping this in one place makes the estimator colour and formulation line
    style consistent across figures.
    """
    style = {
        "color": COLORS[estimator],
        "ls": {name: line_style for name, line_style, _ in FRAMEWORKS}[framework],
    }
    style.update(overrides)
    if style.get("dashes", False) is None:
        del style["dashes"]
    return style


def legend_handle(estimator, label):
    """A legend entry matching `line_kwargs` for the magnitude formulation."""
    kw = line_kwargs(estimator)
    return plt.Line2D([], [], lw=LINE_BASE, label=label, **kw)

# The vertical axis names the probability, not the method used to estimate it.
SURVIVAL_LABEL = r"$P(M \geq m)$"
XLABEL_CONT = r"$m-m_0$"
XLABEL_ROUNDED = r"$m-m_0^{*}$"
PDF_LABEL_CONT = r"$\hat{f}_M(m)$"
PDF_LABEL_ROUNDED = r"$\hat{f}^{*}_M(m)$"
PDF_PRED_CONT = r"$\tilde{f}_M(m)$"
PDF_PRED_ROUNDED = r"$\tilde{f}^{*}_M(m)$"
RATIO_LABEL = "estimated / truth"
# A posterior predictive is not an estimate of a parameter, so the Bayesian
# figures say what they draw.
RATIO_LABEL_PREDICTED = "predicted / truth"
# Root mean square of the ratio's deviation from one, sqrt(E[(S_hat/S - 1)^2]),
# so it is dimensionless. An absolute RMSE would shrink wherever S is small,
# which is where these figures show the error growing.
RMSE_LABEL = "relative RMSE"

# The bottom of a rounded catalogue is the lower bin edge at -delta, where every
# survival function is one. Starting at zero cropped it, so the panels opened
# part-way up the first bin. One bin of margin below the reference shows it, and
# the continuous row uses the same range so the two can be read against each
# other.
X_MAX = 5.0


def x_limits(delta):
    """Axis range in excess above the reference, with room for the first bin."""
    return [-2.0 * delta, X_MAX]

# Suppressed so that two runs of the same figure give the same bytes. The PDF
# and EPS backends otherwise stamp the wall clock into CreationDate, and name
# their own version in Creator and Producer, which would make the manifest hash
# change on a matplotlib patch release without the figure changing at all.
_METADATA = {
    "png": {},
    "pdf": {"CreationDate": None, "Creator": "bfree", "Producer": "bfree"},
    "eps": {},
}


def save_figure(fig, name, figure_dir, formats, inputs=()):
    """Write one figure in each requested format and record its provenance."""
    figure_dir = Path(figure_dir)
    # The original series was written this way; the page series takes its
    # resolution from the style, and must not be cropped to the ink.
    extra = {"dpi": 150, "bbox_inches": "tight"} if ORIGINAL_SIZES else {}
    written = []
    for fmt in formats:
        path = figure_dir / f"{name}.{fmt}"
        fig.savefig(path, metadata=_METADATA[fmt], **extra)
        record_figure(path, inputs)
        written.append(path)
    plt.close(fig)
    return written


def model_title(meta, rounded):
    """Heading naming the model a panel is drawn under, or None under BSSA.

    The model and bin width are caption material, and the string is wider than
    a panel at the page width, so it is dropped with the other figure titles.
    """
    if not TITLES_IN_FIGURES:
        return None
    if rounded:
        return (
            f"Rounded catalogues, $2\\delta={2 * meta['delta']:g}$:  "
            r"$K\sim\mathrm{Geom}(q)$"
        )
    return "Continuous catalogues:  " + r"$X=M-m_0\sim\mathrm{Exp}(\beta)$"


def estimator_panels(
    entry,
    estimators,
    estimator_labels,
    frameworks=("mag",),
    quantity="pdf",
    show_spread=False,
    rounded=True,
    key="Estimator",
    ratio_label=RATIO_LABEL,
    predictive=False,
):
    """
    Left panel: mean estimated size distribution. Middle: ratio to the ground truth.

    With show_spread, the middle panel also carries the median ratio and a third
    panel shows the RMSE. The mean alone hides that an estimator can be unbiased
    while falling below the truth for most individual catalogues.
    """
    meta = entry["metadata"]
    mrange = entry[f"{quantity}_mrange"]

    ncols = 3 if show_spread else 2
    # A two-channel key is two titled blocks, so it is taller than a single list
    # and the figure has to carry it rather than the panels giving up height.
    aspect = 0.52 if show_spread else (0.72 if len(frameworks) > 1 else 0.55)
    fig, axs = plt.subplots(
        nrows=1, ncols=ncols,
        figsize=size(aspect, original=(8 * ncols, 6)),
    )
    axs[0].plot(
        mrange, entry[f"{quantity}_ground_truth"], "k-", lw=LINE_THICK, alpha=0.5,
        label="Ground truth"
    )

    for estimator in estimators:
        for framework in frameworks:
            e = entry["estimators"].get(KEYS[(estimator, framework)])
            if e is None:
                continue
            kw = line_kwargs(estimator, framework, alpha=0.7, lw=LINE_BASE)
            axs[0].plot(mrange, e[f"{quantity}_mean"], **kw)
            axs[1].plot(mrange, e[f"{quantity}_ratio_mean"], **kw)
            if show_spread:
                # The median keeps the input-data line style and is separated
                # from the mean by weight: line style is already spent, and
                # drawing it dotted would collide with overlapping differences.
                axs[1].plot(
                    mrange,
                    e[f"{quantity}_ratio_median"],
                    **line_kwargs(estimator, framework, alpha=0.55, lw=LINE_THIN),
                )
                axs[2].plot(mrange, e[f"{quantity}_ratio_rmse"], **kw)

    handles = [legend_handle(e, estimator_labels[e]) for e in estimators]
    framework_handles = []
    if len(frameworks) > 1:
        for name, ls, label in FRAMEWORKS:
            if name in frameworks:
                framework_handles.append(plt.Line2D(
                    [], [], color=NEUTRAL, ls=ls, lw=LINE_BASE, label=label))

    figure_title(
        fig,
        f"Average over {meta['n_simulations']:,} simulated catalogues of "
        f"{meta['n_events']} events with $b_{{true}}={meta['b_true']}$",
    )

    if quantity == "sf":
        first = SURVIVAL_LABEL
    else:
        first = (PDF_PRED_ROUNDED if rounded else PDF_PRED_CONT) if predictive \
            else (PDF_LABEL_ROUNDED if rounded else PDF_LABEL_CONT)
    ylabels = [first, ratio_label]
    if show_spread:
        ylabels.append(RMSE_LABEL)
    xlabel = XLABEL_ROUNDED if rounded else XLABEL_CONT
    for ax, ylabel in zip(axs, ylabels, strict=True):
        ax.set(yscale="log", xlim=x_limits(meta["delta"]), xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, alpha=0.3, which="both")
    for ax in axs[1:]:
        ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    panel_descriptions = ["Estimated distribution", "Ratio to truth", "Relative RMSE"]
    for index, ax in enumerate(axs):
        panel_title(ax, chr(ord("a") + index), panel_descriptions[index])
    stat_handles = []
    if show_spread:
        axs[2].set_yscale("linear")
        axs[1].axhline(1.0, color="0.3", lw=LINE_THIN, ls="-", zorder=0)
        stat_handles = [
            plt.Line2D([], [], color=NEUTRAL, ls="-", lw=LINE_BASE, label="mean"),
            plt.Line2D([], [], color=NEUTRAL, ls="-", lw=LINE_THIN, alpha=0.55,
                       label="median"),
        ]
    groups = [(f"{key}, by colour", handles, 1)]
    if framework_handles:
        groups.append(("Input data, by line style", framework_handles, 1))
    if stat_handles:
        groups.append(("Statistic, by line weight", stat_handles, 1))
    if len(groups) > 1:
        legend_below_groups(fig, groups)
    else:
        legend_below(fig, handles, [h.get_label() for h in handles],
                     title=model_title(meta, rounded))
    return fig


def _draw_estimator_row(
    entry, estimators, estimator_labels, axes, rounded, xlim, frameworks=("mag",),
    quantity="pdf", label_x=True, ratio_label=RATIO_LABEL, predictive=False,
):
    """Draw one data type into a (pdf, ratio, rmse) row of axes."""
    meta = entry["metadata"]
    x = np.asarray(entry[f"{quantity}_mrange"], dtype=float)
    if rounded:
        x = x - meta["mc"]  # stored as absolute magnitude
    ax_pdf, ax_ratio, ax_rmse = axes

    ax_pdf.plot(x, entry[f"{quantity}_ground_truth"], "k-", lw=LINE_THICK, alpha=0.5)
    for estimator in estimators:
        for framework in frameworks:
            e = entry["estimators"].get(KEYS[(estimator, framework)])
            if e is None:
                continue
            kw = line_kwargs(estimator, framework, alpha=0.8, lw=LINE_BASE)
            ax_pdf.plot(x, e[f"{quantity}_mean"], **kw)
            ax_ratio.plot(x, e[f"{quantity}_ratio_mean"], **kw)
            ax_ratio.plot(
                x,
                e[f"{quantity}_ratio_median"],
                **line_kwargs(estimator, framework, ls=(0, (1, 1)),
                              alpha=0.9, lw=LINE_BASE, dashes=None),
            )
            ax_rmse.plot(x, e[f"{quantity}_ratio_rmse"], **kw)

    handles = [plt.Line2D([], [], color="k", lw=LINE_THICK, alpha=0.5, label="Ground truth")]
    handles += [legend_handle(e, estimator_labels[e]) for e in estimators]
    title = model_title(meta, rounded)
    xlabel = XLABEL_ROUNDED if rounded else XLABEL_CONT
    if quantity == "sf":
        pdf_label = SURVIVAL_LABEL
    else:
        pdf_label = (PDF_PRED_ROUNDED if rounded else PDF_PRED_CONT) if predictive \
            else (PDF_LABEL_ROUNDED if rounded else PDF_LABEL_CONT)

    if title:
        ax_pdf.set_title(title, loc="left")
    ax_pdf.set(yscale="log", ylabel=pdf_label)
    ax_ratio.set(yscale="log", ylabel=ratio_label)
    ax_rmse.set(ylabel=RMSE_LABEL)
    ax_ratio.axhline(1.0, color="0.3", lw=LINE_THIN, ls="-", zorder=0)
    for ax in axes:
        ax.set(xlim=xlim)
        if label_x:
            ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3, which="both")
    # Labelling every minor tick collides in a panel this short, so only a few
    # subdivisions of each decade are named.
    ax_ratio.yaxis.set_major_locator(mticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax_ratio.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:g}"))
    ax_ratio.yaxis.set_minor_formatter(mticker.NullFormatter())
    return handles


def estimator_matrix_pair(
    entry_cont,
    entry_rounded,
    cont_estimators,
    rounded_estimators,
    labels,
    frameworks=("mag",),
    quantity="pdf",
    key="Estimator",
    ratio_label=RATIO_LABEL,
    predictive=False,
):
    """Continuous and rounded cases as two aligned rows sharing a scale per column.

    One label mapping serves both rows, so an estimator drawn in each cannot end
    up named differently in the two.
    """
    # Not sharex: the two rows are measured from different references, m_0 and
    # the bin centre m_0*, so each carries its own axis. The limits are set
    # explicitly in the row, which is what keeps the columns comparable. The
    # rounded bin width sets both, so the continuous row is not cropped
    # differently from the one whose first bin has to be visible.
    # 0.74 rather than less: the rotated "estimated / truth" is longer than a
    # shorter panel, and was being cut off at the top of the saved figure.
    xlim = x_limits(entry_rounded["metadata"]["delta"])
    fig, axs = plt.subplots(2, 3, figsize=size(0.74, original=(17, 10)))
    _draw_estimator_row(
        entry_cont,
        cont_estimators,
        labels,
        axs[0],
        rounded=False,
        xlim=xlim,
        frameworks=frameworks,
        quantity=quantity,
        ratio_label=ratio_label,
        predictive=predictive,
    )
    handles = _draw_estimator_row(
        entry_rounded,
        rounded_estimators,
        labels,
        axs[1],
        rounded=True,
        xlim=xlim,
        frameworks=frameworks,
        quantity=quantity,
        ratio_label=ratio_label,
        predictive=predictive,
    )

    # Align each column so the two cases can be read against each other directly.
    for top, bottom in zip(axs[0], axs[1], strict=True):
        lo = min(top.get_ylim()[0], bottom.get_ylim()[0])
        hi = max(top.get_ylim()[1], bottom.get_ylim()[1])
        top.set_ylim(lo, hi)
        bottom.set_ylim(lo, hi)

    panel_descriptions = (
        "Continuous: estimated distribution",
        "Continuous: ratio to truth",
        "Continuous: relative RMSE",
        "Rounded: estimated distribution",
        "Rounded: ratio to truth",
        "Rounded: relative RMSE",
    )
    for letter, ax, description in zip("abcdef", axs.ravel(), panel_descriptions, strict=True):
        panel_title(ax, letter, description)

    meta = entry_rounded["metadata"]
    figure_title(
        fig,
        f"{meta['n_simulations']:,} simulated catalogues of {meta['n_events']} events "
        f"with $b_{{true}}={meta['b_true']}$",
    )
    # The rounded row draws every estimator the continuous row does, so its
    # handles cover both. Both rows draw the mean and the median, which only
    # the suptitle used to say, and BSSA drops suptitles.
    stat_handles = [
        plt.Line2D([], [], color=NEUTRAL, ls="-", lw=LINE_BASE, label="mean"),
        plt.Line2D([], [], color=NEUTRAL, ls=(0, (1, 1)), lw=LINE_BASE, label="median"),
    ]
    legend_below_groups(fig, [
        (f"{key}, by colour", handles, 1),
        ("Statistic, by line style", stat_handles, 1),
    ])
    return fig
