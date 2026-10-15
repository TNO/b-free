"""
Figure settings for the Bulletin of the Seismological Society of America.

Every constraint here is quoted from the SSA Art Guidelines or the BSSA
Submission Guidelines, with the source noted, so that a reader can check the
rule rather than trust the code:

  https://www.seismosoc.org/publications/ssa-art-guidelines/
  https://www.seismosoc.org/publications/bssa-submission-guidelines/

The rule that drives everything else is the type size. BSSA asks for 10 to 12
point type in figures reproduced at 100%, and says the size after reduction
must be no smaller than 6 point. A figure drawn 17 inches wide and printed at
7.16 inches is reduced 2.4 times, which puts even 14 point type at 5.9 point. So
figures are drawn at the width they will be printed, and the type is set here
rather than per call.
"""

import matplotlib as mpl

# SSA-BSSA.cls sets \textwidth to 43.15pc, which is 7.16 inches, and the
# manuscript includes every figure at \textwidth. Drawing at anything narrower
# has LaTeX scale the result up, which is the resampling this module exists to
# avoid; measured in the built PDF, 6.5 inch figures were being placed at 7.16.
# The half width leaves a gutter between two figures set side by side.
FULL_WIDTH_IN = 7.16
HALF_WIDTH_IN = 3.47

# "use a type size between 10 and 12 points" and "use no more than a 2-point
# variation" (Text within Figures). Panel labels are separately specified as
# 12 point (Labels). Sitting at the bottom of the allowed range buys room,
# which multi-panel figures at the page width badly need.
FONT_SMALL = 10.0
FONT_BASE = 10.0
FONT_LARGE = 11.0
PANEL_LABEL_PT = 12.0

# "line weights of 0.5 to 1.0 point"; thinner than 0.5 may not print at all.
LINE_THIN = 0.5
LINE_BASE = 0.8
LINE_THICK = 1.0

# "between 500 and 900 dpi for combination line-art and gray-scale or color
# images", which is what these figures are.
RASTER_DPI = 600

# PDF is the submission format. EPS cannot carry alpha -- PostScript has no
# alpha channel -- and these figures rely on it: translucent uncertainty bands
# (fill_between at alpha=0.15), overlapping scatter at alpha=0.25, and every
# grid. Flattening those to opaque would let one band hide another, which
# changes what the figure says rather than how it looks. PNG is for the
# repository README and for previewing.
# Nothing reads this tuple yet; the format list is passed at the call site.
SUBMISSION_FORMATS = ("pdf",)
SANS_LATEX_PREAMBLE = r"\usepackage{amsmath}\usepackage{helvet}\usepackage{sfmath}"

RC = {
    "figure.figsize": (FULL_WIDTH_IN, FULL_WIDTH_IN * 0.6),
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "font.size": FONT_BASE,
    "axes.titlesize": FONT_LARGE,
    "axes.labelsize": FONT_BASE,
    "xtick.labelsize": FONT_SMALL,
    "ytick.labelsize": FONT_SMALL,
    "legend.fontsize": FONT_SMALL,
    "legend.title_fontsize": FONT_SMALL,
    "figure.titlesize": FONT_LARGE,
    "lines.linewidth": LINE_BASE,
    "axes.linewidth": LINE_THIN,
    "grid.linewidth": LINE_THIN,
    "xtick.major.width": LINE_THIN,
    "ytick.major.width": LINE_THIN,
    "xtick.minor.width": LINE_THIN,
    "ytick.minor.width": LINE_THIN,
    "patch.linewidth": LINE_THIN,
    "figure.dpi": RASTER_DPI,
    "savefig.dpi": RASTER_DPI,
    # Use Helvetica-compatible sans text and matching sans math.
    "text.usetex": True,
    "text.latex.preamble": SANS_LATEX_PREAMBLE,
    # PDF: type 42 embeds the TrueType outlines, which is what a compositor
    # wants. EPS: type 42 produces a file Ghostscript refuses outright
    # ("/invalidfont in definefont"), so the PostScript backend keeps type 3,
    # which embeds each glyph as a drawing procedure. Either way no font is
    # referenced externally, which is what "embed all fonts" is asking for.
    "pdf.fonttype": 42,
    "ps.fonttype": 3,
    # "Key lines or boxes around figures are not necessary and are discouraged."
    "legend.frameon": False,
    "savefig.bbox": "tight",
}


def apply():
    """Install the BSSA settings as matplotlib defaults for this process."""
    mpl.rcParams.update(RC)


def panel_label(ax, letter, *, x=-0.14, y=1.06):
    """Put a panel label outside the axes, at the top left.

    "Figure-part labels (a) (b) (c) etc. should be placed outside of and to the
    top left of the figure parts. Use lowercase letters placed in parentheses."
    """
    ax.text(
        x, y, f"({letter})",
        transform=ax.transAxes,
        fontsize=PANEL_LABEL_PT,
        va="bottom",
        ha="left",
    )


def _reaches_the_edge(fig):
    """Report content running to the edge of the saved image.

    A label longer than its panel is tall gets cut there, and neither the type
    sizes nor the line weights say so. Measured on the saved bytes rather than
    on the figure's own coordinates, because those do not say what savefig will
    crop: with bbox_inches="tight" the canvas grows around the artists it knows
    about, and an artist it mismeasures is silently trimmed.
    """
    import io

    import matplotlib.image as mpimg
    import numpy as np

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image = mpimg.imread(buffer)[..., :3]
    ink = image.min(axis=2) < 0.9
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return []
    margins = {
        "top": int(rows.min()),
        "bottom": int(image.shape[0] - 1 - rows.max()),
        "left": int(cols.min()),
        "right": int(image.shape[1] - 1 - cols.max()),
    }
    touching = sorted(side for side, m in margins.items() if m < 1)
    if touching:
        return [f"content runs to the {', '.join(touching)} edge of the saved "
                "image, so something is cut off there"]
    return []


def check(fig, width_in=FULL_WIDTH_IN):
    """Report where a figure breaks the guidelines, as a list of strings.

    A style that cannot be checked is a style that quietly stops being applied,
    so this walks the figure as drawn rather than trusting the rcParams.
    """
    problems = []
    drawn_width = fig.get_size_inches()[0]
    reduction = drawn_width / width_in

    sizes = {round(t.get_fontsize(), 1) for t in fig.findobj(mpl.text.Text)
             if t.get_text().strip()}
    if sizes:
        effective = {round(s / reduction, 1) for s in sizes}
        if min(effective) < 6.0:
            problems.append(
                f"type as small as {min(effective)} pt after {reduction:.2f}x "
                f"reduction to {width_in}in (floor is 6 pt)"
            )
        if max(sizes) - min(sizes) > 2.0:
            problems.append(
                f"type varies by {max(sizes) - min(sizes):.1f} pt "
                f"({min(sizes)}-{max(sizes)}); the limit is 2 pt"
            )

    widths = {round(ln.get_linewidth(), 2) for ln in fig.findobj(mpl.lines.Line2D)
              if ln.get_linestyle() != "None" and ln.get_linewidth() > 0}
    if widths:
        effective = {round(w / reduction, 2) for w in widths}
        thin = sorted(w for w in effective if w < LINE_THIN)
        thick = sorted(w for w in effective if w > LINE_THICK)
        if thin:
            problems.append(f"line weights below 0.5 pt after reduction: {thin}")
        if thick:
            problems.append(f"line weights above 1.0 pt after reduction: {thick}")

    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        problems.append(
            "has a suptitle; BSSA asks that a figure not carry text that "
            "belongs in its caption"
        )
    problems.extend(_reaches_the_edge(fig))
    return problems
