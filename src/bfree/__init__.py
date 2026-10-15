"""
Estimators for Gutenberg-Richter earthquake size distributions.

The public API is re-exported here so that callers write ``from bfree import
bfree_survival_rounded`` rather than reaching into the module layout, which is free to
change.
"""

from bfree.estimators import (
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
    difference_statistics_continuous,
    difference_statistics_rounded,
    gr_survival,
    magnitude_statistics_continuous,
    magnitude_statistics_rounded,
    survival_to_pdf,
)
from bfree.provenance import record_figure

__all__ = [
    "LOG10",
    "b_aki",
    "b_assembled",
    "b_grouped",
    "b_ogata",
    "bayesian_pdf_continuous",
    "bayesian_pdf_rounded",
    "bayesian_survival_continuous",
    "bayesian_survival_rounded",
    "bfree_pdf_continuous",
    "bfree_pdf_rounded",
    "bfree_survival_continuous",
    "bfree_survival_rounded",
    "conjugate_prior_continuous",
    "conjugate_prior_rounded",
    "difference_statistics_continuous",
    "difference_statistics_rounded",
    "gr_survival",
    "magnitude_statistics_continuous",
    "magnitude_statistics_rounded",
    "record_figure",
    "survival_to_pdf",
]
