# Figures

The registry contains eleven figure builders. By default, `make_figures.py`
produces the seven-figure paper set. Cross-formulation comparisons, the extended
catalog-size sweep, and the fixed-return-period amplitude figure are research-
only. Select the former with `--only frequentist_by_formulation bayesian_by_formulation error_by_b_and_catalog_size`; the
fixed-return-period figure requires its separate result store and is selected
with `--only hazard_fixed_rp_amplitude`. `--list` marks all four.

| Name | Reads | Shows |
|---|---|---|
| `frequentist_survival` | `analysis_results`, `analysis_results_continuous` | The frequentist estimators against the truth, as survival functions. Two rows: continuous catalogues above, rounded below, on a shared scale per column. The plug-in bias appears in both, so it is not a rounding artefact. |
| `bayesian_survival` | same | The Bayesian posterior-predictive estimators under the reference and conjugate priors, as survival functions. |
| `frequentist_by_formulation` | `analysis_results` | All three formulations — magnitudes, non-overlapping differences, overlapping differences — for the frequentist estimators, with mean, median and relative RMSE. Research-only; generate explicitly with `--only frequentist_by_formulation`. |
| `bayesian_by_formulation` | `analysis_results` | The same three formulations for the Bayesian estimators. Research-only; generate explicitly with `--only bayesian_by_formulation`. |
| `bayesian_priors` | `analysis_results` | What the priors assume, in three parameterisations and as a prior predictive, continuous above and rounded below. The reference priors are improper, so they are drawn unnormalised and scaled to meet at one point. |
| `error_by_b_and_catalog_size` | `analysis_results` | The ratio of estimated to true survival function across every simulated catalogue size and true b-value, as a matrix. Research-only; generate explicitly with `--only error_by_b_and_catalog_size`. |
| `eshm20_ratio_by_catalog_size` | `eshm20_results` | Every qualifying ESHM20 zone: the ratio to the Aki-Utsu plug-in against the number of events, with the marginal distribution beside it. |
| `eshm20_example_zones` | `eshm20_results` | Three zones in detail — a small catalogue, a large one, and the largest difference — with the zone ID in the top right of each upper panel and the event count, b-value and reference magnitude in the lower left. Magnitude limits are shared across all panels, and y-limits are shared within each row. The dotted line marks the 250-year magnitude used for the exceedance-probability comparison. That return period is a reading level chosen here; only the a-value and b-value behind the magnitude come from ESHM20. |
| `hazard_propagation` | `hazard_results` | The ratio to the Aki-Utsu plug-in at each stage: the unbounded size distribution, the same distribution truncated at the maximum magnitude, and then the ground-motion exceedance rate for three intensity measures. The stages are not a transfer function, since the magnitude-domain ratio is read at a magnitude the hazard integral does not single out; each is compared only with the reference. |
| `return_period_sensitivity` | `eshm20_results`, `hazard_results` | How the comparison depends on the return period it is made at, in the magnitude domain and in the hazard domain. |
| `hazard_fixed_rp_amplitude` | `hazard_fixed_rp_results` | The percentage change in PGA or spectral acceleration at 475- and 2475-year return periods, relative to Aki-Utsu, with zone-level values and medians. Research-only; requires its separate store. |

Two further figures come from analysis scripts rather than from the registry,
and `make figures` runs both: `analyse_mc_sensitivity.py` on the placement of
the reference magnitude relative to completeness, and
`analyse_plugin_estimators.py` on the four plug-in estimators
(`plugin_corrections`, the supplement's two-by-two figure). Both take `--style`,
so each is drawn in the series of the directory it is written to.

The fixed-return-period amplitude figure uses a separate hazard result store.
Generate it with `scripts/analyse_hazard_fixed_rp.py`, then select it with:

```bash
mkdir -p figures/hazard
python scripts/make_figures.py --figure-dir figures/hazard \
  --results-dir results --format png --style screen \
  --only hazard_fixed_rp_amplitude
```

The plotted percentage is the b-free amplitude divided by the Aki-Utsu
amplitude at the same annual exceedance rate, minus one; it is not a percentage
change in exceedance rate.

## Color encoding

Estimator identity is consistently encoded with a distinct, color-vision-safe
hue. A second dimension, such as magnitude formulation or intensity measure, is
encoded by line style or marker and identified in the legend.

![The frequentist estimators, colour series](images/frequentist_survival.png)

*Hue identifies the estimator; line style identifies the formulation.*

## Figure sizes

`make figures` writes two color series, each to its own directory under
`figures/`, so they can be chosen between. Both come from the same
results and the same code.

| Target | Directory | What it is |
|---|---|---|
| `figures-bssa` | `bssa-colour/` | the page series, 7.16 in wide, journal type sizes |
| `figures-original` | `original/` | the pre-page sizes, which is what reads on a projector |

The figures as first submitted are kept as images with the manuscript, which is
where a record of what was sent belongs. They are not regenerated here. They
differ from the `original` series only in things since corrected: a continuous
panel carrying the lattice reference $m_0^{*}$, one legend drawn over its tick
labels, and the same estimator named two ways in one figure.

## Sizes and formats

There are three styles, and the page one is not simply better than the others.

`--style bssa` draws each figure at the width it will be printed and sets type
and line weights from `src/bfree/bssa.py`, where each constant carries the rule
it comes from. This matters because type is measured in points: a figure drawn
large and reduced to fit a page carries its type down with it, and the same
labels that are legible on screen fall below the legible floor in print.

`--style original` draws at the sizes used before the figures were fitted to
the page, 13 to 17 inches wide, saved at 150 dpi and cropped to the ink. Those
are what reads on a projector and in a document viewer, where a page-width figure
scaled up is coarse. Text uses Helvetica-compatible sans fonts, with matching
sans-serif math in both print styles. Each figure carries its original size
beside its printed one at the point where it is drawn, so the two cannot drift
apart.

`--style screen` keeps matplotlib's defaults and the on-figure titles. It draws
at 6.4 inches, so the result is a thumbnail; `original` is usually what is
wanted instead.

`scripts/analyse_mc_sensitivity.py` draws its figure from simulation rather
than from a stored result, so it is a separate command, but it takes the same
`--figure-dir`, `--format` and `--style` as `make_figures.py` and writes through
the same provenance record. Each figure series calls it.

`--format` takes `png`, `pdf` and `eps`:

- **PDF** is the best vector option here. It preserves transparency, which
  these figures use to show overlapping density.
- **EPS** is vector but cannot carry transparency; overlapping points flatten
  to solid and the density they encode is lost. Produced on request.
- **PNG** is raster, written at 600 dpi under the BSSA style.

PNG and PDF are byte-reproducible for an unchanged figure. EPS carries a
creation date in a header comment, which the manifest hash skips.

## Provenance

Each written figure gets a row in `MANIFEST.tsv` beside it:

```
figure          image_md5   code_commit   inputs_opened          generated
```

The hash is of the image rather than the file, because matplotlib stamps its
own version into PNG metadata and two pixel-identical files would otherwise
differ. The commit is suffixed `-dirty` if the tree was not clean, since a
clean hash against a dirty tree is worse than none.

The manifest is written by the same call that writes the figure, so it cannot
be forgotten separately. A figure set that is half from one version of the code
and half from another is the specific failure this prevents.

## Checking

`--check` reports each figure against the guidelines: type size after
reduction, the spread of type sizes, line weights, and text that belongs in a
caption. It does **not** detect overlapping labels or legends, so a clean
report is necessary rather than sufficient — look at the figures too.
