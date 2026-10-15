# b-free

Direct estimation of Gutenberg–Richter earthquake size distributions, without
first estimating a *b*-value.

The usual route to an earthquake size distribution estimates *b* and then
substitutes it into the Gutenberg–Richter law. That plug-in step is biased even
when the *b*-value estimate is not, by Jensen's inequality, and the bias grows
with extrapolation distance — exactly where seismic hazard reads the
distribution. The *b*-free estimators avoid the intermediate quantity and
estimate the size distribution directly, in a frequentist form that is
unbiased for the survival function and a Bayesian form that returns a posterior
predictive distribution.

This repository holds the estimators, the simulation and analysis code that
characterises them, and the figure generation.

## Installation

```bash
uv sync --group dev
```

The library needs only `numpy` and `scipy`. Optional dependency groups add
`zarr` (`analysis`), `geopandas`/`pandas`/`openpyxl` (`eshm20`) and
`matplotlib` (`figures`); `--group dev` installs all of them. The hazard
analysis needs OpenQuake in a separate interpreter — see
[requirements_hazard.txt](requirements_hazard.txt).

## Usage

```python
from bfree import bfree_survival_rounded, magnitude_statistics_rounded

n, t, w = magnitude_statistics_rounded(magnitudes, m0=2.0, delta=0.05)
sf = bfree_survival_rounded(w, n, k=range(0, 40))   # P(M >= m0 - delta + 2*delta*k)
```

Magnitudes reported as real numbers take the `_continuous` forms instead:

```python
from bfree import bfree_survival_continuous, magnitude_statistics_continuous

n, t = magnitude_statistics_continuous(magnitudes, m0=2.0)
sf = bfree_survival_continuous(t, n, m=np.linspace(0, 4, 41))   # P(M >= m0 + m)
```

The two are different models of the data and the estimators do not carry
across, so every function says in its name which case it assumes. The names say
rounded because that is a property of the catalogue; the lattice is a property
of the model, and the word is kept for the grid itself. See
[docs/api.md](docs/api.md).

## Getting the data

The analyses read reduced results from `results/`. There are two ways to get
them, and `make help` lists both:

```bash
make data          # download the reduced results from Zenodo (small, fast)
make data-full     # also the raw catalogues, around 1.3 GB
make simulate      # generate everything locally
```

Generating from scratch simulates 10 million catalogues for each combination of
`b` in {0.8, 1.0, 1.2} and `n` in {50, 100, 500, 1000}. It takes hours, about
1.4 GB of disk and roughly 9 GB of memory at peak. Pass a smaller run for a
smoke test:

```bash
make simulate NSIM=1e4
```

Downloads are verified against the MD5 checksums the Zenodo record reports.

## Publishing the data

`make package` assembles a deposit — the reduced results, the zipped catalogue
store on request, and a manifest recording sizes, checksums and the commit that
produced them:

```bash
make package                       # -> dist/zenodo/
make package DIST=/tmp/deposit     # somewhere else
```

The packaging and retrieval scripts share one list of file names, so the
deposit a run produces is the deposit the download expects.

## Figures

```bash
make figures        # both colour series
make figure-check   # report against the BSSA guidelines
```

Two colour series are written, each to its own directory under `figures/`:
`bssa-colour/` at the printed page width and `original/` at the wider sizes used
before the figures were fitted to the page, which is what reads on a projector.
Both come from the same results and the same code.

`--style bssa` draws at the width the figure will be printed, because type is
measured in points and a figure drawn large and reduced to fit carries type
below the legible floor. `--format` takes `png`, `pdf` and `eps`. PDF and PNG
are byte-reproducible for an unchanged figure; EPS cannot carry transparency,
which these figures use, so PDF is the better vector format here.

Every figure written is recorded in `MANIFEST.tsv` beside it, with a hash of
the image, the commit that produced it and hashes of the data it read. The
manifest is written by the run that writes the figure, so the two cannot drift
apart.

See [docs/figures.md](docs/figures.md) for what each figure shows, and
[docs/scripts.md](docs/scripts.md) for the command-line tools.

## Tests

```bash
make check     # ruff and pytest
```

## Citing

If you use this code, please cite the paper:

> Osinga, S., Kraaijpoel, D., Aben, F., & Pluymaekers, M. (in review). *b-free:
> Inferring Gutenberg–Richter earthquake size distributions without estimating
> b-values.* Bulletin of the Seismological Society of America.
> <!-- TODO: volume, pages and DOI on acceptance -->

and the archived data:

> Osinga, S., Kraaijpoel, D., Aben, F., & Pluymaekers, M. (2026). *Data for
> "b-free: Inferring Gutenberg–Richter earthquake size distributions without
> estimating b-values"* \[Data set]. Zenodo.
> <https://doi.org/10.5281/zenodo.21773391>

## License

MIT. See [LICENSE](LICENSE).

The ESHM20 input data retrieved by `analyse_eshm20.py` are released separately
under CC BY 4.0 and carry their own citation requirement, which that script
prints.
