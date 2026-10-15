# b-free investigations: data, figures and checks.
#
# Every target is a thin wrapper over a script in scripts/, which remains the
# authoritative interface. Run `make help` for the list, and override any
# variable on the command line, e.g. `make figures FIGURES=/tmp/fig`.

PYTHON       ?= uv run --no-sync python
RESULTS      ?= results
FIGURES      ?= figures
DIST         ?= dist/zenodo

# Publication to the public GitHub mirror. Synced from preseis-essentials; see
# docs/GIT_PUBLICATION_FACILITIES.md there. The remote URL lives in
# GIT_PUBLICATION_LOCAL.env, which is clone-local and never published.
# Optional on purpose: the fragment is excluded from the published tree, where
# a hard include would make every target fail.
-include make/internal-public-git.mk

# Figure output. FORMAT takes any of png, pdf, eps. Each figure series has its
# own target and its own directory, so STYLE is not a variable here.
FORMAT       ?= png pdf


# The unrounded catalogues are generated on the fly rather than read from the
# store, so the seed is the only record of what was analysed. This is the one
# the archived result was made with.
SEED         ?= 20260825

# A full simulation is 10 million catalogues per cell. Override for a smoke
# test: `make simulate NSIM=1e4`.
NSIM         ?= 1e7
# The paper's hazard covers every qualifying zone, the same set as the
# magnitude-domain comparison. A count takes the smallest catalogues first.
HAZARD_ZONES ?= all
# The hazard interpreter carries OpenQuake; see requirements_hazard.txt.
HAZARD_PYTHON ?= .venv-hazard/bin/python

.PHONY: help install test lint check data data-full simulate reduce eshm20 \
	hazard figures figures-bssa figures-original figure-check package clean

help:
	@echo "Setup"
	@echo "  make install         install the package and its dev dependencies"
	@echo ""
	@echo "Getting the data (pick one)"
	@echo "  make data            download the reduced results from Zenodo (small, fast)"
	@echo "  make data-full       also download the raw catalogues (~1.3 GB)"
	@echo "  make simulate        generate everything locally (hours, ~9 GB RAM)"
	@echo ""
	@echo "Individual analyses"
	@echo "  make reduce          re-run the reduction over an existing store"
	@echo "  make eshm20          download ESHM20 inputs and apply the estimators"
	@echo "  make hazard          propagate through the hazard integral (needs OpenQuake)"
	@echo ""
	@echo "Figures"
	@echo "  make figures         both colour series    -> \$$(FIGURES)/"
	@echo "    figures-bssa         page width, colour        bssa-colour/"
	@echo "    figures-original     pre-page width, for talks original/"
	@echo "  make figure-check    report figures against the BSSA guidelines"
	@echo ""
	@echo "Publishing"
	@echo "  make package         assemble a Zenodo deposit in \$$(DIST)"
	@echo ""
	@echo "Quality"
	@echo "  make test            run the test suite"
	@echo "  make lint            run ruff"
	@echo "  make check           lint and test"
	@echo ""
	@echo "Variables: PYTHON RESULTS FIGURES DIST FORMAT STYLE SEED NSIM HAZARD_ZONES HAZARD_PYTHON"

install:
	uv sync --group dev

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

data:
	@mkdir -p $(RESULTS)
	$(PYTHON) scripts/retrieve_from_zenodo.py

data-full:
	@mkdir -p $(RESULTS)
	$(PYTHON) scripts/retrieve_from_zenodo.py --with-zarr

# Simulates the catalogues, writes the store, and reduces it in one pass.
simulate:
	@mkdir -p $(RESULTS)
	$(PYTHON) scripts/run_simulations.py --nr_simulations $(NSIM) \
	    --output $(RESULTS)/simulation_results.zarr \
	    --analysis-output $(RESULTS)/analysis_results.zarr.zip
	$(MAKE) reduce-continuous

reduce:
	$(PYTHON) scripts/reduce_simulations.py \
	    --input $(RESULTS)/simulation_results.zarr \
	    --output $(RESULTS)/analysis_results.zarr.zip

reduce-continuous:
	$(PYTHON) scripts/reduce_simulations.py --continuous --seed $(SEED) \
	    --output $(RESULTS)/analysis_results_continuous.zarr.zip

eshm20:
	@mkdir -p $(RESULTS)
	$(PYTHON) scripts/analyse_eshm20.py --mc-convention edge --output $(RESULTS)/eshm20_results.zarr.zip

# Needs the separate OpenQuake interpreter; see requirements_hazard.txt.
# centroid-edge: TRAS430 is concave and its centroid lies outside it.
hazard:
	@mkdir -p $(RESULTS)
	$(HAZARD_PYTHON) scripts/analyse_hazard.py --mc-convention edge \
	    --n-zones $(HAZARD_ZONES) --site centroid-edge \
	    --output $(RESULTS)/hazard_results.zarr.zip

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Two colour series, each in its own directory:
#
#   bssa-colour  the page series, 6.5 in wide, journal type sizes
#   original     the pre-page sizes, which is what reads on a projector
figures: figures-bssa figures-original

figures-bssa:
	@mkdir -p $(FIGURES)/bssa-colour
	$(PYTHON) scripts/make_figures.py --figure-dir $(FIGURES)/bssa-colour \
	    --results-dir $(RESULTS) --format $(FORMAT) --style bssa
	$(PYTHON) scripts/analyse_mc_sensitivity.py --figure-dir $(FIGURES)/bssa-colour \
	    --format $(FORMAT) --style bssa
	$(PYTHON) scripts/analyse_plugin_estimators.py --figure-dir $(FIGURES)/bssa-colour \
	    --zarr $(RESULTS)/simulation_results.zarr --format $(FORMAT) --style bssa

figures-original:
	@mkdir -p $(FIGURES)/original
	$(PYTHON) scripts/make_figures.py --figure-dir $(FIGURES)/original \
	    --results-dir $(RESULTS) --format $(FORMAT) --style original
	$(PYTHON) scripts/analyse_mc_sensitivity.py --figure-dir $(FIGURES)/original \
	    --format $(FORMAT) --style original
	$(PYTHON) scripts/analyse_plugin_estimators.py --figure-dir $(FIGURES)/original \
	    --zarr $(RESULTS)/simulation_results.zarr --format $(FORMAT) --style original

figure-check:
	@mkdir -p $(FIGURES)/bssa-colour
	$(PYTHON) scripts/make_figures.py --figure-dir $(FIGURES)/bssa-colour \
	    --results-dir $(RESULTS) --format png --style bssa --check

# ---------------------------------------------------------------------------
# Publishing and quality
# ---------------------------------------------------------------------------

package:
	$(PYTHON) scripts/package_for_zenodo.py --results-dir $(RESULTS) --out $(DIST)

test:
	uv run --no-sync pytest -q

lint:
	uv run --no-sync ruff check

check: lint test

clean:
	rm -rf $(DIST) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
