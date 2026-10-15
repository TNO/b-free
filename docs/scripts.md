# Command-line tools

Every script lives in `scripts/`, takes `--help`, and is run from the
repository root so that relative paths resolve against it. The `make` targets
in the project root are thin wrappers over these; the scripts remain the
authoritative interface.

## Producing data

| Script | Produces | Notes |
|---|---|---|
| `run_simulations.py` | `simulation_results.zarr`, `analysis_results.zarr.zip` | Simulates rounded Gutenberg–Richter catalogues into a Zarr store and reduces them in the same pass. 10 million catalogues per (b, n) cell by default: hours, ~1.4 GB of disk, ~9 GB of memory at peak. Use `--nr_simulations 1e4` for a smoke test. |
| `reduce_simulations.py` | `analysis_results.zarr.zip` | Reduces an existing store to the mean estimated size distribution and its ratio to the truth, per estimator. With `--continuous` it generates continuous catalogues on the fly instead, in which case `--seed` is the only record of what was analysed. |
| `analyse_eshm20.py` | `data/eshm20/`, `eshm20_results.zarr.zip` | Downloads the ESHM20 source model at a pinned revision, verifies each file against an embedded SHA-256 digest, and applies the estimators to the shallow area source zones. `--download-only` fetches without analysing. `--mc-convention {edge,centre}` is required for every analysis and has no default: `edge` reads a published completeness level as the lower edge of the first complete 0.2 bin, `centre` as that bin's centre. The manuscript uses `edge`, and the hazard scripts take the same option and refuse a stored zone set made under the other reading. |
| `analyse_hazard.py` | `hazard_results.zarr.zip` | Propagates the estimated size distributions through the seismic hazard integral with the ESHM20 backbone ground-motion model. Needs the separate OpenQuake interpreter. `--site centroid-edge` uses the centroid unless it lies outside a zone, then falls back to its boundary. |
| `analyse_hazard_fixed_rp.py` | `hazard_fixed_rp_results.zarr.zip` | Compares ground-motion amplitudes for the Aki-Utsu and b-free magnitude distributions at the same 475- and 2475-year return periods. Reports b-free amplitude ratios and percentage changes relative to Aki-Utsu. Needs the separate OpenQuake interpreter. `--site centroid-edge` uses the centroid unless it lies outside a zone, then falls back to its boundary. |

## Moving data

| Script | Purpose |
|---|---|
| `retrieve_from_zenodo.py` | Downloads the archived outputs so the figures can be made without repeating the simulations. Verifies every file against the MD5 the record reports. `--with-zarr` also fetches the raw catalogues (~1.3 GB); `--list` shows what the record holds. |
| `package_for_zenodo.py` | Assembles a deposit: the reduced results, optionally the zipped catalogue store, plus `CHECKSUMS.txt` and `MANIFEST.json` recording sizes, checksums and the commit that produced them. |

The two share one list of file names, so a deposit this produces is a deposit
that one can read.

## Figures

| Script | Purpose |
|---|---|
| `make_figures.py` | Produces the nine default paper figures described in [figures.md](figures.md). Cross-formulation comparisons, the catalog-size sweep, and the fixed-return-period amplitude figure are research-only; the latter needs a separate results store. `--only` selects figures, `--list` names figures and inputs, `--numbers` prints ESHM20 values, and `--check` reports against BSSA guidelines. |
| `analyse_mc_sensitivity.py` | Quantifies how each formulation responds to the placement of the reference magnitude relative to the completeness magnitude, and writes its own figure. |
| `analyse_plugin_estimators.py` | Compares the four plug-in b-value estimators — Aki-Utsu, Ogata, grouped, and the assembled one this manuscript puts together — over catalogues drawn directly as geometric bin indices. Prints bias and RMSE tables by default; `--figure-dir` draws the supplement figure instead. Takes `--style` like the other figure scripts. |

## Checks

| Script | Purpose |
|---|---|
| `analyse_continuation.py` | Measures the bias of the continued rounded estimators by deterministic summation over the law of the bin total, rather than by simulation, so the reported bias is exact rather than noisy. |
| `analyse_seismostats.py` | Regression check of `b_grouped` and `b_aki` against the values the independent SeismoStats package publishes. The same comparison runs in the test suite. |
