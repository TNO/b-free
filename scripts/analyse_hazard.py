#!/usr/bin/env python
"""
analyse_hazard.py

Propagates the three earthquake size distributions (Aki-Utsu plug-in,
frequentist b-free, Bayesian b-free) through the seismic hazard integral for
ESHM20 area source zones, and reports the resulting ratios in ground-motion
exceedance rate.

The manuscript compares the estimators pointwise, at the 250-year magnitude
m_250.  Hazard, however, is an integral over all magnitudes weighted by the
ground motion model.  This script reports both, side by side:

    magnitude domain   S_method(m_250) / S_AkiUtsu(m_250)
    hazard domain      lambda_method   / lambda_AkiUtsu

These are two summaries of the same underlying difference, not cause and
effect, and neither is derived from the other: the hazard ratio is a weighted
average over the whole size distribution, of which m_250 is one point.  Their
quotient is deliberately not reported.  It would look like a transfer
efficiency, but its denominator depends entirely on the choice of m_250, which
is free: holding the physics fixed and moving that choice from a 50- to a
2475-year magnitude moves the quotient from 0.47 to 0.22 for PGA and from 0.99
to 0.44 for SA(1.0), while the hazard ratios it claims to explain do not move
at all.

The hazard ratio depends strongly on the intensity measure, because the ESHM20
backbone ground motion model saturates with magnitude at short periods: PGA
barely distinguishes an M6.5 from an M7.5, so upper-tail differences between
the estimators - which is where they disagree - cannot reach PGA hazard.
Long-period spectral acceleration does not saturate and retains them.

Ground motion model
-------------------
KothaEtAl2020ESHM20, the shallow-crustal backbone used by ESHM20 via the
Weatherill et al. (2020) scaled-backbone logic tree.  Only the backbone median
and total sigma are used; logic-tree branches are not propagated.

    Kotha, S. R., Weatherill, G., Bindi, D., & Cotton, F. (2020). A
    regionally-adaptable ground-motion model for shallow crustal earthquakes in
    Europe. Bulletin of Earthquake Engineering, 18, 4091-4125.

    Weatherill, G., Kotha, S. R., & Cotton, F. (2020). A regionally-adaptable
    "scaled backbone" ground motion logic tree for shallow seismicity in Europe.
    Bulletin of Earthquake Engineering, 18, 5087-5117.

Requires OpenQuake, which is not part of the base environment because it is
only needed here.  Either route works:

    uv venv .venv-hazard --python 3.11
    uv pip install --python .venv-hazard/bin/python \
        --overrides overrides_hazard.txt -r requirements_hazard.txt

    conda env create -f env_hazard.yml

The --overrides flag drops openquake's `gdal` requirement, which is source-only
on PyPI and cannot build without a system libgdal.  Nothing here uses it; see
overrides_hazard.txt.

Known limitation
----------------
Ruptures are approximated by their epicentre, so R_JB can approach zero for a
site inside the zone.  Real ruptures have finite extent, and OpenQuake area
sources build rupture planes.  Absolute ground motion levels are therefore
overestimated.  Exceedance-rate *ratios* between the three size distributions
are far less sensitive to this than the absolute levels, but the controlling
magnitude does shift, so the ratios should be treated as indicative until finite
ruptures are added.

How large the overestimate is has not been measured.  An eyeball comparison
against the published ESHM20 maps suggested something like a factor of two in
the most active zones, but that was an impression and not a calculation, so no
number is quoted here or in the manuscript.  Measuring it means computing PGA at
the zone centroids and reading the published maps at the same points.

Usage
-----
    python analyse_hazard.py
    python analyse_hazard.py --zones ESAS136 SIAS423 --imts PGA SA(1.0)
"""

import argparse
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import norm

from analyse_eshm20 import (
    CONVENTIONS,
    analyze_zone,
    classify_zones,
    infer_delta,
    last_period_catalogue,
    load_data,
)
from bfree import (
    LOG10,
    bayesian_survival_rounded,
    bfree_survival_rounded,
    gr_survival,
    magnitude_statistics_rounded,
)
from bfree.store import load as load_results
from bfree.store import save as save_results

# Equal-area projection for Europe; used for both area sampling and distances.
EQUAL_AREA_CRS = "EPSG:3035"

VS30 = 800.0                     # ESHM20 reference rock
RETURN_PERIODS = (475, 2475)
RP_CURVE = np.geomspace(50.0, 5000.0, 60)   # ground motion return periods for
                                            # the ratio-versus-return-period
                                            # curve; unlike the pointwise
                                            # comparison this needs no m_250.
N_SITE_SAMPLES = 6000            # Monte Carlo points for the R_JB distribution.
                                 # At 3000 the spread across seeds is ~13% of
                                 # the smallest rate ratios; 6000 halves it.
N_LEVELS = 400                   # ground motion levels per hazard curve
MIN_MAG = 4.5                    # smallest magnitude carried into the hazard
                                 # integral; below this earthquakes are not
                                 # damaging and standard PSHA excludes them.
                                 # Retained bins are NOT renormalised, so all
                                 # three estimators stay anchored to the same
                                 # observed rate above Mc.


def _require_openquake():
    """Import OpenQuake, with a message naming the separate environment if absent."""
    try:
        from openquake.hazardlib.gsim.kotha_2020 import KothaEtAl2020ESHM20
        from openquake.hazardlib.imt import from_string
    except ImportError as exc:
        raise SystemExit(
            "OpenQuake is required for the hazard analysis but is not installed.\n"
            "Install it with either:\n"
            "    uv pip install --overrides overrides_hazard.txt "
            "-r requirements_hazard.txt\n"
            "    conda env create -f env_hazard.yml\n"
            f"Original error: {exc}"
        ) from exc
    return KothaEtAl2020ESHM20, from_string


# ---------------------------------------------------------------------------
# Source geometry
# ---------------------------------------------------------------------------

def _site_for_zone(zone, site_placement):
    """Choose the site point and report whether centroid or edge was used."""
    if site_placement in ("centroid", "centroid-edge"):
        site = zone.centroid
        if zone.contains(site):
            return site, "centroid"
        if site_placement == "centroid":
            raise ValueError(
                "zone is concave and its centroid lies outside the polygon; "
                "use --site centroid-edge or --site edge"
            )
        site_placement = "edge"

    if site_placement == "edge":
        return zone.exterior.interpolate(0.0, normalized=True), "edge"
    raise ValueError(f"unknown site_placement: {site_placement!r}")


def distance_distribution(zone_geom, site_placement="centroid",
                          n_samples=N_SITE_SAMPLES, seed=0,
                          return_site_placement=False):
    """
    Joyner-Boore distances from a site to points drawn uniformly over the zone.

    Ruptures are approximated by their epicentre, which understates R_JB for
    large magnitudes; see the module docstring.

    Distances are computed in ETRS89-LAEA (EPSG:3035), which preserves area for
    the uniform sampling and keeps distance error below 1% across the ESHM20
    footprint.

    Returns (rjb_km, weights) with equal weights.
    """
    zone = gpd.GeoSeries([zone_geom], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).iloc[0]

    site, resolved_placement = _site_for_zone(zone, site_placement)

    minx, miny, maxx, maxy = zone.bounds
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    while len(xs) < n_samples:
        px = rng.uniform(minx, maxx, n_samples)
        py = rng.uniform(miny, maxy, n_samples)
        inside = gpd.points_from_xy(px, py).within(zone)
        xs.extend(px[inside].tolist())
        ys.extend(py[inside].tolist())
    xs = np.asarray(xs[:n_samples])
    ys = np.asarray(ys[:n_samples])

    rjb = np.hypot(xs - site.x, ys - site.y) / 1000.0
    weights = np.full(n_samples, 1.0 / n_samples)
    if return_site_placement:
        return rjb, weights, resolved_placement
    return rjb, weights


# ---------------------------------------------------------------------------
# Ground motion
# ---------------------------------------------------------------------------

def gmm_table(gsim, mags, rjb, imt, hypo_depth):
    """Median ln(IM) and total sigma on the mags x rjb grid."""
    mg, rg = np.meshgrid(np.asarray(mags, float), np.asarray(rjb, float),
                         indexing="ij")
    dtype = np.dtype([("mag", "f8"), ("rjb", "f8"), ("vs30", "f8"),
                      ("vs30measured", "bool"), ("region", "i4"),
                      ("hypo_depth", "f8"), ("sids", "i4")])
    ctx = np.zeros(mg.size, dtype).view(np.recarray)
    ctx.mag, ctx.rjb = mg.ravel(), rg.ravel()
    ctx.vs30, ctx.vs30measured, ctx.region = VS30, False, 0
    ctx.hypo_depth, ctx.sids = hypo_depth, np.arange(mg.size)

    out = [np.zeros((1, mg.size)) for _ in range(4)]
    gsim.compute(ctx, [imt], *out)
    return out[0][0].reshape(mg.shape), out[1][0].reshape(mg.shape)


def hazard_curves(pmfs, mu, sigma, weights_r, nu, n_levels=N_LEVELS):
    """
    Exceedance rate curves for each size distribution on a shared level grid.

    Returns (ln_levels, {name: rates}, exceedance_probabilities).
    """
    lo = mu.min() - 4.0 * sigma.max()
    hi = mu.max() + 4.0 * sigma.max()
    ln_levels = np.linspace(lo, hi, n_levels)

    # P(IM > level | m, r) for every level, magnitude and distance
    pex = norm.sf((ln_levels[:, None, None] - mu[None]) / sigma[None])
    rates = {name: nu * np.einsum("xmr,m,r->x", pex, pmf, weights_r)
             for name, pmf in pmfs.items()}
    return ln_levels, rates, pex


def _controlling_magnitude(pex_level, pmf, weights_r, mags):
    """Magnitude that contributes most to exceedance, by deaggregation."""
    w = pmf * (pex_level @ weights_r)
    return float(np.sum(w * mags) / w.sum())


def magnitude_return_period(mag, a_val, b_val):
    """Return period of `mag` on the ESHM20 central a-value and b-value branch.

    The inverse of m_RP = (a_val + log10(RP)) / b_val, which is how the
    magnitude domain fixes m_250 and every other level it is read at.  Both
    directions therefore have to use the published branch: decaying back with a
    refitted b-value instead would break the round trip, and feeding m_250 in
    would not return 250 years.
    """
    return float(10.0 ** (b_val * mag - a_val))


# ---------------------------------------------------------------------------
# Per-zone analysis
# ---------------------------------------------------------------------------

def size_distributions(zone_row, catalog_mags, mc_z):
    """
    The three size distributions for one zone, discretised out to Mmax.

    zone_curves() in analyse_eshm20 stops the grid shortly above m_250 because
    it only needs to plot that far.  Hazard needs the full distribution up to
    Mmax, otherwise the upper tail - where the estimators actually differ - is
    truncated away.
    """
    n_obs = len(catalog_mags)
    delta = infer_delta(catalog_mags)
    step = 2 * delta
    mmax = float(min(zone_row["MAXMAG01"], zone_row["MAXMAG02"],
                     zone_row["MAXMAG03"]))

    _, t_mag, w_mag = magnitude_statistics_rounded(catalog_mags, mc_z, delta)
    b_hat = n_obs / (t_mag * LOG10)
    m_250 = (zone_row["A_VAL"] - np.log10(1 / 250.0)) / zone_row["B_VAL"]

    # Bin edges lie at mc_z - delta + step*j, and the mass of bin k is the drop
    # in exceedance across it, so the distribution is truncated at the last edge
    # at or below Mmax rather than wherever arange happens to stop.
    n_bins = int(np.floor((mmax - (mc_z - delta)) / step))
    if n_bins < 3:
        raise ValueError(f"Mmax={mmax:.2f} too close to Mc={mc_z:.2f}")
    centres = mc_z + step * np.arange(n_bins + 1)     # reported magnitude values
    k_range = np.arange(len(centres))

    surv = {"aki": gr_survival(centres, b_hat, mc_z),
            "bfree": bfree_survival_rounded(w_mag, n_obs, k_range)[0],
            "bayes": bayesian_survival_rounded(w_mag, n_obs, k_range)[0]}

    # surv[-1] is the exceedance of the truncation edge, so it is the mass the
    # renormalisation below discards.  Recorded because that step is applied to
    # the posterior-averaged masses rather than inside the average.
    tail_mass = {name: float(np.asarray(s, float)[-1]) for name, s in surv.items()}

    pmfs = {}
    for name, s in surv.items():
        pmf = -np.diff(np.asarray(s, float))
        if pmf.sum() <= 0:
            raise ValueError(f"degenerate size distribution for {name}")
        pmfs[name] = pmf / pmf.sum()

    # surv[k] is the exceedance of bin k's lower edge, so pmf[k] is the mass in
    # [centres[k] - delta, centres[k] + delta) and the representative magnitude
    # of that bin is centres[k] itself.
    u_250 = (m_250 - mc_z + delta) / (2 * delta)
    s_aki = 10.0 ** (-b_hat * (m_250 - mc_z + delta))
    pointwise = {
        "bfree": float(bfree_survival_rounded(w_mag, n_obs, u_250)[0][0]) / s_aki,
        "bayes": float(bayesian_survival_rounded(w_mag, n_obs, u_250)[0][0]) / s_aki,
    }

    # The same ratio on the distributions the hazard integral actually consumes,
    # which are truncated at the edge above and renormalised.  Reporting only the
    # unbounded one would charge the truncation step to the ground motion model.
    k_top = float(k_range[-1])
    s_top = {"aki": 10.0 ** (-b_hat * 2 * delta * k_top),
             "bfree": float(bfree_survival_rounded(w_mag, n_obs, k_top)[0][0]),
             "bayes": float(bayesian_survival_rounded(w_mag, n_obs, k_top)[0][0])}
    s_250 = {"aki": s_aki,
             "bfree": float(bfree_survival_rounded(w_mag, n_obs, u_250)[0][0]),
             "bayes": float(bayesian_survival_rounded(w_mag, n_obs, u_250)[0][0])}
    if m_250 < centres[-1] - delta:
        trunc = {k: (s_250[k] - s_top[k]) / (1.0 - s_top[k]) for k in s_250}
        pointwise_bounded = {k: trunc[k] / trunc["aki"] for k in ("bfree", "bayes")}
    else:
        pointwise_bounded = {"bfree": float("nan"), "bayes": float("nan")}

    return {"centres": centres, "mags": centres[:-1], "pmfs": pmfs,
            "surv": surv, "n": n_obs, "mc": mc_z, "delta": delta,
            "m_250": m_250, "mmax": mmax, "b_aki": b_hat,
            "m_trunc": float(centres[-1] - delta), "tail_mass": tail_mass,
            "pointwise": pointwise, "pointwise_bounded": pointwise_bounded}


def analyse_zone(zone_row, dist, gsim, imt, site_placement="centroid",
                 return_periods=RETURN_PERIODS, min_mag=MIN_MAG):
    """Hazard ratios and deaggregation for a single zone and intensity measure."""
    keep = dist["mags"] >= min_mag
    if keep.sum() < 3:
        raise ValueError(f"fewer than 3 magnitude bins above Mmin={min_mag}")
    mags = dist["mags"][keep]
    pmfs = {k: v[keep] for k, v in dist["pmfs"].items()}

    # The size distributions are conditional on M >= mc - delta, the bottom of
    # the catalogue, so the rate multiplying them has to count the same events.
    nu = 10.0 ** (zone_row["A_VAL"]
                  - zone_row["B_VAL"] * (dist["mc"] - dist["delta"]))
    rjb, weights_r, resolved_site = distance_distribution(
        zone_row["geometry"], site_placement, return_site_placement=True
    )
    hypo = float(zone_row["HYPODEPTH1"])

    mu, sigma = gmm_table(gsim, mags, rjb, imt, hypo)
    ln_levels, rates, pex = hazard_curves(pmfs, mu, sigma, weights_r, nu)

    result = {"zone_id": zone_row["ZONE_ID"], "imt": str(imt), "n": dist["n"],
              "mc": dist["mc"], "m_250": dist["m_250"], "mmax": dist["mmax"],
              "m_trunc": dist["m_trunc"], "tail_mass": dist["tail_mass"],
              "nu": nu, "pointwise": dist["pointwise"],
              "pointwise_bounded": dist["pointwise_bounded"], "min_mag": min_mag,
              "area_km2": float(zone_row["AREAKM2"]),
              "site_placement": resolved_site}

    for rp in return_periods:
        target = 1.0 / rp
        if rates["aki"].min() > target or rates["aki"].max() < target:
            result[rp] = None                       # return period out of range
            continue
        j = int(np.argmin(np.abs(rates["aki"] - target)))
        ctrl = _controlling_magnitude(pex[j], pmfs["aki"], weights_r, mags)
        # Magnitude return period matching the controlling magnitude. Answers
        # "which m_RP would make the magnitude-domain comparison representative
        # of this hazard level?" - see the module docstring.
        entry = {"gm": float(np.exp(ln_levels[j])),
                 "ctrl_mag": ctrl,
                 "ctrl_mag_return_period": magnitude_return_period(
                     ctrl, zone_row["A_VAL"], zone_row["B_VAL"])}
        for name in ("bfree", "bayes"):
            entry[name] = float(rates[name][j] / rates["aki"][j])
        result[rp] = entry

    # Ratio across return periods. Unlike the pointwise comparison this does not
    # depend on m_250: the choice of magnitude enters nowhere in the integral.
    curve = {"return_period": RP_CURVE}
    in_range = ((rates["aki"].min() <= 1.0 / RP_CURVE)
                & (rates["aki"].max() >= 1.0 / RP_CURVE))
    idx = np.abs(rates["aki"][None, :] - (1.0 / RP_CURVE)[:, None]).argmin(axis=1)
    for name in ("bfree", "bayes"):
        curve[name] = np.where(in_range, rates[name][idx] / rates["aki"][idx],
                               np.nan)
    result["curve"] = curve
    return result

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _default_zones(eshm20_results, convention, n_zones):
    """
    The manuscript's qualifying zones, smallest catalogues first: a count, or
    "all".

    Reusing results/eshm20_results.zarr.zip keeps the zone set identical to the pointwise
    analysis, so the two domains compare like with like.  The store must have
    been made under the same completeness reading, or the zone set is another
    analysis's.
    """
    path = Path(eshm20_results)
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run analyse_eshm20.py first, or pass --zones "
            "explicitly."
        )
    store = load_results(path)
    stored = store["metadata"]["mc_convention"]
    if stored != convention:
        raise SystemExit(f"{path} was made with --mc-convention {stored}, "
                         f"not {convention}")
    zones = store["zones"]
    if n_zones == "all":
        return zones.sort_values("n")["zone_id"].tolist()
    return zones.nsmallest(int(n_zones), "n")["zone_id"].tolist()


def n_zones_arg(value):
    """--n-zones: a count, or 'all' for every qualifying zone."""
    if value == "all":
        return value
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("--n-zones must be positive or 'all'")
    return count


def run(output_path, convention, zone_ids=None, imt_names=("PGA", "SA(0.2)", "SA(1.0)"),
        site_placement="centroid", n_zones=8, min_mag=MIN_MAG,
        eshm20_results="results/eshm20_results.zarr.zip"):
    """Propagate each zone's estimated size distributions through the hazard
    integral and write the exceedance-rate ratios.

    Report at least PGA and SA(1.0): they respond in opposite directions,
    because magnitude saturation differs with period.
    """
    warnings.filterwarnings("ignore")
    KothaEtAl2020ESHM20, from_string = _require_openquake()
    gsim = KothaEtAl2020ESHM20()

    if zone_ids is None:
        zone_ids = _default_zones(eshm20_results, convention, n_zones)
        print(f"Using the {len(zone_ids)} smallest qualifying zones: "
              f"{', '.join(zone_ids)}")

    print("Loading ESHM20 data ...")
    gdf, gdf_d, gdf_with_csz, mc_shallow, _ = load_data()
    gdf_d = classify_zones(gdf, gdf_d, mc_shallow, convention)

    records = []
    for zid in zone_ids:
        try:
            r = analyze_zone(gdf_d, gdf_with_csz, mc_shallow, gdf, zid)
            mags, mc_z = last_period_catalogue(r, convention)
            zone_row = gdf_d[gdf_d["ZONE_ID"] == zid].iloc[0]
            dist = size_distributions(zone_row, mags, mc_z)
        except Exception as exc:                      # zone-level data problem
            print(f"  {zid}: skipped ({exc})")
            continue

        for name in imt_names:
            try:
                records.append(analyse_zone(zone_row, dist, gsim,
                                            from_string(name), site_placement,
                                            min_mag=min_mag))
            except Exception as exc:
                print(f"  {zid} {name}: skipped ({exc})")

    if not records:
        raise SystemExit("No zones could be analysed.")

    print(f"\n{'zone':<10}{'n':>4}{'IM':>10}", end="")
    for rp in RETURN_PERIODS:
        print(f"{f'  ------- {rp} yr -------':>39}", end="")
    print()
    print(f"{'':<10}{'':>4}{'':>10}", end="")
    for _ in RETURN_PERIODS:
        print(f"{'ctrlM':>8}{'RPmag':>8}{'free/AU':>10}{'bayes/AU':>10}{'':>3}",
              end="")
    print()
    for rec in records:
        print(f"{rec['zone_id']:<10}{rec['n']:>4}{rec['imt']:>10}", end="")
        for rp in RETURN_PERIODS:
            e = rec[rp]
            if e is None:
                print(f"{'-':>8}{'-':>8}{'-':>10}{'-':>10}{'':>3}", end="")
            else:
                print(f"{e['ctrl_mag']:>8.2f}{e['ctrl_mag_return_period']:>8.0f}"
                      f"{e['bfree']:>10.3f}{e['bayes']:>10.3f}{'':>3}", end="")
        print()

    summary = _summarise(records)
    print("\nMedian ratios relative to the Aki-Utsu plug-in "
          "(no ground truth on real catalogues):")
    print(pd.DataFrame(summary).to_string(index=False))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, {
        "records": records, "summary": summary,
        # Everything the supplement states about this calculation is recorded
        # here, so the two cannot drift apart.
        "metadata": {"gsim": "KothaEtAl2020ESHM20", "vs30": VS30,
                     "site_placement": site_placement,
                     "return_periods": list(RETURN_PERIODS),
                     "min_mag": min_mag,
                     "n_site_samples": N_SITE_SAMPLES,
                     "n_levels": N_LEVELS,
                     "level_grid_sigmas": 4.0,
                     "gmm_sigma_truncation": "none",
                     "equal_area_crs": EQUAL_AREA_CRS,
                     "vs30measured": False,
                     "hypo_depth_source": "ESHM20 zone attribute HYPODEPTH1",
                     "mc_convention": convention}})
    print(f"\nSaved: {output_path}")
    return records


def _summarise(records):
    """
    Median ratios relative to Aki-Utsu, in both domains, per IM and period.

    Returns plain dicts rather than a DataFrame: the result is stored and read
    back by the plotting environment, which may have a different pandas version.
    """
    rows = []
    for imt in dict.fromkeys(r["imt"] for r in records):
        for rp in RETURN_PERIODS:
            sel = [r for r in records if r["imt"] == imt and r[rp] is not None]
            if not sel:
                continue
            rows.append({
                "imt": imt,
                "return_period": rp,
                "n_zones": len(sel),
                # magnitude domain: what the manuscript reports
                "P_m250_bfree": float(np.median([r["pointwise"]["bfree"] for r in sel])),
                "P_m250_bayes": float(np.median([r["pointwise"]["bayes"] for r in sel])),
                # ground motion domain: what hazard actually sees
                "hazard_bfree": float(np.median([r[rp]["bfree"] for r in sel])),
                "hazard_bayes": float(np.median([r[rp]["bayes"] for r in sel])),
                # magnitude return period this hazard level actually probes
                "ctrl_mag_RP": float(np.median(
                    [r[rp]["ctrl_mag_return_period"] for r in sel])),
            })
    return rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Propagate the b-free size distributions through the "
                    "seismic hazard integral for ESHM20 zones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="results/hazard_results.zarr.zip", metavar="FILE")
    parser.add_argument("--mc-convention", choices=CONVENTIONS, required=True,
                        help="How a published Mc relates to the magnitude bins; "
                             "see analyse_eshm20.py --help.")
    parser.add_argument("--zones", nargs="+", metavar="ZONE_ID",
                        help="Zone IDs to analyse. Default: the smallest "
                             "data-based zones.")
    parser.add_argument("--n-zones", type=n_zones_arg, default=8,
                        help="Number of zones to use when --zones is not given, "
                             "smallest catalogues first, or 'all'.")
    parser.add_argument("--imts", nargs="+", default=["PGA", "SA(0.2)", "SA(1.0)"],
                        metavar="IMT",
                        help="Intensity measures. Report at least PGA and "
                             "SA(1.0): they differ in direction.")
    parser.add_argument(
        "--site", choices=("centroid", "edge", "centroid-edge"),
        default="centroid",
        help="Site placement: centroid-edge uses the centroid when inside and "
             "the boundary only when the centroid is outside.",
    )
    parser.add_argument("--min-mag", type=float, default=MIN_MAG,
                        help="Smallest magnitude carried into the hazard "
                             "integral.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(run(Path(args.output), args.mc_convention, zone_ids=args.zones,
                 imt_names=args.imts, site_placement=args.site, n_zones=args.n_zones,
                 min_mag=args.min_mag) and 0)
