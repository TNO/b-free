#!/usr/bin/env python
"""
Compare ground-motion amplitudes at fixed return periods.

For each ESHM20 zone and intensity measure, this script inverts the hazard
curve for the Aki-Utsu, frequentist b-free and Bayesian b-free magnitude
distributions. It reports each b-free amplitude relative to the Aki-Utsu
amplitude at the same annual exceedance rate (1 / return period).

Run from the repository root. ESHM20 input files are downloaded when absent;
by default the zones are selected from results/eshm20_results.zarr.zip.

    python scripts/analyse_hazard_fixed_rp.py
    python scripts/analyse_hazard_fixed_rp.py --zones ESAS136 SIAS423
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from analyse_eshm20 import (
    CONVENTIONS,
    classify_zones,
    last_period_catalogue,
    load_data,
)
from analyse_eshm20 import (
    analyze_zone as analyse_eshm20_zone,
)
from analyse_hazard import (
    MIN_MAG,
    N_SITE_SAMPLES,
    RETURN_PERIODS,
    _default_zones,
    _require_openquake,
    distance_distribution,
    gmm_table,
    hazard_curves,
    n_zones_arg,
    size_distributions,
)
from bfree.store import save as save_results


def amplitude_at_return_period(ln_levels, rates, return_period):
    """Interpolate ln(amplitude) at annual exceedance rate 1 / return_period."""
    target = 1.0 / return_period
    rates = np.asarray(rates, dtype=float)
    ln_levels = np.asarray(ln_levels, dtype=float)
    valid = np.isfinite(rates) & (rates > 0) & np.isfinite(ln_levels)
    rates, ln_levels = rates[valid], ln_levels[valid]
    if not len(rates) or rates.max() < target or rates.min() > target:
        return None

    log_rates = np.log(rates[::-1])
    levels = ln_levels[::-1]
    strictly_increasing = np.r_[True, np.diff(log_rates) > 0]
    log_rates, levels = log_rates[strictly_increasing], levels[strictly_increasing]
    return float(np.exp(np.interp(np.log(target), log_rates, levels)))


def analyse_zone_fixed_rp(zone_row, dist, gsim, imt, from_string,
                          site_placement="centroid",
                          return_periods=RETURN_PERIODS, min_mag=MIN_MAG):
    """Return ground-motion amplitudes and ratios at fixed return periods."""
    keep = dist["mags"] >= min_mag
    if keep.sum() < 3:
        raise ValueError(f"fewer than 3 magnitude bins above Mmin={min_mag}")
    mags = dist["mags"][keep]
    pmfs = {key: value[keep] for key, value in dist["pmfs"].items()}
    nu = 10.0 ** (
        zone_row["A_VAL"]
        - zone_row["B_VAL"] * (dist["mc"] - dist["delta"])
    )

    rjb, weights_r, resolved_site = distance_distribution(
        zone_row["geometry"], site_placement, n_samples=N_SITE_SAMPLES,
        return_site_placement=True,
    )
    mu, sigma = gmm_table(
        gsim, mags, rjb, from_string(imt), float(zone_row["HYPODEPTH1"])
    )
    ln_levels, rates, _ = hazard_curves(pmfs, mu, sigma, weights_r, nu)

    by_rp = []
    for return_period in return_periods:
        amplitudes = {
            name: amplitude_at_return_period(ln_levels, rates[name], return_period)
            for name in ("aki", "bfree", "bayes")
        }
        aki = amplitudes["aki"]
        ratios = {
            name: (amplitudes[name] / aki
                   if amplitudes[name] is not None and aki is not None else None)
            for name in ("bfree", "bayes")
        }
        by_rp.append({
            "return_period": int(return_period),
            "amplitudes": amplitudes,
            "ratios": ratios,
        })

    return {
        "zone_id": zone_row["ZONE_ID"],
        "imt": str(imt),
        "n": dist["n"],
        "site_placement": resolved_site,
        "return_periods": by_rp,
    }


def summarise(records):
    """Median amplitude ratios over zones, for each IM and return period."""
    rows = []
    for imt in dict.fromkeys(record["imt"] for record in records):
        imt_records = [record for record in records if record["imt"] == imt]
        return_periods = imt_records[0]["return_periods"]
        for index, rp_entry in enumerate(return_periods):
            row = {"imt": imt, "return_period": rp_entry["return_period"]}
            for name in ("bfree", "bayes"):
                ratios = [
                    record["return_periods"][index]["ratios"][name]
                    for record in imt_records
                    if record["return_periods"][index]["ratios"][name] is not None
                ]
                row[f"{name}_n_zones"] = len(ratios)
                row[f"{name}_median_ratio"] = (
                    float(np.median(ratios)) if ratios else float("nan")
                )
                row[f"{name}_median_percent_change"] = (
                    100 * (row[f"{name}_median_ratio"] - 1)
                    if ratios else float("nan")
                )
            rows.append(row)
    return rows


def run(output_path, convention, zone_ids=None, imt_names=("PGA", "SA(0.2)", "SA(1.0)"),
        site_placement="centroid", n_zones=8, min_mag=MIN_MAG,
        eshm20_results="results/eshm20_results.zarr.zip"):
    """Compute fixed-return-period amplitudes and save the zone-level results."""
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
    for zone_id in zone_ids:
        try:
            catalogue = analyse_eshm20_zone(gdf_d, gdf_with_csz, mc_shallow, gdf, zone_id)
            mags, mc_z = last_period_catalogue(catalogue, convention)
            zone_row = gdf_d[gdf_d["ZONE_ID"] == zone_id].iloc[0]
            dist = size_distributions(zone_row, mags, mc_z)
        except Exception as exc:
            print(f"  {zone_id}: skipped ({exc})")
            continue

        for imt in imt_names:
            try:
                records.append(analyse_zone_fixed_rp(
                    zone_row, dist, gsim, imt, from_string, site_placement,
                    min_mag=min_mag,
                ))
            except Exception as exc:
                print(f"  {zone_id} {imt}: skipped ({exc})")

    if not records:
        raise SystemExit("No zones could be analysed.")

    summary = summarise(records)
    print("\nMedian ground-motion amplitude ratios relative to Aki-Utsu:")
    for row in summary:
        print(
            f"{row['imt']:<10} {row['return_period']:>4} yr  "
            f"frequentist={row['bfree_median_ratio']:.4f} "
            f"({row['bfree_median_percent_change']:+.2f}%, "
            f"n={row['bfree_n_zones']})  "
            f"Bayesian={row['bayes_median_ratio']:.4f} "
            f"({row['bayes_median_percent_change']:+.2f}%, "
            f"n={row['bayes_n_zones']})"
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, {
        "records": records,
        "summary": summary,
        "metadata": {
            "gsim": "KothaEtAl2020ESHM20",
            "vs30": 800.0,
            "site_placement": site_placement,
            "min_mag": min_mag,
            "n_site_samples": N_SITE_SAMPLES,
            "return_periods": list(RETURN_PERIODS),
            "reference": "Aki-Utsu amplitude at the same annual exceedance rate",
            "mc_convention": convention,
        },
    })
    print(f"\nSaved: {output_path}")
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ground-motion amplitudes at fixed return periods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", default="results/hazard_fixed_rp_results.zarr.zip",
        metavar="FILE",
    )
    parser.add_argument("--mc-convention", choices=CONVENTIONS, required=True,
                        help="How a published Mc relates to the magnitude bins; "
                             "see analyse_eshm20.py --help.")
    parser.add_argument("--zones", nargs="+", metavar="ZONE_ID")
    parser.add_argument("--n-zones", type=n_zones_arg, default=8)
    parser.add_argument(
        "--imts", nargs="+", default=["PGA", "SA(0.2)", "SA(1.0)"],
        metavar="IMT",
    )
    parser.add_argument(
        "--site", choices=("centroid", "edge", "centroid-edge"),
        default="centroid",
        help="Site placement: centroid-edge uses the centroid when inside and "
             "the boundary only when the centroid is outside.",
    )
    parser.add_argument("--min-mag", type=float, default=MIN_MAG)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run(
        Path(args.output),
        args.mc_convention,
        zone_ids=args.zones,
        imt_names=args.imts,
        site_placement=args.site,
        n_zones=args.n_zones,
        min_mag=args.min_mag,
    ) and 0)
