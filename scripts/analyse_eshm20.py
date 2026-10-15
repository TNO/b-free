#!/usr/bin/env python
"""
analyse_eshm20.py

Downloads the ESHM20 input shapefiles (once, into data/eshm20/), applies the
plug-in and b-free estimators to the ESHM20 area source zones, and writes the
per-zone results to results/eshm20_results.zarr.zip, ready for plotting.

The download is pinned to a specific commit because the upstream repository
replaces these files in place; `master` is not reproducible. Every file is
verified against a recorded SHA-256 digest.

The data are distributed under CC-BY 4.0 and must be cited:

    Danciu, L., Nandan, S., Reyes, C., Basili, R., Weatherill, G., Beauval, C.,
    Rovida, A., Vilanova, S., Sesetyan, K., Bard, P.-Y., Cotton, F., Wiemer, S.,
    & Giardini, D. (2021). Main Datasets of the 2020 Update of the European
    Seismic Hazard Model (ESHM20) [Data set]. EFEHR European Facilities of
    Earthquake Hazard and Risk. https://doi.org/10.12686/ESHM20-MAIN-DATASETS

By default only the zones for which ESHM20 derived a zone-specific b-value from
the catalogue are included, which is the set reported in the manuscript.

How a published completeness level Mc relates to the 0.2 magnitude bins is a
required choice, --mc-convention, with no default.  'edge' reads Mc as the
lower edge of the first complete bin, a raw threshold (magnitude >= Mc);
'centre' reads it as that bin's centre, which admits raw magnitudes from
Mc - 0.1.  The manuscript uses 'edge'; the evidence is in the working notes
(multiperiod_notes.md).

Usage
-----
    python analyse_eshm20.py --mc-convention edge
    python analyse_eshm20.py --mc-convention edge --all-zones
"""

import argparse
import hashlib
import io
import posixpath
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from bfree import (
    LOG10,
    bayesian_survival_rounded,
    bfree_survival_rounded,
    conjugate_prior_rounded,
    gr_survival,
    magnitude_statistics_rounded,
)
from bfree.store import save as save_results

DATA = Path("data/eshm20")

RETURN_PERIOD = 250
RP_CURVE = np.geomspace(25.0, 5000.0, 60)   # return periods for the sensitivity
N_CURVE_POINTS = 400          # threshold grid for the plotted survival curves
BIN_SIZE = 0.2
BIN_OFFSET = 0.1
MAX_DEPTH = 59
MIN_EVENTS = 30       # minimum events in the last completeness period
B_VALID_RANGE = (0.70, 1.15)   # ESHM20 validity range for zone-level b-values


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------

REPO = "https://gitlab.seismo.ethz.ch/efehr/eshm20"
REF = "524f8ed3588081ae8872321f655e8e60d43dd176"

CITATION = """\
ESHM20 input data are licensed CC-BY 4.0 and must be cited as:

  Danciu, L., Nandan, S., Reyes, C., Basili, R., Weatherill, G., Beauval, C.,
  Rovida, A., Vilanova, S., Sesetyan, K., Bard, P.-Y., Cotton, F., Wiemer, S.,
  & Giardini, D. (2021). Main Datasets of the 2020 Update of the European
  Seismic Hazard Model (ESHM20) [Data set]. EFEHR.
  https://doi.org/10.12686/ESHM20-MAIN-DATASETS
"""

# folder -> required file names (the extensions of a shapefile must stay together)
REQUIRED = {
    "eshm20_input_a_unified_eq_catalogue": [
        f"eshm20_unified_catalogue_declustered_v02a.{ext}"
        for ext in ("shp", "shx", "dbf", "prj", "cpg")
    ],
    "eshm20_input_b_completeness_superzones": [
        f"eshm20_csz_v06b.{ext}"
        for ext in ("shp", "shx", "dbf", "prj", "cpg", "qpj")
    ] + ["McTable_CSZ_ESHM20_ver_dec2020.xlsx"],
    "eshm20_input_d_asm_shallow_crust": [
        f"eshm20_asz_shallow_winGT_fs017_dm02_results_rev.{ext}"
        for ext in ("shp", "shx", "dbf", "prj", "qix")
    ],
}

# SHA-256 of each required file at commit REF; guards against silent upstream changes.
CHECKSUMS = {
    'eshm20_input_a_unified_eq_catalogue': {
        'eshm20_unified_catalogue_declustered_v02a.shp': '6e560b3a3ac6ad0e41a904e0acfcc8b4743e27d513bf278a3b3c69e57e53e033',
        'eshm20_unified_catalogue_declustered_v02a.shx': '1d2ac7784e7be8b27c7e54b589cec257b56bad31f9275dba6ab73ee991d9b60e',
        'eshm20_unified_catalogue_declustered_v02a.dbf': '45ee685d9d7906d4e2df4f5a7132ae7941db1597ee715a86c1d4e0c67676d831',
        'eshm20_unified_catalogue_declustered_v02a.prj': 'a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c4',
        'eshm20_unified_catalogue_declustered_v02a.cpg': '3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824',
    },
    'eshm20_input_b_completeness_superzones': {
        'eshm20_csz_v06b.shp': 'f923bf17411d5e00c2a302aa60c3b9ef1795ab57334c3cab3113779ab81e61e2',
        'eshm20_csz_v06b.shx': 'c4329348cbe48d10a64103c9ca4944ecda852cb2b62a20b60cb893d8439dfe50',
        'eshm20_csz_v06b.dbf': 'e28d3cc8bd15f75c7f67ce86dac7cec282100d25af7c2c19070a909ca74cc195',
        'eshm20_csz_v06b.prj': '98aaf3d1c0ecadf1a424a4536de261c3daf4e373697cb86c40c43b989daf52eb',
        'eshm20_csz_v06b.cpg': '3ad3031f5503a4404af825262ee8232cc04d4ea6683d42c5dd0a2f2a27ac9824',
        'eshm20_csz_v06b.qpj': '1de411dcdeedce3219242306fc29bfa1d7fa08883e4ff6779baf798ec50d1657',
        'McTable_CSZ_ESHM20_ver_dec2020.xlsx': 'bf06e93dec25d906e758dbee99fc3d1270b67bd7b77a9cec0789b359cbcab55d',
    },
    'eshm20_input_d_asm_shallow_crust': {
        'eshm20_asz_shallow_winGT_fs017_dm02_results_rev.shp': 'da07a5c1a026afcc28bcf953f8f8ec55fab99683e958f66276369762e9f50724',
        'eshm20_asz_shallow_winGT_fs017_dm02_results_rev.shx': 'ba8e85e1e0eea22a626ea0d0e1a987783d31b21286eceb67b9ddac334c166e28',
        'eshm20_asz_shallow_winGT_fs017_dm02_results_rev.dbf': 'ed262ae6413859b89e3351ae423b6e7e706f25b405d3157897b430fc08f40e5e',
        'eshm20_asz_shallow_winGT_fs017_dm02_results_rev.prj': '1931a86098419282a10f3db172e796defa72dcdcd23717dd8133ac3953f52b6f',
        'eshm20_asz_shallow_winGT_fs017_dm02_results_rev.qix': '85c84dfaa4cae61d8dc75ff6fbeed8b23ba9f52e47658f87d02117715aae95ef',
    },
}


def _archive_url(folder):
    """GitLab archive URL for one input folder at the pinned revision."""
    return (f"{REPO}/-/archive/{REF}/eshm20-{REF}.zip"
            f"?path=input_shapefiles/{folder}")


def _safe_member_name(name):
    """Reject archive entries that would escape the destination directory."""
    if posixpath.isabs(name) or name.startswith("/"):
        return False
    return ".." not in Path(name).parts


def _sha256(path):
    """SHA-256 of a file, used to verify the downloaded inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fetch_folder(folder, dest, force):
    """Download and verify one folder of ESHM20 input shapefiles."""
    out_dir = dest / folder
    wanted = REQUIRED[folder]

    if not force and all((out_dir / f).exists() for f in wanted):
        print(f"  {folder}: already present, skipping")
        return {f: _sha256(out_dir / f) for f in wanted}

    print(f"  {folder}: downloading ...")
    with urllib.request.urlopen(_archive_url(folder)) as response:  # noqa: S310 - https URL is a fixed constant
        payload = response.read()

    out_dir.mkdir(parents=True, exist_ok=True)
    digests = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        by_name = {}
        for member in archive.namelist():
            if not _safe_member_name(member):
                raise RuntimeError(f"unsafe archive entry: {member!r}")
            by_name[posixpath.basename(member)] = member

        for filename in wanted:
            if filename not in by_name:
                raise RuntimeError(
                    f"{filename!r} is missing from the archive of {folder!r}")
            data = archive.read(by_name[filename])
            (out_dir / filename).write_bytes(data)
            digests[filename] = hashlib.sha256(data).hexdigest()

    return digests


def download(dest=DATA, force=False, print_checksums=False):
    """Fetch the ESHM20 input shapefiles and verify them against CHECKSUMS."""
    print(f"Fetching ESHM20 input shapefiles at commit {REF[:8]} into {dest}/")
    digests = {folder: _fetch_folder(folder, dest, force) for folder in REQUIRED}

    if print_checksums:
        print("\nCHECKSUMS = {")
        for folder, files in digests.items():
            print(f"    {folder!r}: {{")
            for filename, digest in files.items():
                print(f"        {filename!r}: {digest!r},")
            print("    },")
        print("}")
        return

    mismatches = [
        f"{folder}/{filename}"
        for folder, expected in CHECKSUMS.items()
        for filename, digest in expected.items()
        if digests[folder].get(filename) != digest
    ]
    if mismatches:
        raise RuntimeError(
            "checksum mismatch for: " + ", ".join(mismatches)
            + f"\nThe upstream files at commit {REF} differ from the versions "
              "this analysis was built on.")

    print(f"\nAll files verified against the recorded checksums.\n\n{CITATION}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def infer_delta(mags):
    """
    Infer the half-bin width from the spacing of the reported magnitudes.

    Raises rather than guessing when no lattice can be established, which
    happens in two ways: fewer than two distinct magnitudes, or a common spacing
    no coarser than the resolution floor below.  The second is what an unrounded
    catalogue looks like, and inferring a bin width of 1e-4 from it would put the
    rounded estimators on a lattice the data does not have.  Such a catalogue
    should go through the continuous estimators in `bfree` instead.
    """
    resolution = 1e-4                       # finest spacing we will call a lattice
    scale = round(1.0 / resolution)
    unique = np.sort(np.unique(np.round(mags, 4)))
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        raise ValueError(
            "cannot infer a bin width: fewer than two distinct magnitudes"
        )
    scaled = np.round(diffs * scale).astype(int)
    g = int(scaled[0])
    for v in scaled[1:]:
        g = np.gcd(g, int(v))
    if g <= 1:
        raise ValueError(
            f"magnitudes share no common spacing coarser than {resolution}; "
            "the catalogue appears to be unrounded, so the rounded estimators "
            "do not apply to it"
        )
    return g / scale / 2


def analyze_zone(gdf_d, gdf_with_csz, mc_shallow, gdf, zone_id,
                 max_depth=MAX_DEPTH, declustering_flag="winGT_fs01"):
    """
    The declustered events of one zone, raw, with its published completeness history.

    Nothing here bins magnitudes or applies completeness: both depend on how a
    published Mc is read, so they happen once, in `complete_catalogue`.
    """
    zone_poly = gdf_d[gdf_d["ZONE_ID"] == zone_id][["ZONE_ID", "geometry"]]
    if zone_poly.empty:
        raise ValueError(f"Zone {zone_id!r} not found.")
    ev = gpd.sjoin(
        gdf[["eventID", "year", "magnitude", "depth",
             "winGT_fs01", "winGK_fs01", "geometry"]],
        zone_poly, how="inner", predicate="within"
    ).drop(columns="index_right")
    csz_lookup = gdf_with_csz.drop_duplicates("eventID").set_index("eventID")["CSZ_ID"]
    ev["CSZ_ID"] = ev["eventID"].map(csz_lookup)
    ev_f = ev[(ev["depth"] <= max_depth) & (ev[declustering_flag] == "TRUE")]
    # The completeness history of a source is the one ESHM20 assigns to it, not
    # one reconstructed from where its events happen to fall: a source straddling
    # a superzone boundary would otherwise inherit its neighbour's history.
    published_csz = gdf_d.loc[gdf_d["ZONE_ID"] == zone_id, "CSZ_ID"].iloc[0]
    dom_csz = ev_f["CSZ_ID"].dropna().mode()
    dom_csz = dom_csz.iloc[0] if not dom_csz.empty else None
    mc_steps = (mc_shallow[mc_shallow["CSZ_ID"] == published_csz].sort_values("Time")
                if published_csz else pd.DataFrame(columns=["Time", "Mc"]))
    last_start = mc_steps["Time"].iloc[-1] if not mc_steps.empty else None
    return dict(events=ev_f, mc_steps=mc_steps, last_period_start=last_start,
                csz=published_csz, dominant_csz=dom_csz)


# How a published completeness level Mc relates to the 0.2 lattice.  "centre"
# is the earlier reading: Mc is the centre of the first complete bin,
# magnitudes go to the nearest centre on the 0.1-offset lattice.  "edge" reads
# Mc as a raw threshold, the lower edge of the first complete bin, so the
# lattice is offset 0.0 and bins are [c - 0.1, c + 0.1).
CONVENTIONS = ("centre", "edge")
# The offset of the bin centres from multiples of BIN_SIZE, by convention.
LATTICE_OFFSET = {"centre": BIN_OFFSET, "edge": 0.0}


def bin_magnitudes(m, convention):
    """Raw magnitudes to bin centres under one convention."""
    m = np.asarray(m, dtype=float)
    if convention == "centre":
        return np.round(np.round((m - BIN_OFFSET) / BIN_SIZE) * BIN_SIZE + BIN_OFFSET, 6)
    if convention == "edge":
        # Rounded before the floor: (3.7 + 0.1) / 0.2 is 18.999... in floating point.
        k = np.floor(np.round((m + BIN_SIZE / 2) / BIN_SIZE, 9))
        return np.round(k * BIN_SIZE, 6)
    raise ValueError(f"unknown convention {convention!r}; expected one of {CONVENTIONS}")


def completeness_steps(steps, convention):
    """A CSZ history with each Mc replaced by the centre of its first complete bin."""
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown convention {convention!r}; expected one of {CONVENTIONS}")
    shift = 0.0 if convention == "centre" else BIN_SIZE / 2
    return steps.assign(Mc=np.round(steps["Mc"].to_numpy(float) + shift, 6))


def complete_mask(mags_raw, years, steps, convention):
    """
    Which raw events are complete, each against its own period's threshold.

    Returns the mask in the order given, with the binned magnitudes and the
    thresholds (centres of the first complete bin) it was decided on.
    """
    steps = completeness_steps(steps, convention)
    times, mcs = steps["Time"].to_numpy(), steps["Mc"].to_numpy(float)
    mags = bin_magnitudes(mags_raw, convention)
    period = np.searchsorted(times, np.asarray(years), side="right") - 1
    inside = period >= 0
    thr = np.where(inside, mcs[np.clip(period, 0, None)], np.inf)
    return inside & (mags >= thr - 1e-9), mags, thr


def complete_catalogue(mags_raw, years, steps, convention):
    """
    The complete events among raw ones, binned, each with its own period's threshold.

    The single place where raw magnitudes become a catalogue: every count, b
    and estimator downstream goes through here, so they cannot disagree on
    which events are complete.  ``thr`` is the centre of the first complete bin.
    """
    if steps.empty or len(mags_raw) == 0:
        return dict(mags=np.array([]), thr=np.array([]), years=np.array([]))
    years = np.asarray(years)
    keep, mags, thr = complete_mask(mags_raw, years, steps, convention)
    order = np.argsort(mags[keep])
    return dict(mags=mags[keep][order], thr=thr[keep][order],
                years=years[keep][order])


def all_periods_catalogue(r, convention):
    """Every complete event of a zone from `analyze_zone`, under one convention."""
    return complete_catalogue(r["events"]["magnitude"].to_numpy(float),
                              r["events"]["year"].to_numpy(), r["mc_steps"], convention)


def last_period_catalogue(r, convention):
    """Sorted magnitudes of the last completeness period, and that period's threshold."""
    cat = all_periods_catalogue(r, convention)
    if r["last_period_start"] is None or len(cat["mags"]) == 0:
        return np.array([]), None
    recent = cat["years"] >= r["last_period_start"]
    mc_z = float(completeness_steps(r["mc_steps"], convention)["Mc"].iloc[-1])
    return np.sort(cat["mags"][recent]), mc_z


def rp_sensitivity(zone_row, mags, mc_z, mmax):
    """
    Ratios to the Aki-Utsu plug-in at m_RP, across a range of return periods.

    The magnitude-domain comparison is made at a single return period, so its
    size is partly a consequence of that choice; this records the dependence.
    Return periods whose m_RP exceeds Mmax are NaN.
    """
    delta = infer_delta(mags)
    n_obs = len(mags)
    _, t_mag, w_mag = magnitude_statistics_rounded(mags, mc_z, delta)
    b_hat = n_obs / (t_mag * LOG10)

    m_rp = (zone_row["A_VAL"] - np.log10(1.0 / RP_CURVE)) / zone_row["B_VAL"]
    u = (m_rp - mc_z + delta) / (2 * delta)
    p_aki = 10.0 ** (-b_hat * (m_rp - mc_z + delta))

    keep = m_rp <= mmax
    rf = bfree_survival_rounded(w_mag, n_obs, u)[0] / p_aki
    rb = bayesian_survival_rounded(w_mag, n_obs, u)[0] / p_aki
    return np.where(keep, rf, np.nan), np.where(keep, rb, np.nan)


def zone_curves(zone_row, mags, mc_z):
    """
    Survival functions of the three estimators for one zone, on a common grid of
    magnitude thresholds running from m0 - delta past the 250-year magnitude.

    The estimators are functions of the bin coordinate u, and the exceedance
    level of bin u applies at that bin's lower edge, so a threshold m corresponds
    to u = (m - m0 + delta) / (2*delta).  Evaluating at real u rather than
    snapping to the nearest bin assumes Gutenberg-Richter behaviour within a bin.
    """
    n_obs = len(mags)
    delta = infer_delta(mags)
    m_250 = (zone_row["A_VAL"] - np.log10(1 / RETURN_PERIOD)) / zone_row["B_VAL"]

    _, t_mag, w_mag = magnitude_statistics_rounded(mags, mc_z, delta)
    b_hat = n_obs / (t_mag * LOG10)

    m_thresh = np.linspace(mc_z - delta, m_250 + 1.0, N_CURVE_POINTS)

    def at(m):
        """The three survival functions evaluated at magnitude *m*."""
        u = (np.atleast_1d(m) - mc_z + delta) / (2 * delta)
        return (gr_survival(np.atleast_1d(m), b_hat, mc_z - delta),
                bfree_survival_rounded(w_mag, n_obs, u)[0],
                bayesian_survival_rounded(w_mag, n_obs, u)[0])

    s_aki, s_bfree, s_bayes = at(m_thresh)
    return dict(
        n=n_obs, mc=mc_z, delta=delta, m_250=m_250, b_aki=b_hat,
        w_mag=w_mag,
        m_thresh=m_thresh,
        surv_aki=s_aki, surv_bfree=s_bfree, surv_bayes=s_bayes,
        p_250=tuple(float(x[0]) for x in at(m_250)),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load the catalogue, source zones and time-varying completeness levels."""
    missing = [p for p in (
        DATA / "eshm20_input_a_unified_eq_catalogue"
             / "eshm20_unified_catalogue_declustered_v02a.shp",
        DATA / "eshm20_input_d_asm_shallow_crust"
             / "eshm20_asz_shallow_winGT_fs017_dm02_results_rev.shp",
        DATA / "eshm20_input_b_completeness_superzones" / "eshm20_csz_v06b.shp",
        DATA / "eshm20_input_b_completeness_superzones"
             / "McTable_CSZ_ESHM20_ver_dec2020.xlsx",
    ) if not p.exists()]
    if missing:
        download(DATA)

    gdf = gpd.read_file(DATA / "eshm20_input_a_unified_eq_catalogue"
                        / "eshm20_unified_catalogue_declustered_v02a.shp")
    gdf_d = gpd.read_file(DATA / "eshm20_input_d_asm_shallow_crust"
                          / "eshm20_asz_shallow_winGT_fs017_dm02_results_rev.shp")
    gdf_b = gpd.read_file(DATA / "eshm20_input_b_completeness_superzones"
                          / "eshm20_csz_v06b.shp")
    mc_shallow = pd.ExcelFile(
        DATA / "eshm20_input_b_completeness_superzones"
        / "McTable_CSZ_ESHM20_ver_dec2020.xlsx"
    ).parse("McTable_CSZ_branch01_shallow").copy()

    gdf_with_csz = gpd.sjoin(
        gdf[["eventID", "year", "magnitude", "geometry"]],
        gdf_b[["CSZ_ID", "geometry"]], how="left", predicate="within"
    ).drop(columns="index_right")
    merged = gdf_with_csz.merge(mc_shallow[["CSZ_ID", "Mc", "Time"]],
                                on="CSZ_ID", how="left")
    merged = merged[merged["Time"] <= merged["year"]]
    best_idx = merged.groupby("eventID")["Time"].idxmax()
    event_mc = merged.loc[best_idx, ["eventID", "Mc", "Time"]].set_index("eventID")
    return gdf, gdf_d, gdf_with_csz, mc_shallow, event_mc


def zone_catalogues(gdf, gdf_d, mc_shallow, convention):
    """Every zone's complete catalogue, all periods, from one spatial join."""
    ev = gpd.sjoin(
        gdf[["eventID", "year", "magnitude", "depth", "winGT_fs01", "geometry"]],
        gdf_d[["ZONE_ID", "geometry"]], how="inner", predicate="within"
    ).drop(columns="index_right")
    ev = ev[(ev["depth"] <= MAX_DEPTH) & (ev["winGT_fs01"] == "TRUE")]
    by_zone = dict(tuple(ev.groupby("ZONE_ID")))
    steps_by_csz = dict(tuple(mc_shallow.sort_values("Time").groupby("CSZ_ID")))
    empty_steps = pd.DataFrame(columns=["Time", "Mc"])
    empty = dict(mags=np.array([]), thr=np.array([]), years=np.array([]))
    out = {}
    for zid, csz in zip(gdf_d["ZONE_ID"], gdf_d["CSZ_ID"], strict=True):
        zev = by_zone.get(zid)
        out[zid] = empty if zev is None else complete_catalogue(
            zev["magnitude"].to_numpy(float), zev["year"].to_numpy(),
            steps_by_csz.get(csz, empty_steps), convention)
    return out


def classify_zones(gdf, gdf_d, mc_shallow, convention):
    """
    Flag the zones for which ESHM20 used a zone-level (data-based) b-value.

    Per the ESHM20 report, section 4: a zone is data-based when it has at least
    30 complete events over all completeness periods and its b-value converges
    within the validity range; otherwise a regional (TECTO) fall-back is used.
    Zones whose B_VAL matches the TECTO modal b but whose data b is in range are
    treated as coincidental matches.  Counts and b come from `complete_catalogue`
    with the zone's published completeness history, as everywhere else, each
    event measured from its own period's threshold.
    """
    rows = []
    for zid, cat in zone_catalogues(gdf, gdf_d, mc_shallow, convention).items():
        b = np.nan
        if len(cat["mags"]) >= 2:
            n, t, _ = magnitude_statistics_rounded(cat["mags"], cat["thr"], BIN_SIZE / 2)
            b = n / (t * LOG10)
        rows.append(dict(ZONE_ID=zid, n_complete_all=len(cat["mags"]), b_aki_all=b))
    gdf_d = gdf_d.merge(pd.DataFrame(rows), on="ZONE_ID", how="left")
    gdf_d["n_complete_all"] = gdf_d["n_complete_all"].fillna(0).astype(int)

    tecto_b = gdf_d.groupby("TECTO_ID")["B_VAL"].agg(lambda x: x.mode().iloc[0])
    gdf_d["TECTO_B"] = gdf_d["TECTO_ID"].map(tecto_b)
    matches_tecto = np.isclose(gdf_d["B_VAL"], gdf_d["TECTO_B"], atol=1e-4)
    b_valid = gdf_d["b_aki_all"].between(*B_VALID_RANGE).fillna(False)

    gdf_d["eshm20_class"] = np.select(
        [~matches_tecto,
         gdf_d["n_complete_all"] < MIN_EVENTS,
         b_valid],
        ["data-based (distinct b)",
         "fallback (too few events)",
         "data-based (coincidental TECTO match)"],
        default="fallback (b outside valid range)",
    )
    gdf_d["eshm20_data_based"] = pd.Series(gdf_d["eshm20_class"]).str.startswith("data-based")
    return gdf_d


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def collect_zone_results(gdf, gdf_d, gdf_with_csz, mc_shallow, convention,
                         data_based_only=True):
    """Evaluate all three estimators at m_250 for every qualifying zone."""
    records, skipped_mmax, failed, inclusion = [], [], [], []
    sens_free, sens_bayes = [], []

    for i, zid in enumerate(gdf_d["ZONE_ID"].tolist()):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {len(gdf_d)} zones processed ...")
        zone_row = gdf_d[gdf_d["ZONE_ID"] == zid].iloc[0]
        lowest_m_max = min(zone_row["MAXMAG01"], zone_row["MAXMAG02"],
                           zone_row["MAXMAG03"])
        # One row per zone, recorded before any filter so that the inventory
        # accounts for every zone in the model, and filled in as the analysis
        # proceeds.  The classification inputs are the ones reconstruct_class()
        # used, so the reconstruction can be audited zone by zone.
        rec = dict(
            zone_id=zid, tecto_id=zone_row["TECTO_ID"],
            n_complete_all=int(zone_row["n_complete_all"]),
            b_recomputed=float(zone_row["b_aki_all"]),
            b_in_valid_range=bool(B_VALID_RANGE[0] <= zone_row["b_aki_all"]
                                  <= B_VALID_RANGE[1]),
            b_published=float(zone_row["B_VAL"]),
            tecto_modal_b=float(zone_row["TECTO_B"]),
            matches_tecto_b=bool(np.isclose(zone_row["B_VAL"], zone_row["TECTO_B"],
                                            atol=1e-4)),
            eshm20_class=zone_row["eshm20_class"],
            data_based=bool(zone_row["eshm20_data_based"]),
            lowest_m_max=lowest_m_max,
            csz="", n_complete_last=np.nan, n=np.nan, mc=np.nan,
            b_aki=np.nan, m_250=np.nan,
            candidate=False, included=False, exclusion="",
        )
        inclusion.append(rec)
        try:
            r = analyze_zone(gdf_d, gdf_with_csz, mc_shallow, gdf, zid)
            mags, mc_z = last_period_catalogue(r, convention)
            rec.update(csz=r["csz"], n_complete_last=len(mags))
            if len(mags) < MIN_EVENTS:
                rec["exclusion"] = f"fewer than {MIN_EVENTS} events in last period"
                continue
            if data_based_only and not zone_row["eshm20_data_based"]:
                rec["exclusion"] = "not data-based"
                continue

            c = zone_curves(zone_row, mags, mc_z)
            rec.update(n=c["n"], mc=c["mc"], b_aki=c["b_aki"], m_250=c["m_250"],
                       candidate=True, included=bool(c["m_250"] <= lowest_m_max))
            if c["m_250"] > lowest_m_max:
                rec["exclusion"] = "m_250 > lowest Mmax"
                skipped_mmax.append(zid)
                continue

            p_aki, p_bfree, p_bayes = c["p_250"]
            rf, rb = rp_sensitivity(zone_row, mags, mc_z, lowest_m_max)
            sens_free.append(rf)
            sens_bayes.append(rb)
            records.append(dict(
                zone_id=zid, n=c["n"], n_complete_all=int(zone_row["n_complete_all"]),
                mc=c["mc"], delta=c["delta"], b_aki=c["b_aki"], w_mag=c["w_mag"],
                csz=r["csz"],
                m_250=c["m_250"], lowest_m_max=lowest_m_max,
                eshm20_class=zone_row["eshm20_class"],
                P_aki=p_aki, P_bfree=p_bfree, P_bayes=p_bayes,
                ratio_bfree=p_bfree / p_aki if p_aki > 0 else np.nan,
                ratio_bayes=p_bayes / p_aki if p_aki > 0 else np.nan,
            ))
        except Exception as exc:
            rec["exclusion"] = "analysis failed"
            failed.append((zid, repr(exc)))

    res = pd.DataFrame(records).sort_values("n").reset_index(drop=True)
    # Keyed by zone: res is re-sorted below, the sensitivity arrays are not.
    sensitivity = {"zone_id": [r["zone_id"] for r in records],
                   "return_periods": RP_CURVE,
                   "ratio_bfree": np.array(sens_free),
                   "ratio_bayes": np.array(sens_bayes)}
    return res, skipped_mmax, failed, sensitivity, pd.DataFrame(inclusion)


def prior_sensitivity(res, b_priors=(0.8, 1.0, 1.2), n0_priors=(10, 100)):
    """
    The qualifying-zone median ratio under conjugate priors instead of Jeffreys.

    The application quotes a reference-prior number, and a reader cannot tell
    from it how much of the answer is the prior; this computes that rather than
    leaving it to be reconstructed.
    """
    out = []
    for b_prior in b_priors:
        for n0_prior in n0_priors:
            ratios = []
            for _, r in res.iterrows():
                w0, n0 = conjugate_prior_rounded(b_prior, n0_prior, r["delta"])
                u = (r["m_250"] - r["mc"] + r["delta"]) / (2 * r["delta"])
                p = float(bayesian_survival_rounded(np.array([r["w_mag"]]), int(r["n"]),
                                            np.array([u]), w0=w0, n0=n0)[0][0])
                ratios.append(p / r["P_aki"])
            out.append({"b_prior": b_prior, "n0_prior": n0_prior,
                        "median_ratio": float(np.median(ratios)),
                        "ratios": np.array(ratios)})
    return out


def select_highlights(res, small_max_n=40):
    """
    Pick the three zones shown in detail in the manuscript:
      A - representative small catalogue (n < small_max_n), closest to the
          median divergence of that same subset;
      B - largest catalogue, where all methods converge;
      C - largest divergence between the Bayesian and frequentist b-free results.
    """
    small = res[res["n"] < small_max_n].copy()
    # Target the subset's own medians, so the example tracks the analysis rather
    # than a pair of values fixed when the numbers were different.  Distances are
    # taken in log space because these are ratios; that also breaks a near-tie
    # that the linear distance leaves between the two zones straddling the median.
    target_free = small["ratio_bfree"].median()
    target_bayes = small["ratio_bayes"].median()
    small["_dist"] = (np.log(small["ratio_bfree"] / target_free) ** 2
                      + np.log(small["ratio_bayes"] / target_bayes) ** 2) ** 0.5
    zone_a = small.loc[small["_dist"].idxmin(), "zone_id"]
    zone_b = res.loc[res["n"].idxmax(), "zone_id"]
    divergence = (res["ratio_bayes"] / res["ratio_bfree"].replace(0, np.nan)).dropna()
    zone_c = res.loc[divergence.idxmax(), "zone_id"]
    return {"A": zone_a, "B": zone_b, "C": zone_c}


def highlight_curves(gdf, gdf_d, gdf_with_csz, mc_shallow, convention, highlights):
    """Full survival curves and observed catalogue for each highlight zone."""
    out = {}
    for label, zid in highlights.items():
        r = analyze_zone(gdf_d, gdf_with_csz, mc_shallow, gdf, zid)
        mags, mc_z = last_period_catalogue(r, convention)
        zone_row = gdf_d[gdf_d["ZONE_ID"] == zid].iloc[0]
        out[label] = dict(zone_id=zid, magnitudes=mags,
                          **zone_curves(zone_row, mags, mc_z))
    return out


def run(output_path, convention, data_based_only=True, zone_table=None):
    """Apply the estimators to the ESHM20 source zones and write the results."""
    warnings.filterwarnings("ignore")

    print("Loading data ...")
    gdf, gdf_d, gdf_with_csz, mc_shallow, _ = load_data()
    print(f"Loaded {len(gdf)} events, {len(gdf_d)} zones.  Mc read as bin {convention}.")

    print("Classifying zones ...")
    gdf_d = classify_zones(gdf, gdf_d, mc_shallow, convention)
    print(gdf_d["eshm20_class"].value_counts().to_string())

    mode = "data-based zones only" if data_based_only else "all qualifying zones"
    print(f"\nCollecting per-zone statistics ({mode}) ...")
    res, skipped_mmax, failed, sensitivity, inclusion = collect_zone_results(
        gdf, gdf_d, gdf_with_csz, mc_shallow, convention, data_based_only)

    highlights = select_highlights(res)
    curves = highlight_curves(gdf, gdf_d, gdf_with_csz, mc_shallow, convention,
                              highlights)

    summary = {
        "n_zones_total": len(gdf_d),
        "n_zones_qualifying": len(res),
        "n_skipped_mmax": len(skipped_mmax),
        "n_failed": len(failed),
        "frac_bfree_below_aki": float((res["ratio_bfree"] < 1).mean()),
        "frac_bayes_below_aki": float((res["ratio_bayes"] < 1).mean()),
        "median_ratio_bfree": float(res["ratio_bfree"].median()),
        "median_ratio_bayes": float(res["ratio_bayes"].median()),
    }
    priors = prior_sensitivity(res)

    print(f"\nQualifying zones       : {summary['n_zones_qualifying']}")
    print(f"Skipped (m_250 > Mmax) : {summary['n_skipped_mmax']}")
    print(f"Failed                 : {summary['n_failed']}")
    print(f"Median ratio, frequentist b-free : {summary['median_ratio_bfree']:.3f}")
    print(f"Median ratio, Bayesian b-free    : {summary['median_ratio_bayes']:.3f}")
    print("Median ratio under conjugate priors:")
    for row in priors:
        print(f"   b_prior={row['b_prior']:.1f} n0={row['n0_prior']:>3}"
              f"  -> {row['median_ratio']:.3f}")
    print(f"Highlight zones: {highlights}")

    print(f"\nInventory ({len(inclusion)} zones, one row each):")
    counts = inclusion["exclusion"].replace("", "retained").value_counts()
    for reason, count in counts.items():
        print(f"   {reason:<45} {count:>4}")
    print(f"   {'candidates (reaching the Mmax test)':<45} "
          f"{int(inclusion['candidate'].sum()):>4}")
    if zone_table:
        emit_zone_table(inclusion, zone_table)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_results(output_path, {
        "zones": res,
        "highlights": highlights,
        "highlight_curves": curves,
        "rp_sensitivity": sensitivity,
        "prior_sensitivity": priors,
        "inclusion": inclusion,
        "skipped_mmax": skipped_mmax,
        "failed": failed,
        "summary": summary,
        "metadata": {
            "return_period": RETURN_PERIOD,
            "min_events": MIN_EVENTS,
            "bin_size": BIN_SIZE,
            "bin_offset": LATTICE_OFFSET[convention],
            "mc_convention": convention,
            "max_depth": MAX_DEPTH,
            "data_based_only": data_based_only,
        },
    })
    print(f"\nSaved: {output_path}")


def emit_zone_table(inclusion, path):
    """Write the supplementary zone table.

    Generated rather than transcribed, so the table cannot drift from the
    artifact it reports.  Two zones per printed row.
    """
    cand = (inclusion[inclusion["candidate"]]
            .sort_values("zone_id").reset_index(drop=True))
    half = (len(cand) + 1) // 2
    left, right = cand.iloc[:half], cand.iloc[half:].reset_index(drop=True)

    def cells(row):
        """One row of the supplementary zone table, or an empty row."""
        if row is None:
            return " & " * 7 + " "
        # Zones admitted by the coincidental-match rule are the contestable part
        # of the reconstruction, so they are marked for the reader.
        mark = "$^\\dagger$" if "coincidental" in row["eshm20_class"] else ""
        return (f"{row['zone_id']}{mark} & {row['csz']} & {int(row['n'])} & "
                f"{row['mc']:.1f} & {row['b_aki']:.2f} & {row['m_250']:.2f} & "
                f"{row['lowest_m_max']:.1f} & "
                f"{'yes' if row['included'] else 'no'}")

    n_coinc = int(cand["eshm20_class"].str.contains("coincidental").sum())
    header = ("zone & CSZ & $n$ & $m_c$ & $\\hat{b}$ & $m_{250}$ & "
              "$m_{\\mathrm{max}}$ & in")
    # The whole tabular is emitted, not just its rows: TeX cannot take an \input
    # boundary inside an alignment.
    lines = [f"% Generated by scripts/analyse_eshm20.py --emit-zone-table; do not edit.\n"
             f"% {len(cand)} candidate zones, {int(cand['included'].sum())} retained, "
             f"{n_coinc} admitted by the coincidental-match rule.\n",
             "\\begin{tabular}{llrrrrrc @{\\hspace{1em}} llrrrrrc}\n",
             "\\toprule\n",
             f"{header} &\n{header} \\\\\n",
             "\\midrule\n"]
    for i in range(half):
        r = right.iloc[i] if i < len(right) else None
        lines.append(f"{cells(left.iloc[i])} &\n{cells(r)} \\\\\n")
    lines += ["\\bottomrule\n", "\\end{tabular}\n"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as fh:
        fh.writelines(lines)
    print(f"Wrote {len(cand)} candidate zones to {path}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Download the ESHM20 input data and apply the b-free "
                    "estimators to the ESHM20 source zones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", default="results/eshm20_results.zarr.zip", metavar="FILE")
    parser.add_argument(
        "--mc-convention", choices=CONVENTIONS,
        help="How a published completeness level Mc relates to the 0.2 magnitude "
             "bins: 'edge' reads it as the lower edge of the first complete bin "
             "(raw magnitude >= Mc), 'centre' as that bin's centre, which admits "
             "raw magnitudes from Mc - 0.1.  Required for every analysis; see "
             "multiperiod_notes.md for the evidence.",
    )
    parser.add_argument(
        "--all-zones", action="store_true",
        help="Also include zones for which ESHM20 fell back to a regional "
             "b-value instead of a zone-specific one.",
    )
    parser.add_argument(
        "--emit-zone-table", metavar="FILE",
        help="Also write the supplementary zone table, as a complete tabular "
             "environment, to FILE.",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Fetch and verify the input data, then exit without analysing.",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download the input data even if it is already present.",
    )
    parser.add_argument(
        "--print-checksums", action="store_true",
        help="Print the downloaded files' SHA-256 digests and exit "
             "(used to regenerate CHECKSUMS after a version bump).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        if args.download_only or args.force_download or args.print_checksums:
            download(DATA, args.force_download, args.print_checksums)
            if args.download_only or args.print_checksums:
                sys.exit(0)
        if args.mc_convention is None:
            sys.exit("error: --mc-convention is required (edge or centre)")
        run(args.output, args.mc_convention, data_based_only=not args.all_zones,
            zone_table=args.emit_zone_table)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")
