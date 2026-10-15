#!/usr/bin/env python
"""
package_for_zenodo.py

Assemble the archive that `retrieve_from_zenodo.py` expects, so that the two
ends of the loop cannot drift apart: this script writes exactly the file names
that one reads, and fails if any of them is missing.

    eshm20_results.zarr.zip            ESHM20 per-zone results
    hazard_results.zarr.zip            ground motion exceedance ratios per zone
    analysis_results.zarr.zip          reduced synthetic results, rounded
    analysis_results_continuous.zarr.zip   reduced synthetic results, unrounded
    simulation_results.zarr.zip        raw synthetic catalogues (large)

Each reduced result is already a single-file Zarr store and is copied as it is.
The raw catalogues are a directory, so they are zipped here; they are included
only with --with-zarr, since they are around 1.3 GB and are needed only to
re-run the reduction.

Alongside the files it writes CHECKSUMS.txt and MANIFEST.json, recording the
size and MD5 of each, the commit that produced them, and the arguments the
generating scripts were given, so that a deposit can be traced back to a run.

Usage
-----
    python scripts/package_for_zenodo.py --out dist/zenodo
    python scripts/package_for_zenodo.py --out dist/zenodo --with-zarr
"""

import argparse
import json
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from bfree.provenance import code_commit
from bfree.util import file_md5, human_size

SMALL = (
    "eshm20_results.zarr.zip",
    "hazard_results.zarr.zip",
    "analysis_results.zarr.zip",
    "analysis_results_continuous.zarr.zip",
)
ZARR_DIR = "simulation_results.zarr"
ZARR_ZIP = "simulation_results.zarr.zip"


def _zip_store(source, dest):
    """Zip a Zarr store directory with stored (uncompressed) entries.

    The chunks are already compressed, so deflating them again costs minutes
    and saves almost nothing. Members are written under the store's own name,
    so unzipping reconstructs the directory.
    """
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source.parent).as_posix())


def package(results_dir, out_dir, with_zarr):
    """Assemble a deposit in *out_dir*, with checksums and a manifest.

    Refuses to write a partial deposit: every file `retrieve_from_zenodo.py`
    expects must be present, or nothing is copied.
    """
    results_dir, out_dir = Path(results_dir), Path(out_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"results directory does not exist: {results_dir}")

    missing = [n for n in SMALL if not (results_dir / n).is_file()]
    if with_zarr and not (results_dir / ZARR_DIR).is_dir():
        missing.append(ZARR_DIR)
    if missing:
        raise SystemExit(
            "missing input(s): " + ", ".join(missing)
            + "\nGenerate them first (make simulate eshm20) or fetch them "
              "with retrieve_from_zenodo.py."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []

    for name in SMALL:
        target = out_dir / name
        shutil.copy2(results_dir / name, target)
        entries.append(target)

    if with_zarr:
        target = out_dir / ZARR_ZIP
        print(f"zipping {ZARR_DIR} (this takes a few minutes) ...")
        _zip_store(results_dir / ZARR_DIR, target)
        entries.append(target)

    records = [
        {"name": p.name, "size": p.stat().st_size, "md5": file_md5(p)} for p in entries
    ]

    (out_dir / "CHECKSUMS.txt").write_text(
        "".join(f"{r['md5']}  {r['name']}\n" for r in records)
    )
    (out_dir / "MANIFEST.json").write_text(json.dumps({
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_commit": code_commit(),
        "files": records,
    }, indent=2) + "\n")

    print(f"\n{'file':<40}{'size':>12}  md5")
    for r in records:
        print(f"{r['name']:<40}{human_size(r['size']):>12}  {r['md5']}")
    print(f"\nwritten to {out_dir}")
    print("Upload these files to Zenodo, then set ZENODO_RECORD (or --record) "
          "in retrieve_from_zenodo.py to the new record id.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results-dir", default="results",
                        help="where the generated outputs are")
    parser.add_argument("--out", required=True,
                        help="directory to assemble the deposit in")
    parser.add_argument("--with-zarr", action="store_true",
                        help="also include the raw catalogues, around 1.3 GB")
    args = parser.parse_args()
    return package(args.results_dir, args.out, args.with_zarr)


if __name__ == "__main__":
    sys.exit(main())
