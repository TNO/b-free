#!/usr/bin/env python
"""
retrieve_from_zenodo.py

Downloads the archived outputs of run_simulations.py, reduce_simulations.py and
analyse_eshm20.py so that the figures can be made without repeating the
simulations, which take hours and around 9 GB of RAM.

    eshm20_results.zarr.zip            ESHM20 per-zone results        (small)
    hazard_results.zarr.zip            ground motion exceedance
                                       ratios per zone               (small)
    analysis_results.zarr.zip          reduced synthetic results,
                                       rounded catalogues            (small)
    analysis_results_continuous.zarr.zip  reduced synthetic results,
                                       unrounded catalogues          (small)
    simulation_results.zarr            raw synthetic catalogues       (~1.3 GB)

The reduced results are single-file Zarr stores and are written into results/
as they are. The raw catalogues are a directory, held zipped on Zenodo and
unpacked here.

The four reduced results cover every figure but the plug-in comparison, which
is drawn from the raw catalogues directly; those are also what
reduce_simulations.py reruns on. They are therefore downloaded on request
rather than by default.

Every file is verified against the MD5 checksum that Zenodo reports for it.

Usage
-----
    python retrieve_from_zenodo.py                 # the reduced results
    python retrieve_from_zenodo.py --with-zarr     # also the raw catalogues
    python retrieve_from_zenodo.py --list          # show what the record holds
"""

import argparse
import hashlib
import json
import os
import posixpath
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from bfree.util import human_size

# Record id of the archived data. This is the version DOI, so it pins the exact
# files the manuscript was built from; use the concept record id instead to always
# get the latest version. Override with --record or the ZENODO_RECORD variable.
ZENODO_RECORD = os.environ.get("ZENODO_RECORD", "21773391")

CITATION = """\
The archived data are licensed CC-BY 4.0 and should be cited as:

  Osinga, S., Kraaijpoel, D., Aben, F., & Pluymaekers, M. (2026). Data for
  "b-free: Inferring Gutenberg-Richter earthquake size distributions without
  estimating b-values" [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21773391
"""

# Zenodo file name -> local target. A target that is not itself a .zip unpacks.
RESULTS = Path("results")
TARGETS = {
    "eshm20_results.zarr.zip": RESULTS / "eshm20_results.zarr.zip",
    "hazard_results.zarr.zip": RESULTS / "hazard_results.zarr.zip",
    "analysis_results.zarr.zip": RESULTS / "analysis_results.zarr.zip",
    "analysis_results_continuous.zarr.zip": RESULTS / "analysis_results_continuous.zarr.zip",
    "simulation_results.zarr.zip": RESULTS / "simulation_results.zarr",
}

SMALL = ["eshm20_results.zarr.zip", "hazard_results.zarr.zip",
         "analysis_results.zarr.zip",
         "analysis_results_continuous.zarr.zip"]
LARGE = ["simulation_results.zarr.zip"]

# Checksums pinned here rather than taken only from the API response.
#
# Verifying against the MD5 the record reports catches corruption in transit
# but nothing else: that digest arrives from the same response as the download
# URL, so it agrees with whatever the record holds. Pinning the digest in the
# repository is what makes the check meaningful, because an altered record then
# disagrees with this file.
#
# The stores hold arrays and JSON attributes, so a substituted file yields wrong
# numbers rather than running code, but wrong numbers that look plausible are
# worth catching too.
#
# These are the digests package_for_zenodo.py recorded in CHECKSUMS.txt for the
# deposit at record 21773391, assembled at commit a029c0d. Regenerating any
# store rewrites its archive and changes the digest even when the numbers are
# identical, so a mismatch here means the record and this file have drifted
# apart, not necessarily that the data are wrong.
EXPECTED_MD5 = {
    "eshm20_results.zarr.zip": "b3ef84a950fea94be24e58fee9372971",
    "hazard_results.zarr.zip": "c9922ec5d66a9168f2bbbb3f68e12f87",
    "analysis_results.zarr.zip": "05e1817c8a1328665049f46747dc0e85",
    "analysis_results_continuous.zarr.zip": "25b60eecb690a9d782641ecfe3db0f75",
    "simulation_results.zarr.zip": "a4f6f6fcea0d6a7de3ee73439624be1a",
}

CHUNK = 1 << 20


def _api_url(record, sandbox):
    """The Zenodo REST endpoint for one record, on the live site or the sandbox."""
    host = "sandbox.zenodo.org" if sandbox else "zenodo.org"
    return f"https://{host}/api/records/{record}"


def _open(url):
    """Open a URL, turning the two urllib failure modes into one message."""
    try:
        return urllib.request.urlopen(url)  # noqa: S310 - https URL built from a fixed host
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url} returned HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc


def manifest(record, sandbox):
    """Return {filename: {'size': int, 'md5': str, 'url': str}} for the record."""
    url = _api_url(record, sandbox)
    with _open(url) as response:
        payload = json.load(response)

    entries = payload.get("files", [])
    if isinstance(entries, dict):          # InvenioRDM-style serialisation
        entries = list(entries.get("entries", {}).values())

    files = {}
    for entry in entries:
        name = entry.get("key")
        if not name:
            continue
        checksum = entry.get("checksum", "")
        files[name] = {
            "size": entry.get("size"),
            "md5": checksum.split(":", 1)[-1] if checksum else "",
            "url": entry.get("links", {}).get("self")
                   or f"{url.replace('/api/records/', '/records/')}"
                      f"/files/{name}?download=1",
        }
    if not files:
        raise RuntimeError(f"record {record} lists no files")
    return files



def _download(url, dest, expected_md5, expected_size):
    """Download to *dest*, verifying size and MD5 before the file is put in place.

    Written to a .part file and renamed only once it checks out, so an
    interrupted run cannot leave something that looks like a complete download.
    """
    digest = hashlib.md5(usedforsecurity=False)
    done = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _open(url) as response, tmp.open("wb") as handle:
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            done += len(block)
            if expected_size:
                print(f"\r    {done * 100 / expected_size:5.1f}%"
                      f"  {human_size(done)} / {human_size(expected_size)}",
                      end="", flush=True)
    if expected_size:
        print()

    if expected_md5 and digest.hexdigest() != expected_md5:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name}: the download is "
                           "incomplete or the record has changed")
    tmp.replace(dest)


def _safe_member_name(name):
    """Reject archive entries that would escape the destination directory."""
    if posixpath.isabs(name) or name.startswith("/"):
        return False
    return ".." not in Path(name).parts


def _extract(archive_path, target, force):
    """Unpack a downloaded archive to *target*, skipping if it is already there."""
    if target.exists():
        if not force:
            print(f"  {target}: already present, skipping extraction")
            archive_path.unlink(missing_ok=True)
            return
        shutil.rmtree(target)

    print(f"  extracting into {target}/ ...")
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        for member in members:
            if not _safe_member_name(member):
                raise RuntimeError(f"unsafe archive entry: {member!r}")
        # Tolerate archives that wrap everything in a single top-level folder.
        roots = {Path(m).parts[0] for m in members}
        nested = len(roots) == 1 and all(len(Path(m).parts) > 1 for m in members)
        strip = f"{roots.pop()}/" if nested else ""
        for member in members:
            if member.endswith("/"):
                continue
            relative = member[len(strip):] if strip else member
            out = target / relative
            out.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    archive_path.unlink(missing_ok=True)


def _check_pinned(record, wanted, files):
    """Compare the record's digests with the ones pinned in this file.

    The question is not only whether the download arrived intact but whether it
    came from the deposit this repository was written against. Only a digest
    held here can answer that.
    """
    if not EXPECTED_MD5:
        print("  note: no checksums are pinned in this script, so integrity is\n"
              "        verified only against what the record itself reports.")
        return
    wrong = [n for n in wanted
             if n in EXPECTED_MD5 and files[n]["md5"] != EXPECTED_MD5[n]]
    if wrong:
        raise RuntimeError(
            f"record {record} reports a different checksum than this script "
            f"pins for: {', '.join(wrong)}\n"
            "Refusing to download. Either the record is not the one this code "
            "was written against, or EXPECTED_MD5 is out of date.")


def retrieve(record, wanted, sandbox=False, force=False):
    """Download the named files from a Zenodo record into the results directory.

    Fails before downloading anything if the record does not hold every file
    asked for, so a partial fetch is not mistaken for a complete one.
    """
    files = manifest(record, sandbox)

    missing = [name for name in wanted if name not in files]
    if missing:
        raise RuntimeError(
            f"record {record} does not contain: {', '.join(missing)}\n"
            f"available: {', '.join(sorted(files))}")

    _check_pinned(record, wanted, files)

    for name in wanted:
        target = TARGETS[name]
        if target.exists() and not force:
            print(f"  {target}: already present, skipping")
            continue

        info = files[name]
        print(f"  {name} ({human_size(info['size'])})")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".zip":
            _download(info["url"], target, info["md5"], info["size"])
        else:
            _download(info["url"], Path(name), info["md5"], info["size"])
            _extract(Path(name), target, force)

    print(f"\nDone.\n\n{CITATION}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", default=ZENODO_RECORD,
                        help="Zenodo record id (default: the archive for this paper)")
    parser.add_argument("--with-zarr", action="store_true",
                        help="also download the raw simulation catalogues (~1.3 GB)")
    parser.add_argument("--only", nargs="+", choices=sorted(TARGETS),
                        help="download only these files")
    parser.add_argument("--list", action="store_true",
                        help="list the files in the record and exit")
    parser.add_argument("--force", action="store_true",
                        help="re-download files that are already present")
    parser.add_argument("--sandbox", action="store_true",
                        help="use sandbox.zenodo.org (for testing a draft record)")
    args = parser.parse_args()

    if not args.record:
        raise RuntimeError(
            "no Zenodo record id configured; pass --record, set ZENODO_RECORD, "
            "or fill in ZENODO_RECORD at the top of this script")

    if args.list:
        for name, info in sorted(manifest(args.record, args.sandbox).items()):
            print(f"  {name:32s} {human_size(info['size']):>10s}  md5:{info['md5']}")
        return

    wanted = args.only or (SMALL + LARGE if args.with_zarr else SMALL)
    retrieve(args.record, wanted, sandbox=args.sandbox, force=args.force)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")
