"""
Provenance for generated figures.

A committed figure carries no record of which code produced it. That is how a
figure set ends up mixed: some panels from one version of the pipeline and some
from another, with nothing in the repository that disagrees. The manifest
written here is that record, and it is written by the same call that writes the
figure so that it cannot be forgotten separately.

The image hash is over the PNG's compressed image data, not the file: matplotlib
stamps its own version into a text chunk, so two byte-different files can be
pixel-identical.
"""

import hashlib
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from bfree.util import file_md5, tree_md5

MANIFEST_NAME = "MANIFEST.tsv"
COLUMNS = ("figure", "image_md5", "code_commit", "inputs_opened", "generated")


def _image_md5(path):
    """A hash of the picture, not of the file.

    For PNG that means the IDAT chunks, because matplotlib stamps its version
    into a text chunk and two pixel-identical files would otherwise hash
    differently. For PDF the whole file serves, but only because `save_figure`
    suppresses the creation date and fixes the producer strings. EPS carries
    its date in a header comment that matplotlib does not expose, so that one
    line is dropped before hashing.
    """
    if path.suffix == ".pdf":
        return file_md5(path)
    if path.suffix == ".eps":
        digest = hashlib.md5()  # noqa: S324
        for line in path.read_bytes().splitlines(keepends=True):
            if not line.startswith(b"%%CreationDate:"):
                digest.update(line)
        return digest.hexdigest()
    if path.suffix != ".png":
        raise ValueError(f"no image hash defined for {path.suffix} files")

    data = path.read_bytes()[8:]
    digest = hashlib.md5()  # noqa: S324 - an identity check, not a security one
    offset = 0
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        if data[offset + 4:offset + 8] == b"IDAT":
            digest.update(data[offset + 8:offset + 8 + length])
        offset += 12 + length
    return digest.hexdigest()



def _git(root, *args):
    """Run git in *root* and return its stdout, stripped."""
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(root), *args],  # noqa: S607
        capture_output=True, text=True, check=True).stdout.strip()


def code_commit():
    """The commit of the repository this module is installed from.

    Suffixed `-dirty` when the tree has uncommitted changes, because a clean
    hash against a dirty tree is worse than no hash at all.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        sha = _git(root, "rev-parse", "--short", "HEAD")
        dirty = _git(root, "status", "--porcelain")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def record_figure(figure_path, inputs=()):
    """Record one figure in the manifest beside it, replacing any earlier row.

    *inputs* are the data files opened before the figure was written. They are
    hashed because they live outside version control, so naming them is not
    enough. The list over-reports rather than under-reports: a caller that
    tracks opened files gives every figure everything read up to that point,
    which errs toward regenerating too much.
    """
    figure_path = Path(figure_path)
    manifest = figure_path.parent / MANIFEST_NAME

    rows = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines()[1:]:
            if line.strip():
                rows[line.split("\t")[0]] = line

    stamp = "; ".join(f"{Path(p).name}={tree_md5(p)[:12]}" for p in inputs) or "-"
    rows[figure_path.name] = "\t".join((
        figure_path.name,
        _image_md5(figure_path),
        code_commit(),
        stamp,
        datetime.now(UTC).strftime("%Y-%m-%d"),
    ))

    manifest.write_text("\t".join(COLUMNS) + "\n"
                        + "\n".join(rows[k] for k in sorted(rows)) + "\n")
    return manifest
