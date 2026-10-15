"""Small helpers shared by the package and the command-line scripts."""

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def file_md5(path):
    """MD5 of a file's bytes, read in chunks so large stores do not load whole.

    Used for integrity checks against what Zenodo reports, not for security.
    """
    digest = hashlib.md5()  # noqa: S324
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_md5(path):
    """MD5 of a file, or of a directory's contents taken as a whole.

    A result store is a directory, so identifying one needs the whole tree. The
    relative path of each member goes into the digest as well as its bytes, so
    that renaming a member changes the answer.
    """
    path = Path(path)
    if path.is_file():
        return file_md5(path)
    digest = hashlib.md5()  # noqa: S324
    for member in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(member.relative_to(path)).encode())
        digest.update(file_md5(member).encode())
    return digest.hexdigest()


def human_size(size):
    """A byte count as a short human-readable string, or '?' if unknown.

    Powers of 1024, labelled as such: these figures sit beside the sizes Zenodo
    reports, and those are powers of 1000.
    """
    if size is None:
        return "?"
    size = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")
