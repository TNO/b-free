"""
Tests for the result store.

The store replaced pickles, so the thing worth testing is that it round-trips
the shapes the analyses actually produce - nested mappings, tuple keys, tables,
ragged sequences and text - without quietly changing a type along the way.
"""

import numpy as np
import pytest

from bfree import store

pd = pytest.importorskip("pandas")


def assert_same(a, b, path=""):
    """Deep equality that fails on a changed type, not only a changed value."""
    assert type(a) is type(b) or isinstance(a, type(b)) or isinstance(b, type(a)), (
        f"{path}: {type(a).__name__} became {type(b).__name__}"
    )
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for k in a:
            assert_same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, pd.DataFrame):
        assert list(a.columns) == list(b.columns), f"{path}: columns differ"
        assert a.shape == b.shape, f"{path}: shape differs"
        for c in a.columns:
            assert_same(a[c].to_numpy(), b[c].to_numpy(), f"{path}.{c}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length differs"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{path}[{i}]")
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape, f"{path}: shape differs"
        if a.dtype.kind == "f":
            assert np.allclose(a, b, equal_nan=True), f"{path}: values differ"
        else:
            assert (a == b).all(), f"{path}: values differ"
    elif isinstance(a, float) and np.isnan(a):
        assert np.isnan(b), f"{path}: {a} became {b}"
    else:
        assert a == b, f"{path}: {a!r} became {b!r}"


def structure():
    """A structure carrying every shape the real results use."""
    return {
        # Tuple keys, as the synthetic results are keyed by (b_true, n).
        (1.0, 100): {
            "metadata": {"b_true": 1.0, "n_events": 100, "label": "rounded"},
            "sf_mrange": np.linspace(0, 5, 11),
            "estimators": {"b_aki": {"sf_mean": np.arange(4.0)}},
        },
        (0.8, 50): {"metadata": {"b_true": 0.8}, "sf_mrange": np.zeros(3)},
        "zones": pd.DataFrame({
            "zone_id": ["GRAS250", "SIAS423"],
            "n": [38, 172],
            "ratio": [0.71, 1.29],
            "flagged": [True, False],
        }),
        "records": [
            {"imt": "PGA", "value": 1.5, "curve": np.array([1.0, 2.0])},
            {"imt": "SA(1.0)", "value": 2.5, "curve": np.array([3.0])},
        ],
        "return_periods": [475, 2475],
        "names": np.array(["alpha", "beta"]),
        "nothing": None,
        "count": 7,
        "ratio": 0.5,
    }


def test_round_trip_preserves_structure(tmp_path):
    original = structure()
    store.save(tmp_path / "s.zarr.zip", original)
    assert_same(original, store.load(tmp_path / "s.zarr.zip"))


def test_ragged_sequences_survive(tmp_path):
    """Members of a sequence need not share a shape."""
    original = {"curves": [np.arange(3.0), np.arange(7.0)]}
    store.save(tmp_path / "s.zarr.zip", original)
    assert_same(original, store.load(tmp_path / "s.zarr.zip"))


def test_nan_survives(tmp_path):
    original = {"x": np.array([1.0, np.nan]), "y": float("nan")}
    store.save(tmp_path / "s.zarr.zip", original)
    assert_same(original, store.load(tmp_path / "s.zarr.zip"))


def test_refuses_a_foreign_store(tmp_path):
    """A store written by something else must not be read as if it were ours."""
    import zarr
    from zarr.storage import ZipStore
    with ZipStore(tmp_path / "other.zarr.zip", mode="w") as other:
        zarr.open_group(other, mode="w")
    with pytest.raises(ValueError, match="bfree-store"):
        store.load(tmp_path / "other.zarr.zip")


def test_refuses_colliding_keys(tmp_path):
    """Two keys that map to one group name would silently lose data."""
    with pytest.raises(ValueError, match="both map to"):
        store.save(tmp_path / "s.zarr.zip", {(1.0, 2): "a", "1_2": "b"})


def test_text_is_stored_as_specified_utf8(tmp_path):
    """Fixed-width UTF-32 has no Zarr v3 spec, so text must not be stored as it."""
    import zarr
    from zarr.storage import ZipStore
    store.save(tmp_path / "s.zarr.zip", {"names": np.array(["a", "bb"])})
    with ZipStore(tmp_path / "s.zarr.zip", mode="r") as opened:
        node = zarr.open_group(opened, mode="r")["names"]
        assert "U" not in str(node.dtype), f"text stored as {node.dtype}"


def test_each_entry_is_written_once(tmp_path):
    """A Zip store appends where a directory overwrites.

    Setting an attribute on a group that already exists leaves a second copy of
    its metadata, and readers then depend on which copy they happen to take.
    """
    import zipfile
    from collections import Counter
    store.save(tmp_path / "s.zarr.zip", structure())
    names = zipfile.ZipFile(tmp_path / "s.zarr.zip").namelist()
    repeated = [n for n, count in Counter(names).items() if count > 1]
    assert not repeated, f"written more than once: {repeated}"
