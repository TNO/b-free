"""
Reading and writing the analysis results.

The results used to be pickles. Unpickling executes whatever the file asks it
to, which makes an archived pickle a poor thing to hand to a stranger, and it
ties the data to one language and one set of class definitions. These are plain
numeric results and deserve a format that says what it holds.

Zarr is used because the raw catalogue store already is: it is self-describing,
readable from C, C++, Java, Julia, R and the browser as well as Python, and it
carries no executable content.

Each result is one Zip store, so it is a single file rather than the thousands
that the same structure needs as a directory. That is what makes it something
to deposit and to copy; `unzip` turns it back into the directory form if a
reader's Zarr implementation wants that. The raw catalogue store stays a
directory, because reduce_simulations.py reads it from several processes at once
and a Zip store is not safe to share that way.

The mapping is deliberately small:

    dict              -> group
    numpy array       -> array
    list or tuple     -> group of numbered members, marked as a sequence
    DataFrame         -> group of one array per column, marked as a table
    scalar, str, None -> JSON attribute on the parent group

Text arrays use Zarr's variable-length UTF-8 type rather than NumPy's
fixed-width one, because fixed-width UTF-32 has no Zarr v3 specification and
other implementations are not obliged to read it.

Dictionary keys that are not strings - the results are keyed by ``(b, n)``
tuples - are recorded in a ``_keys`` attribute mapping each group name back to
the original key, so the round trip is exact without making the names
unreadable.
"""

import json

import numpy as np
import zarr
from zarr.storage import ZipStore

# Bumped if the mapping above changes such that older files no longer satisfy it.
FORMAT = "bfree-store-1"

_KIND = "_kind"
_KEYS = "_keys"
_COLUMNS = "_columns"
_LENGTH = "_length"
_RESERVED = {_KIND, _KEYS, _COLUMNS, _LENGTH, "format"}


def _is_scalar(value):
    return (value is None
            or isinstance(value, str | bool | int | float | np.number | np.bool_))


def _is_table(value):
    """A pandas DataFrame, without importing pandas to find out."""
    return hasattr(value, "columns") and hasattr(value, "to_numpy")


def _name_for(key, index):
    """A readable group name for a dictionary key."""
    if isinstance(key, str):
        return key
    if isinstance(key, tuple):
        return "_".join(f"{p:g}" if isinstance(p, float) else str(p) for p in key)
    if _is_scalar(key):
        return f"{key:g}" if isinstance(key, float) else str(key)
    return f"item{index}"


def _write_array(group, name, array):
    """Store one array, using variable-length UTF-8 for text."""
    text = array.dtype.kind in "USO"
    if text:
        array = array.astype(str)
    node = group.create_array(name, shape=array.shape,
                              dtype=str if text else array.dtype,
                              chunks=array.shape or (1,))
    node[...] = array


def _plan(obj):
    """Split *obj* into the attributes its group carries and the members to write.

    The attributes have to be known before the group exists: a Zip store appends
    where a directory would overwrite, so setting an attribute on a group that
    has already been created leaves two copies of its metadata in the file.
    """
    if _is_table(obj):
        names = [str(c) for c in obj.columns]
        attrs = {_KIND: "table", _COLUMNS: names}
        raw = [(name, obj[column].to_numpy())
               for name, column in zip(names, obj.columns, strict=True)]
    elif isinstance(obj, list | tuple):
        attrs = {_KIND: "tuple" if isinstance(obj, tuple) else "sequence",
                 _LENGTH: len(obj)}
        raw = [(f"i{i}", item) for i, item in enumerate(obj)]
    elif isinstance(obj, dict):
        attrs = {_KIND: "mapping"}
        raw, keys = [], {}
        for i, (key, value) in enumerate(obj.items()):
            name = _name_for(key, i)
            if name in keys:
                raise ValueError(
                    f"keys {keys[name]!r} and {key!r} both map to {name!r}")
            keys[name] = key
            raw.append((name, value))
        if any(not isinstance(k, str) for k in obj):
            attrs[_KEYS] = json.dumps(keys, default=list)
    else:
        raise TypeError(f"cannot store a {type(obj).__name__}")

    members = []
    for name, value in raw:
        if _is_scalar(value):
            attrs[name] = value.item() if isinstance(value, np.generic) else value
        else:
            members.append((name, value))
    return attrs, members


def _write_members(group, members):
    for name, value in members:
        if isinstance(value, dict | list | tuple) or _is_table(value):
            attrs, children = _plan(value)
            _write_members(group.create_group(name, attributes=attrs), children)
        else:
            _write_array(group, name, np.asarray(value))


def save(path, obj):
    """Write a nested result structure to the Zarr store at *path*."""
    attrs, members = _plan(obj)
    with ZipStore(path, mode="w") as store:
        root = zarr.open_group(store, mode="w",
                               attributes={**attrs, "format": FORMAT})
        _write_members(root, members)
    return path


def _read_array(node):
    values = node[...]
    if values.dtype.kind in "biufc":
        return values
    # StringDType will not cast directly; go via Python strings.
    return np.array(values.tolist(), dtype=np.str_).reshape(values.shape)


def _member(name, arrays, subgroups, attrs):
    if name in arrays:
        return _read_array(arrays[name])
    if name in subgroups:
        return _read_group(subgroups[name])
    return attrs[name]


def _read_group(group):
    kind = group.attrs.get(_KIND, "mapping")
    arrays = dict(group.arrays())
    subgroups = dict(group.groups())
    attrs = {k: v for k, v in group.attrs.items() if k not in _RESERVED}

    if kind == "table":
        import pandas as pd
        return pd.DataFrame({c: _member(c, arrays, subgroups, attrs)
                             for c in group.attrs[_COLUMNS]})

    if kind in ("sequence", "tuple"):
        items = [_member(f"i{i}", arrays, subgroups, attrs)
                 for i in range(group.attrs[_LENGTH])]
        return tuple(items) if kind == "tuple" else items

    out = {name: _member(name, arrays, subgroups, attrs)
           for name in list(arrays) + list(subgroups) + list(attrs)}

    keys = group.attrs.get(_KEYS)
    if keys:
        original = json.loads(keys)
        out = {(tuple(original[n]) if isinstance(original.get(n), list)
                else original.get(n, n)): v
               for n, v in out.items()}
    return out


def load(path):
    """Read back a structure written by `save`."""
    with ZipStore(path, mode="r") as store:
        root = zarr.open_group(store, mode="r")
        stored = root.attrs.get("format")
        if stored != FORMAT:
            raise ValueError(
                f"{path} is not a {FORMAT} store (found {stored!r}). "
                "Regenerate it, or fetch the current version.")
        return _read_group(root)
