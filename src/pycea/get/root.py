from __future__ import annotations

from collections.abc import Mapping, Sequence

import treedata as td

from pycea.utils import get_root as _get_root
from pycea.utils import get_trees


def root(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> str | Mapping[str, str | None]:
    """
    Get the root of tree.

    Parameters
    ----------
    tdata
        The ``treedata.TreeData`` object.
    tree
        The `obst` key or keys of the trees.

    Returns
    -------
    root - str if a single tree is requested,
        mapping from tree key to root node if multiple trees are requested.
    """
    trees = get_trees(tdata, tree)
    roots = {name: _get_root(t) for name, t in trees.items()}
    if len(roots) == 1:
        return next(iter(roots.values()))  # type: ignore
    return roots
