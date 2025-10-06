from __future__ import annotations

from collections.abc import Mapping, Sequence

import treedata as td

from pycea.utils import get_leaves as _get_leaves
from pycea.utils import get_trees


def leaves(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> list[str] | Mapping[str, list[str]]:
    """
    Get the leaves of a tree.

    Parameters
    ----------
    tdata
        The ``treedata.TreeData`` object.
    tree
        Optional tree key or sequence of keys. If ``None`` (default),
        leaves for all trees with nodes are returned.

    Returns
    -------
    leaves - DFS postorder list of leaves. List if a single tree is requested,
        mapping from tree key to list of leaves if multiple trees are requested.
    """
    trees = get_trees(tdata, tree)
    leaves_map = {name: _get_leaves(t) for name, t in trees.items()}
    if len(leaves_map) == 1:
        return next(iter(leaves_map.values()))
    return leaves_map
