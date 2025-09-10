from __future__ import annotations

from collections.abc import Mapping, Sequence

import treedata as td

from pycea.utils import get_leaves as _get_leaves
from pycea.utils import get_trees


def leaves(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> list[str] | Mapping[str, list[str]]:
    """Get the leaf nodes of tree(s) in ``tdata`` in DFS post-order.

    Parameters
    ----------
    tdata
        The ``treedata.TreeData`` object containing tree(s).
    tree
        Optional tree key or sequence of keys. If ``None`` (default),
        leaves for all trees with nodes are returned.

    Returns
    -------
    list[str] or Mapping[str, list[str]]
        Ordered list of leaf nodes for a single tree, or a mapping from
        tree key to ordered leaf list when multiple trees are requested.
    """
    trees = get_trees(tdata, tree)
    leaves_map = {name: _get_leaves(t) for name, t in trees.items()}
    if len(leaves_map) == 1:
        return next(iter(leaves_map.values()))
    return leaves_map
