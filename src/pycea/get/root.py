from __future__ import annotations

from collections.abc import Mapping, Sequence

import treedata as td

from pycea.utils import get_root as _get_root
from pycea.utils import get_trees


def root(tdata: td.TreeData, tree: str | Sequence[str] | None = None) -> str | Mapping[str, str | None]:
    """Get the root node(s) of tree(s) in ``tdata``.

    Parameters
    ----------
    tdata
        The ``treedata.TreeData`` object containing tree(s).
    tree
        Optional tree key or sequence of keys. If ``None`` (default),
        roots for all trees with nodes are returned.

    Returns
    -------
    str or Mapping[str, str | None]
        Root node for a single tree, or a mapping from tree key to root
        node when multiple trees are requested.
    """
    trees = get_trees(tdata, tree)
    roots = {name: _get_root(t) for name, t in trees.items()}
    if len(roots) == 1:
        return next(iter(roots.values()))
    return roots
