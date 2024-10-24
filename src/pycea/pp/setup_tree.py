from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd
import treedata as td

from pycea.utils import get_keyed_leaf_data, get_keyed_node_data, get_root, get_trees


def _add_depth(tree, depth_key):
    """Adds a depth attribute to the nodes of a tree."""
    root = get_root(tree)
    depths = nx.single_source_shortest_path_length(tree, root)
    nx.set_node_attributes(tree, depths, depth_key)


def _check_tree_overlap(tdata, tree_keys):
    """Check single tree is requested when allow_overlap is True"""
    if tree_keys is None:
        tree_keys = tdata.obst.keys()
        if tdata.allow_overlap and len(tree_keys) > 1:
            raise ValueError("Must specify a tree when tdata.allow_overlap is True.")
    elif isinstance(tree_keys, str):
        pass
    elif isinstance(tree_keys, Sequence):
        if tdata.allow_overlap:
            raise ValueError("Cannot request multiple trees when tdata.allow_overlap is True.")
    else:
        raise ValueError("Tree keys must be a string, list of strings, or None.")


def add_depth(
    tdata: td.TreeData, key_added: str = "depth", tree: str | Sequence[str] | None = None, copy: bool = False
) -> None | pd.DataFrame:
    """Adds a depth attribute to the tree.

    Parameters
    ----------
    tdata
        TreeData object.
    key_added
        Key to store node depths.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`DataFrame <pandas.DataFrame>` with node depths.

    Returns
    -------
    Returns `None` if `copy=False`, else returns node depths.

    Sets the following fields:

    * `tdata.obs[key_added]` : :class:`Series <pandas.Series>` (dtype `float`)
        - Distance from the root node.
    * `tdata.obst[tree].nodes[key_added]` : `float`
        - Distance from the root node.
    """
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    for _, tree in trees.items():
        _add_depth(tree, key_added)
    tdata.obs[key_added] = get_keyed_leaf_data(tdata, key_added, tree_keys)[key_added]
    if copy:
        return get_keyed_node_data(tdata, key_added, tree_keys)
