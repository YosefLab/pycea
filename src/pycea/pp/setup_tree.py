from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import treedata as td

from pycea.utils import get_keyed_node_data, get_root, get_trees


def _add_depth(tree, depth_key):
    """Adds a depth attribute to the nodes of a tree."""
    root = get_root(tree)
    depths = nx.single_source_shortest_path_length(tree, root)
    nx.set_node_attributes(tree, depths, depth_key)


def add_depth(
    tdata: td.TreeData, depth_key: str = "depth", tree: str | Sequence[str] | None = None, copy: bool = False
):
    """Adds a depth attribute to the nodes of a tree.

    Parameters
    ----------
    tdata
        TreeData object.
    depth_key
        Node attribute key to store the depth.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a pd.DataFrame node depths.
    """
    tree_keys = tree
    trees = get_trees(tdata, tree_keys)
    for _, tree in trees.items():
        _add_depth(tree, depth_key)
    if copy:
        return get_keyed_node_data(tdata, depth_key)
