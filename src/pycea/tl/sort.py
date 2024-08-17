from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import treedata as td

from pycea.utils import get_root, get_trees


def _sort_tree(tree, key, reverse=False):
    for node in nx.dfs_postorder_nodes(tree, get_root(tree)):
        if tree.out_degree(node) > 1:
            try:
                sorted_children = sorted(tree.successors(node), key=lambda x: tree.nodes[x][key], reverse=reverse)
            except KeyError as err:
                raise KeyError(
                    f"Node {next(tree.successors(node))} does not have a {key} attribute.",
                    "You may need to call `ancestral_states` to infer internal node values",
                ) from err
            tree.remove_edges_from([(node, child) for child in tree.successors(node)])
            tree.add_edges_from([(node, child) for child in sorted_children])
    return tree


def sort(tdata: td.TreeData, key: str, reverse: bool = False, tree: str | Sequence[str] | None = None) -> None:
    """Reorders branches based on a node attribute.

    Parameters
    ----------
    tdata
        TreeData object.
    key
        Attribute of `tdata.obst[tree].nodes` to sort by.
    reverse
        If True, sort in descending order.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.

    Returns
    -------
    Returns `None` and does not set any fields.
    """
    trees = get_trees(tdata, tree)
    for name, tree in trees.items():
        tdata.obst[name] = _sort_tree(tree.copy(), key, reverse)
    return None
