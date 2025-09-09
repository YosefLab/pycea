from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import treedata as td

from pycea.utils import get_root, get_trees


def _sort_tree(tree: nx.DiGraph, key: str, reverse: bool = False) -> nx.DiGraph:
    for node in nx.dfs_postorder_nodes(tree, get_root(tree)):
        if tree.out_degree(node) > 1:
            # Store edge metadata before removing edges so we can reattach
            # edges with their attributes intact.  Using ``tree.successors``
            # directly returns an iterator which will be exhausted after
            # calling ``sorted`` so we first materialize the list of
            # children.
            children = list(tree.successors(node))
            try:
                sorted_children = sorted(
                    children, key=lambda x: tree.nodes[x][key], reverse=reverse
                )
            except KeyError as err:
                raise KeyError(
                    f"Node {next(iter(children))} does not have a {key} attribute.",
                    "You may need to call `ancestral_states` to infer internal node values",
                ) from err

            # Capture edge attributes prior to removal
            edge_data = {
                child: tree.get_edge_data(node, child, default={}) for child in children
            }

            # Remove existing edges and re-add them in the sorted order with
            # their associated metadata.
            tree.remove_edges_from((node, child) for child in children)
            tree.add_edges_from(
                (node, child, edge_data[child]) for child in sorted_children
            )
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
    for name, t in trees.items():
        tdata.obst[name] = _sort_tree(t.copy(), key, reverse)  # type: ignore
    return None
