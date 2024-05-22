from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import treedata as td

from pycea.utils import get_root


def _nodes_at_depth(tree, parent, nodes, depth, depth_key):
    """Recursively finds nodes at a given depth."""
    if tree.nodes[parent][depth_key] >= depth:
        nodes.append(parent)
    else:
        for child in tree.successors(parent):
            _nodes_at_depth(tree, child, nodes, depth, depth_key)
    return nodes


def clades(
    tdata: td.TreeData,
    key: str | Sequence[str] = None,
    depth: int | float = None,
    depth_key: str = "depth",
    clades: str | Sequence[str] = None,
    clade_key: str = "clade",
    copy: bool = False,
) -> None | Mapping:
    """Identifies clades in a tree.

    Parameters
    ----------
    tdata
        The TreeData object.
    key
        The `obst` key of the tree.
    depth
        Depth to cut tree at. Must be specified if clades is None.
    depth_key
        Key where depth is stored.
    clades
        A dictionary mapping nodes to clades.
    clade_key
        Key to store clades in.
    copy
        If True, returns a dictionary mapping nodes to clades.

    Returns
    -------
    None or Mapping
        If copy is True, returns a dictionary mapping nodes to clades.
    """
    # Get tree
    if not key:
        key = tdata.obs_keys()[0]
    tree = tdata.obst[key]
    # Get clades
    if (depth is not None) and (clades is None):
        nodes = _nodes_at_depth(tree, get_root(tree), [], depth, depth_key)
        clades = {node: str(clade) for clade, node in enumerate(nodes)}
    elif (clades is not None) and (depth is None):
        pass
    else:
        raise ValueError("Must specify either clades or depth.")
    # Set clades
    leaf_clades = {}
    for node, clade in clades.items():
        # Leaf
        if tree.out_degree(node) == 0:
            leaf_clades[node] = clade
            tree.nodes[node][clade_key] = clade
        # Internal node
        for u, v in nx.dfs_edges(tree, node):
            tree.nodes[u][clade_key] = clade
            tree.edges[u, v][clade_key] = clade
            if tree.out_degree(v) == 0:
                leaf_clades[v] = clade
                tree.nodes[v][clade_key] = clade
    tdata.obs[clade_key] = tdata.obs.index.map(leaf_clades)
    if copy:
        return clades
