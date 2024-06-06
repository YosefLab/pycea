from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import pandas as pd
import treedata as td

from pycea.utils import get_root, get_trees


def _nodes_at_depth(tree, parent, nodes, depth, depth_key):
    """Recursively finds nodes at a given depth."""
    if tree.nodes[parent][depth_key] >= depth:
        nodes.append(parent)
    else:
        for child in tree.successors(parent):
            _nodes_at_depth(tree, child, nodes, depth, depth_key)
    return nodes


def _clade_name_generator():
    """Generates clade names."""
    i = 0
    while True:
        yield str(i)
        i += 1


def _clades(tree, depth, depth_key, clades, clade_key, name_generator):
    """Marks clades in a tree."""
    # Check that root has depth key
    root = get_root(tree)
    if depth_key not in tree.nodes[root]:
        raise ValueError(
            f"Tree does not have {depth_key} attribute. You can run `pycea.pp.add_depth` to add depth attribute."
        )
    if (depth is not None) and (clades is None):
        nodes = _nodes_at_depth(tree, root, [], depth, depth_key)
        clades = dict(zip(nodes, name_generator))
    elif (clades is not None) and (depth is None):
        pass
    else:
        raise ValueError("Must specify either clades or depth.")
    leaf_to_clade = {}
    for node, clade in clades.items():
        # Leaf
        if tree.out_degree(node) == 0:
            leaf_to_clade[node] = clade
            tree.nodes[node][clade_key] = clade
        # Internal node
        for u, v in nx.dfs_edges(tree, node):
            tree.nodes[u][clade_key] = clade
            tree.edges[u, v][clade_key] = clade
            if tree.out_degree(v) == 0:
                leaf_to_clade[v] = clade
                tree.nodes[v][clade_key] = clade
    return clades, leaf_to_clade


def clades(
    tdata: td.TreeData,
    depth: int | float = None,
    depth_key: str = "depth",
    clades: str | Sequence[str] = None,
    key_added: str = "clade",
    tree: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None | Mapping:
    """Marks clades in a tree.

    Parameters
    ----------
    tdata
        The TreeData object.
    depth
        Depth to cut tree at. Must be specified if clades is None.
    depth_key
        Key where depth is stored.
    clades
        A dictionary mapping nodes to clades.
    key_added
        Key to store clades in.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`pandas.DataFrame` with clades.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`pandas.DataFrame`. Sets the following fields:

    `tdata.obs[key_added]` : :class:`pandas.Series` (dtype `Object`)
        Clade.
    `tdata.obst[tree].nodes[key_added]` : `Object`
        Clade.

    """
    # Setup
    tree_keys = tree
    trees = get_trees(tdata, tree_keys)
    if clades and len(trees) > 1:
        raise ValueError("Multiple trees are present. Must specify a single tree if clades are given.")
    # Identify clades
    name_generator = _clade_name_generator()
    leaf_to_clade = {}
    clade_nodes = []
    for key, tree in trees.items():
        tree_nodes, tree_leaves = _clades(tree, depth, depth_key, clades, key_added, name_generator)
        tree_nodes = pd.DataFrame(tree_nodes.items(), columns=["node", key_added])
        tree_nodes["tree"] = key
        clade_nodes.append(tree_nodes)
        leaf_to_clade.update(tree_leaves)
    # Update TreeData and return
    tdata.obs[key_added] = tdata.obs.index.map(leaf_to_clade)
    clade_nodes = pd.concat(clade_nodes)
    if copy:
        return clade_nodes
