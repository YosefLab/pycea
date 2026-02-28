from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import networkx as nx
import pandas as pd
import treedata as td

from pycea.utils import (
    _check_tree_overlap,
    check_tree_has_key,
    get_keyed_leaf_data,
    get_keyed_node_data,
    get_root,
    get_trees,
)

from ._utils import _check_colors_length, _remove_attribute


def _nodes_at_depth(tree, parent, nodes, depth, depth_key):
    """Recursively finds nodes at a given depth."""
    if tree.nodes[parent][depth_key] >= depth:
        nodes.append(parent)
    else:
        for child in tree.successors(parent):
            _nodes_at_depth(tree, child, nodes, depth, depth_key)
    return nodes


def _clade_name_generator(dtype: type | str):
    """Generates clade names."""
    valid_dtypes = {"str": str, "int": int, "float": float, str: str, int: int, float: float}
    if dtype not in valid_dtypes:
        raise ValueError("dtype must be one of str, int, or float")
    converter = valid_dtypes[dtype]
    i = 0
    while True:
        yield converter(i)
        i += 1


def _clades(tree, depth, depth_key, clades, clade_key, name_generator, update):
    """Marks clades in a tree."""
    # Check that root has depth key
    root = get_root(tree)
    if (depth is not None) and (clades is None):
        check_tree_has_key(tree, depth_key)
        nodes = _nodes_at_depth(tree, root, [], depth, depth_key)
        clades = dict(zip(nodes, name_generator, strict=False))
    elif (clades is not None) and (depth is None):
        pass
    else:
        raise ValueError("Must specify either clades or depth.")
    for node, clade in clades.items():
        # Leaf
        if tree.out_degree(node) == 0:
            tree.nodes[node][clade_key] = clade
        # Internal node
        for u, v in nx.dfs_edges(tree, node):
            tree.nodes[u][clade_key] = clade
            tree.edges[u, v][clade_key] = clade
            if tree.out_degree(v) == 0:
                tree.nodes[v][clade_key] = clade
    return clades


@overload
def clades(
    tdata: td.TreeData,
    depth: float | None = None,
    depth_key: str = "depth",
    clades: Mapping[Any, Any] | None = None,
    key_added: str = "clade",
    update: bool = False,
    dtype: type | str = str,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...
@overload
def clades(
    tdata: td.TreeData,
    depth: float | None = None,
    depth_key: str = "depth",
    clades: Mapping[Any, Any] | None = None,
    key_added: str = "clade",
    update: bool = False,
    dtype: type | str = str,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None: ...
def clades(
    tdata: td.TreeData,
    depth: float | None = None,
    depth_key: str = "depth",
    clades: Mapping[Any, Any] | None = None,
    key_added: str = "clade",
    update: bool = False,
    dtype: type | str = str,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None | pd.DataFrame:
    """Marks clades in a tree.

    A clade is defined by a **ancestral node**; all nodes and
    edges in the ancestral node's descendant subtree inherit the same clade label.
    You can specify clades in two ways:

    * **Depth-based:**
       Given a ``depth`` threshold, all nodes that are extant
       at that depth are considered ancestral nodes. Each ancestral node and its descendants
       are assigned a unique clade label.

    * **Explicit mapping:**
       When ``clades``, a dictionary mapping nodes to clade labels is provided,
       those nodes are considered ancestral nodes. Each such node and its descendants are assigned
       the corresponding clade label.

    Parameters
    ----------
    tdata
        The TreeData object.
    depth
        Depth to cut tree at. Must be specified if clades is None.
    depth_key
        Attribute of `tdata.obst[tree].nodes` where depth is stored.
    clades
        A dictionary mapping nodes to clades.
    key_added
        Key to store clades in.
    update
        If True, updates existing clades instead of overwriting.
    dtype
        Data type of clade names. One of `str`, `int`, or `float`.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`DataFrame <pandas.DataFrame>` with clades.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`DataFrame <pandas.DataFrame>`.

    Sets the following fields:

    * `tdata.obs[key_added]` : :class:`Series <pandas.Series>` (dtype `Object`)
        - Clade assignment for each observation.
    * `tdata.obst[tree].nodes[key_added]` : `Object`
        - Clade assignment for each node.

    Modifies the following fields:

    * `tdata.uns[f"{key_added}_colors"]` : `List`
        - Removed if its length does not match the number of unique clades.

    Examples
    --------
    Mark clades at specified depth

    >>> tdata = pycea.datasets.koblan25()
    >>> pycea.tl.clades(tdata, depth=4, depth_key="time")

    Highlight descendants of 'node6'

    >>> pycea.tl.clades(tdata, clades={"node6": "node6_descendants"}, key_added="highlight")
    """
    # Setup
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    if clades and len(trees) > 1:
        raise ValueError("Multiple trees are present. Must specify a single tree if clades are given.")
    # Identify clades
    name_generator = _clade_name_generator(dtype=dtype)
    lcas = []
    for key, t in trees.items():
        if not update:
            _remove_attribute(t, key_added)
        tree_lcas = _clades(t, depth, depth_key, clades, key_added, name_generator, update)
        tree_lcas = pd.DataFrame(tree_lcas.items(), columns=["node", key_added])
        tree_lcas["tree"] = key
        lcas.append(tree_lcas)
    # Update TreeData and return
    if tdata.alignment == "leaves":
        node_to_clade = get_keyed_leaf_data(tdata, key_added, tree_keys)
    else:
        node_to_clade = get_keyed_node_data(tdata, key_added, tree_keys, slot="obst")
        node_to_clade.index = node_to_clade.index.droplevel(0)
    tdata.obs[key_added] = tdata.obs.index.map(node_to_clade[key_added])
    _check_colors_length(tdata, key_added)
    if copy:
        return pd.concat(lcas)
