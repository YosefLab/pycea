from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import get_keyed_node_data, get_keyed_obs_data, get_trees


def _most_common(arr):
    """Finds the most common element in a list."""
    unique_values, counts = np.unique(arr, return_counts=True)
    most_common_index = np.argmax(counts)
    return unique_values[most_common_index]


def _ancestral_states(tree, key, method="mean"):
    """Finds the ancestral state of a node in a tree."""
    # Get summation function
    if method == "mean":
        sum_func = np.mean
    elif method == "median":
        sum_func = np.median
    elif method == "mode":
        sum_func = _most_common
    else:
        raise ValueError(f"Method {method} not recognized.")
    # Get aggregation function
    if method in ["mean", "median", "mode"]:
        agg_func = np.concatenate
    # infer ancestral states
    for node in nx.dfs_postorder_nodes(tree):
        if tree.out_degree(node) == 0:
            tree.nodes[node]["_message"] = np.array([tree.nodes[node][key]])
        else:
            subtree_values = agg_func([tree.nodes[child]["_message"] for child in tree.successors(node)])
            tree.nodes[node]["_message"] = subtree_values
            tree.nodes[node][key] = sum_func(subtree_values)
    # remove messages
    for node in tree.nodes:
        del tree.nodes[node]["_message"]


def ancestral_states(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    method: str = "mean",
    tree: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None:
    """Reconstructs ancestral states for an attribute.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to reconstruct.
    method
        Method to reconstruct ancestral states. One of "mean", "median", or "mode".
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a pd.DataFrame with ancestral states.
    """
    if isinstance(keys, str):
        keys = [keys]
    tree_keys = tree
    trees = get_trees(tdata, tree_keys)
    for _, tree in trees.items():
        data, _ = get_keyed_obs_data(tdata, keys)
        for key in keys:
            nx.set_node_attributes(tree, data[key].to_dict(), key)
            _ancestral_states(tree, key, method)
    if copy:
        states = []
        for name, tree in trees.items():
            tree_states = []
            for key in keys:
                data = get_keyed_node_data(tree, key)
                tree_states.append(data)
            tree_states = pd.concat(tree_states, axis=1)
            tree_states["tree"] = name
            states.append(tree_states)
        states = pd.concat(states)
        states["node"] = states.index
        return states.reset_index(drop=True)
