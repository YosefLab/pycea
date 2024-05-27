from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import get_keyed_node_data, get_keyed_obs_data, get_root, get_trees


def _most_common(arr):
    """Finds the most common element in a list."""
    unique_values, counts = np.unique(arr, return_counts=True)
    most_common_index = np.argmax(counts)
    return unique_values[most_common_index]


def _get_node_value(tree, node, key, index):
    """Gets the value of a node attribute."""
    if key in tree.nodes[node]:
        if index is not None:
            return tree.nodes[node][key][index]
        else:
            return tree.nodes[node][key]
    else:
        return None


def _set_node_value(tree, node, key, value, index):
    """Sets the value of a node attribute."""
    if index is not None:
        tree.nodes[node][key][index] = value
    else:
        tree.nodes[node][key] = value


def _reconstruct_fitch_hartigan(tree, key, missing="-1", index=None):
    """Reconstructs ancestral states using the Fitch-Hartigan algorithm."""

    # Recursive function to calculate the downpass
    def downpass(node):
        # Base case: leaf
        if tree.out_degree(node) == 0:
            value = _get_node_value(tree, node, key, index)
            if value == missing:
                tree.nodes[node]["value_set"] = missing
            else:
                tree.nodes[node]["value_set"] = {value}
        # Recursive case: internal node
        else:
            value_sets = []
            for child in tree.successors(node):
                downpass(child)
                value_set = tree.nodes[child]["value_set"]
                if value_set != missing:
                    value_sets.append(value_set)
            if len(value_sets) > 0:
                intersection = set.intersection(*value_sets)
                if intersection:
                    tree.nodes[node]["value_set"] = intersection
                else:
                    tree.nodes[node]["value_set"] = set.union(*value_sets)
            else:
                tree.nodes[node]["value_set"] = missing

    # Recursive function to calculate the uppass
    def uppass(node, parent_state=None):
        value = _get_node_value(tree, node, key, index)
        if value is None:
            if parent_state and parent_state in tree.nodes[node]["value_set"]:
                value = parent_state
            else:
                value = min(tree.nodes[node]["value_set"])
            _set_node_value(tree, node, key, value, index)
        elif value == missing:
            value = parent_state
            _set_node_value(tree, node, key, value, index)
        for child in tree.successors(node):
            uppass(child, value)

    # Run the algorithm
    root = get_root(tree)
    downpass(root)
    uppass(root)
    # Clean up
    for node in tree.nodes:
        if "value_set" in tree.nodes[node]:
            del tree.nodes[node]["value_set"]


def _reconstruct_sankoff(tree, key, costs, missing="-1", index=None):
    """Reconstructs ancestral states using the Sankoff algorithm."""

    # Recursive function to calculate the Sankoff scores
    def sankoff_scores(node):
        # Base case: leaf
        if tree.out_degree(node) == 0:
            leaf_value = _get_node_value(tree, node, key, index)
            if leaf_value == missing:
                return {value: 0 for value in alphabet}
            else:
                return {value: 0 if value == leaf_value else float("inf") for value in alphabet}
        # Recursive case: internal node
        else:
            scores = {value: 0 for value in alphabet}
            pointers = {value: {} for value in alphabet}
            for child in tree.successors(node):
                child_scores = sankoff_scores(child)
                for value in alphabet:
                    min_cost, min_value = float("inf"), None
                    for child_value in alphabet:
                        cost = child_scores[child_value] + costs.loc[value, child_value]
                        if cost < min_cost:
                            min_cost, min_value = cost, child_value
                    scores[value] += min_cost
                    pointers[value][child] = min_value
            tree.nodes[node]["_pointers"] = pointers
            return scores

    # Recursive function to traceback the Sankoff scores
    def traceback(node, parent_value=None):
        for child in tree.successors(node):
            child_value = tree.nodes[node]["_pointers"][parent_value][child]
            _set_node_value(tree, child, key, child_value, index)
            traceback(child, child_value)

    # Get scores
    root = get_root(tree)
    alphabet = set(costs.index)
    root_scores = sankoff_scores(root)
    # Reconstruct ancestral states
    root_value = min(root_scores, key=root_scores.get)
    _set_node_value(tree, root, key, root_value, index)
    traceback(root, root_value)
    # Clean up
    for node in tree.nodes:
        if "_pointers" in tree.nodes[node]:
            del tree.nodes[node]["_pointers"]


def _reconstruct_mean(tree, key, index):
    """Reconstructs ancestral by averaging the values of the children."""

    def subtree_mean(node):
        if tree.out_degree(node) == 0:
            return _get_node_value(tree, node, key, index), 1
        else:
            values, weights = [], []
            for child in tree.successors(node):
                child_value, child_n = subtree_mean(child)
                values.append(child_value)
                weights.append(child_n)
            mean_value = np.average(values, weights=weights)
            _set_node_value(tree, node, key, mean_value, index)
            return mean_value, sum(weights)

    root = get_root(tree)
    subtree_mean(root)


def _reconstruct_list(tree, key, sum_func, index):
    """Reconstructs ancestral states by concatenating the values of the children."""

    def subtree_list(node):
        if tree.out_degree(node) == 0:
            return [_get_node_value(tree, node, key, index)]
        else:
            values = []
            for child in tree.successors(node):
                values.extend(subtree_list(child))
            _set_node_value(tree, node, key, sum_func(values), index)
            return values

    root = get_root(tree)
    subtree_list(root)


def _ancestral_states(tree, key, method="mean", costs=None, missing=None, default=None, index=None):
    """Reconstructs ancestral states for a given attribute using a given method"""
    if method == "sankoff":
        if costs is None:
            raise ValueError("Costs matrix must be provided for Sankoff algorithm.")
        _reconstruct_sankoff(tree, key, costs, missing, index)
    elif method == "fitch_hartigan":
        _reconstruct_fitch_hartigan(tree, key, missing, index)
    elif method == "mean":
        _reconstruct_mean(tree, key, index)
    elif method == "mode":
        _reconstruct_list(tree, key, _most_common, index)
    elif callable(method):
        _reconstruct_list(tree, key, method, index)
    else:
        raise ValueError(f"Method {method} not recognized.")


def ancestral_states(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    method: str = "mean",
    missing_state: str = "-1",
    default_state: str = "0",
    costs: pd.DataFrame = None,
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
        Method to reconstruct ancestral states. One of "mean", "mode", "fitch_hartigan", "sankoff",
         or any function that takes a list of values and returns a single value.
    missing_state
        The state to consider as missing data.
    default_state
        The expected state for the root node.
    costs
        A pd.DataFrame with the costs of changing states (from rows to columns).
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
        data, is_array = get_keyed_obs_data(tdata, keys)
        dtypes = {dtype.kind for dtype in data.dtypes}
        # Check data type
        if dtypes.intersection({"i", "f"}):
            if method in ["fitch_hartigan", "sankoff"]:
                raise ValueError(f"Method {method} requires categorical data.")
        if dtypes.intersection({"O", "S"}):
            if method in ["mean"]:
                raise ValueError(f"Method {method} requires numerical data.")
        # If array add to tree as list
        if is_array:
            length = data.shape[1]
            node_attrs = data.apply(lambda row: list(row), axis=1).to_dict()
            for node in tree.nodes:
                if node not in node_attrs:
                    node_attrs[node] = [None] * length
            nx.set_node_attributes(tree, node_attrs, keys[0])
            for index in range(length):
                _ancestral_states(tree, keys[0], method, costs, missing_state, default_state, index)
        # If column add to tree as scalar
        else:
            for key in keys:
                nx.set_node_attributes(tree, data[key].to_dict(), key)
                _ancestral_states(tree, key, method, missing_state, default_state)
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
