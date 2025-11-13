from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import _check_tree_overlap, get_keyed_node_data, get_keyed_obs_data, get_root, get_trees


def _most_common(arr: np.ndarray) -> Any:
    """Finds the most common element in a list."""
    unique_values, counts = np.unique(arr, return_counts=True)
    most_common_index = np.argmax(counts)
    return unique_values[most_common_index]


def _get_node_value(tree: nx.DiGraph, node: str, key: str, index: int | None) -> Any:
    """Gets the value of a node attribute."""
    if key in tree.nodes[node]:
        if index is not None:
            return tree.nodes[node][key][index]
        else:
            return tree.nodes[node][key]
    else:
        return None


def _set_node_value(tree: nx.DiGraph, node: str, key: str, value: Any, index: int | None) -> None:
    """Sets the value of a node attribute."""
    if index is not None:
        tree.nodes[node][key][index] = value
    else:
        tree.nodes[node][key] = value


def _remove_node_attributes(tree: nx.DiGraph, key: str) -> None:
    """Removes a node attribute from all nodes in the tree."""
    for node in tree.nodes:
        if key in tree.nodes[node]:
            del tree.nodes[node][key]


def _reconstruct_fitch_hartigan(
    tree: nx.DiGraph, key: str, missing: str | None = None, index: int | None = None
) -> None:
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


def _reconstruct_sankoff(
    tree: nx.DiGraph,
    key: str,
    costs: pd.DataFrame,
    missing: str | None = None,
    default: str | None = None,
    index: int | None = None,
) -> None:
    """Reconstructs ancestral states using the Sankoff algorithm."""
    # Set up
    alphabet = list(costs.index)
    num_states = len(alphabet)
    cost_matrix = costs.to_numpy()
    value_to_index = {value: i for i, value in enumerate(alphabet)}

    # Recursive function to calculate the Sankoff scores
    def sankoff_scores(node):
        # Base case: leaf
        if tree.out_degree(node) == 0:
            leaf_value = _get_node_value(tree, node, key, index)
            if leaf_value == missing:
                return np.zeros(num_states)
            else:
                scores = np.full(num_states, float("inf"))
                scores[value_to_index[leaf_value]] = 0
                return scores
        # Recursive case: internal node
        else:
            scores = np.zeros(num_states)
            pointers = np.zeros((num_states, len(list(tree.successors(node)))), dtype=int)
            for i, child in enumerate(tree.successors(node)):
                child_scores = sankoff_scores(child)
                for j in range(num_states):
                    costs_with_child = child_scores + cost_matrix[j, :]
                    min_cost_index = np.argmin(costs_with_child)
                    scores[j] += costs_with_child[min_cost_index]
                    pointers[j, i] = min_cost_index
            tree.nodes[node]["_pointers"] = pointers
            return scores

    # Recursive function to traceback the Sankoff scores
    def traceback(node, parent_value_index):
        for i, child in enumerate(tree.successors(node)):
            child_value_index = tree.nodes[node]["_pointers"][parent_value_index, i]
            _set_node_value(tree, child, key, alphabet[child_value_index], index)
            traceback(child, child_value_index)

    # Get scores
    root = [n for n, d in tree.in_degree() if d == 0][0]
    root_scores = sankoff_scores(root)
    # Reconstruct ancestral states
    root_value_index = np.argmin(root_scores)
    _set_node_value(tree, root, key, alphabet[root_value_index], index)
    traceback(root, root_value_index)
    # Clean up
    for node in tree.nodes:
        if "_pointers" in tree.nodes[node]:
            del tree.nodes[node]["_pointers"]


def _reconstruct_mean(tree: nx.DiGraph, key: str, index: int | None) -> None:
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


def _reconstruct_list(tree: nx.DiGraph, key: str, sum_func: Callable, index: int | None) -> None:
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


def _ancestral_states(
    tree: nx.DiGraph,
    key: str,
    method: str | Callable = "mean",
    costs: pd.DataFrame | None = None,
    missing: str | None = None,
    default: str | None = None,
    index: int | None = None,
) -> None:
    """Reconstructs ancestral states for a given attribute using a given method"""
    if method == "sankoff":
        if costs is None:
            raise ValueError("Costs matrix must be provided for Sankoff algorithm.")
        _reconstruct_sankoff(tree, key, costs, missing, default, index)
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


@overload
def ancestral_states(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    method: str | Callable = "mean",
    missing_state: str | None = None,
    default_state: str | None = None,
    costs: pd.DataFrame | None = None,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...
@overload
def ancestral_states(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    method: str | Callable = "mean",
    missing_state: str | None = None,
    default_state: str | None = None,
    costs: pd.DataFrame | None = None,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None: ...
def ancestral_states(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    method: str | Callable = "mean",
    missing_state: str | None = None,
    default_state: str | None = None,
    costs: pd.DataFrame | None = None,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> pd.DataFrame | None:
    """Reconstructs ancestral states for an attribute.

    This function reconstructs ancestral (internal node) states for categorical or
    continuous attributes defined on tree leaves. Several reconstruction methods
    are supported, ranging from simple aggregation rules to the Sankoff and Fitch-Hartigan
    algorithms for discrete character data, or a custom aggregation function can be provided.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs.keys()`, `var_names`, `obsm.keys()`, or `obsp.keys()` to reconstruct.
    method
        Method to reconstruct ancestral states:

        * 'mean' : The mean of leaves in subtree.
        * 'mode' : The most common value in the subtree.
        * 'fitch_hartigan' : The Fitch-Hartigan algorithm.
        * 'sankoff' : The Sankoff algorithm with specified costs.
        * Any function that takes a list of values and returns a single value.
    missing_state
        The state to consider as missing data.
    default_state
        The expected state for the root node.
    costs
        A pd.DataFrame with the costs of changing states (from rows to columns). Only used if method is 'sankoff'.
    keys_added
        Attribute keys of `tdata.obst[tree].nodes` where ancestral states will be stored. If `None`, `keys` are used.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`DataFrame <pandas.DataFrame>` with ancestral states.

    Returns
    -------
    Returns `None` if `copy=False`, else return :class:`DataFrame <pandas.DataFrame>` with ancestral states.

    Sets the following fields for each key:

    * `tdata.obst[tree].nodes[key_added]` : `float` | `Object` | `List[Object]`
        - Inferred ancestral states. List of states if data was an array.

    Examples
    --------
    Infer the expression of Krt20 and Cd74 based on their mean value in descendant cells:

    >>> tdata = py.datasets.yang22()
    >>> py.tl.ancestral_states(tdata, keys=["Krt20", "Cd74"], method="mean")

    Reconstruct ancestral character states using the Fitch-Hartigan algorithm:

    >>> py.tl.ancestral_states(tdata, keys="characters", method="fitch_hartigan", missing_state=-1)

    """
    if isinstance(keys, str):
        keys = [keys]
    if keys_added is None:
        keys_added = keys
    if isinstance(keys_added, str):
        keys_added = [keys_added]
    if len(keys) != len(keys_added):
        raise ValueError("Length of keys must match length of keys_added.")
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    for _, t in trees.items():
        data, is_array, is_square = get_keyed_obs_data(tdata, keys)
        dtypes = {dtype.kind for dtype in data.dtypes}
        # Check data type
        if dtypes.intersection({"f"}):
            if method in ["fitch_hartigan", "sankoff"]:
                raise ValueError(f"Method {method} requires categorical data.")
        if dtypes.intersection({"O", "S"}):
            if method in ["mean"]:
                raise ValueError(f"Method {method} requires numeric data.")
        # If array add to tree as list
        if is_array:
            length = data.shape[1]
            node_attrs = data.apply(lambda row: list(row), axis=1).to_dict()
            for node in t.nodes:
                if node not in node_attrs:
                    node_attrs[node] = [None] * length
            _remove_node_attributes(t, keys_added[0])
            nx.set_node_attributes(t, node_attrs, keys_added[0])
            for index in range(length):
                _ancestral_states(t, keys_added[0], method, costs, missing_state, default_state, index)
        # If column add to tree as scalar
        else:
            for key, key_added in zip(keys, keys_added, strict=False):
                _remove_node_attributes(t, key_added)
                nx.set_node_attributes(t, data[key].to_dict(), key_added)
                _ancestral_states(t, key_added, method, costs, missing_state, default_state)
    if copy:
        return get_keyed_node_data(tdata, keys_added, tree_keys, slot="obst")
        return get_keyed_node_data(tdata, keys_added, tree_keys)
