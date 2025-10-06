from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import treedata as td

from pycea.utils import (
    _check_tree_overlap,
    check_tree_has_key,
    get_obs_to_tree_map,
    get_root,
    get_tree_to_obs_map,
    get_trees,
)

from ._metrics import _get_tree_metric, _TreeMetric
from ._utils import (
    _check_previous_params,
    _csr_data_mask,
    _format_keys,
    _set_distances_and_connectivities,
    _set_random_state,
)


def _tree_distance(tree, depth_key, metric, pairs=None):
    """Compute distances between pairs of nodes in a tree."""
    rows, cols, data = [], [], []
    check_tree_has_key(tree, depth_key)
    root = get_root(tree)
    lcas = dict(nx.tree_all_pairs_lowest_common_ancestor(tree, root=root, pairs=pairs))
    for (node1, node2), lca in lcas.items():
        rows.append(node1)
        cols.append(node2)
        data.append(metric(tree, depth_key, node1, node2, lca))
    return rows, cols, data


def _all_pairs_shared_tree(tdata, tree_keys, sample_n):
    """Get all pairs of observations that share a tree."""
    tree_to_obs = get_tree_to_obs_map(tdata, tree_keys)
    if sample_n is None:
        tree_pairs = {key: [(i, j) for i in nodes for j in nodes] for key, nodes in tree_to_obs.items()}
    else:
        tree_n_pairs = np.array([len(nodes) ** 2 for nodes in tree_to_obs.values()])
        tree_pairs = defaultdict(set)
        n_pairs = 0
        if sample_n > tree_n_pairs.sum():
            raise ValueError("Sample size is larger than the number of pairs.")
        k = 0
        while k < sample_n:
            tree_keys = list(tree_to_obs.keys())
            tree = random.choices(tree_keys, tree_n_pairs, k=1)[0]  # type: ignore
            i = random.choice(tree_to_obs[tree])
            j = random.choice(tree_to_obs[tree])
            if (i, j) not in tree_pairs[tree]:
                tree_pairs[tree].add((i, j))
                n_pairs += 1
            k += 1
        tree_pairs = {key: list(pairs) for key, pairs in tree_pairs.items()}
    return tree_pairs


def _assign_pairs_to_trees(pairs, tdata, tree_keys):
    """Assign pairs to trees."""
    obs_to_tree = get_obs_to_tree_map(tdata, tree_keys)
    has_tree = set(obs_to_tree.keys())
    tree_pairs = defaultdict(list)
    for i, j in pairs:
        if i in has_tree and j in has_tree and obs_to_tree[i] == obs_to_tree[j]:
            tree_pairs[obs_to_tree[i]].append((i, j))
    return tree_pairs


def _sample_pairs(tree_pairs, sample_n):
    """Given a dictionary of tree pairs, sample n pairs."""
    if sample_n is not None:
        pairs_to_tree = {pair: key for key, pairs in tree_pairs.items() for pair in pairs}
        if sample_n > len(pairs_to_tree):
            raise ValueError("Sample size is larger than the number of pairs.")
        sampled_pairs = random.sample(list(pairs_to_tree.keys()), sample_n)
        tree_pairs = {key: [pair for pair in pairs if pair in sampled_pairs] for key, pairs in tree_pairs.items()}
    return tree_pairs


def _convert_pair_distance_to_matrix(tdata, rows, cols, data):
    """Convert pairs to distance matrix."""
    dense = False
    rows = [tdata.obs_names.get_loc(row) for row in rows]
    cols = [tdata.obs_names.get_loc(col) for col in cols]
    distances = sp.sparse.csr_matrix((data, (rows, cols)), shape=(tdata.n_obs, tdata.n_obs))
    if len(data) == tdata.n_obs**2:
        distances = distances.toarray()
        dense = True
    return distances, dense


@overload
def tree_distance(
    tdata: td.TreeData,
    depth_key: str = "depth",
    obs: str | int | Sequence[Any] | None = None,
    metric: _TreeMetric = "path",
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    key_added: str | None = None,
    update: bool = True,
    tree: str | Sequence[Any] | None = None,
    copy: Literal[True, False] = True,
) -> sp.sparse.csr_matrix | np.ndarray: ...
@overload
def tree_distance(
    tdata: td.TreeData,
    depth_key: str = "depth",
    obs: str | int | Sequence[Any] | None = None,
    metric: _TreeMetric = "path",
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    key_added: str | None = None,
    update: bool = True,
    tree: str | Sequence[Any] | None = None,
    copy: Literal[True, False] = False,
) -> None: ...
def tree_distance(
    tdata: td.TreeData,
    depth_key: str = "depth",
    obs: str | int | Sequence[Any] | None = None,
    metric: _TreeMetric = "path",
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    key_added: str | None = None,
    update: bool = True,
    tree: str | Sequence[Any] | None = None,
    copy: Literal[True, False] = False,
) -> None | sp.sparse.csr_matrix | np.ndarray:
    """Computes tree distances between observations.

    Parameters
    ----------
    tdata
        The TreeData object.
    depth_key
        Attribute of `tdata.obst[tree].nodes` where depth is stored.
    obs
        The observations to use:

        - If `None`, pairwise distance for tree leaves is stored in `tdata.obsp`.
        - If a string, distance to all other tree leaves is `tdata.obs`.
        - If a sequence, pairwise distance is stored in `tdata.obsp`.
        - If a sequence of pairs, distance between pairs is stored in `tdata.obsp`.
    metric
        The type of tree distance to compute:

        - `'lca'`: lowest common ancestor depth.
        - `'path'`: abs(node1 depth + node2 depth - 2 * lca depth).
    sample_n
        If specified, randomly sample `sample_n` pairs of observations.
    connect_key
        If specified, compute distances only between connected observations specified by `tdata.obsp['{connect_key}_connectivities']`.
    random_state
        Random seed for sampling.
    key_added
        Distances are stored in `tdata.obsp['{key_added}_distances']` and
        connectivities in `tdata.obsp['{key_added}_connectivities']`. Defaults to 'tree'.
    update
        If True, updates existing distances instead of overwriting.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`ndarray <numpy.ndarray>` or :class:`csr_matrix <scipy.sparse.csr_matrix>` with distances.

    Returns
    -------
     Returns `None` if `copy=False`, else returns :class:`ndarray <numpy.ndarray>`/:class:`csr_matrix <scipy.sparse.csr_matrix>`.

    Sets the following fields:

    * `tdata.obsp['{key_added}_distances']` : :class:`ndarray <numpy.ndarray>`/:class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if `obs` is `None` or a sequence.
        - Distances between observations.
    * `tdata.obsp['{key_added}_connectivities']` : :class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if distance is sparse.
        - Connectivity between observations.
    * `tdata.obs['{key_added}_distances']` : :class:`Series <pandas.Series>` (dtype `float`) if `obs` is a string.
        - Distance from specified observation to others.
    """
    # Setup
    _set_random_state(random_state)
    key_added = key_added or "tree"
    connect_key = _format_keys(connect_key, "connectivities")
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)
    metric_fn = _get_tree_metric(metric)
    single_obs = False
    if update:
        _check_previous_params(tdata, {"metric": metric}, key_added, ["neighbors", "distances"])
    # Get set of pairs for each tree
    if not obs and not connect_key:
        tree_pairs = _all_pairs_shared_tree(tdata, tree_keys, sample_n)
    else:
        if connect_key:
            pairs = list(zip(*tdata.obsp[connect_key].nonzero(), strict=False))  # type: ignore
            pairs = [(tdata.obs_names[i], tdata.obs_names[j]) for i, j in pairs]
        elif isinstance(obs, str):
            pairs = [(i, obs) for i in tdata.obs_names]
            single_obs = True
        elif isinstance(obs, Sequence) and isinstance(obs[0], str):
            pairs = [(i, j) for i in obs for j in obs]
        elif isinstance(obs, Sequence) and isinstance(obs[0], tuple):
            pairs = obs
        else:
            raise ValueError("Invalid type for parameter `obs`.")
        if sample_n:
            pairs = random.sample(pairs, sample_n)
        tree_pairs = _assign_pairs_to_trees(pairs, tdata, tree_keys)
        tree_pairs = _sample_pairs(tree_pairs, sample_n)
    # Compute distances for each tree
    rows, cols, data = [], [], []
    for key, pairs in tree_pairs.items():
        tree_rows, tree_cols, tree_data = _tree_distance(trees[key], depth_key, metric_fn, pairs)
        rows.extend(tree_rows)
        cols.extend(tree_cols)
        data.extend(tree_data)
    # Distance to single observation
    if single_obs:
        distances = pd.DataFrame({key_added: data}, index=rows)
        tdata.obs[f"{key_added}_distances"] = distances
    # Pairwise distances
    else:
        distances, dense = _convert_pair_distance_to_matrix(tdata, rows, cols, data)
        param_dict = {
            "connectivities_key": f"{key_added}_connectivities",
            "distances_key": f"{key_added}_distances",
            "params": {
                "metric": metric,
                "random_state": random_state,
            },
        }
        tdata.uns[f"{key_added}_distances"] = param_dict
        if dense:
            _set_distances_and_connectivities(tdata, key_added, distances, None, update)
        else:
            _set_distances_and_connectivities(tdata, key_added, distances, _csr_data_mask(distances), update)
    if copy:
        return distances
