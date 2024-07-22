from __future__ import annotations

import random
import warnings
from collections import defaultdict
from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
import scipy as sp
import treedata as td

from pycea.utils import get_leaves, get_root, get_trees


def _lca_distance(tree, depth_key, node1, node2, lca):
    """Compute the lca distance between two nodes in a tree."""
    if node1 == node2:
        return tree.nodes[node1][depth_key]
    else:
        return tree.nodes[lca][depth_key]


def _path_distance(tree, depth_key, node1, node2, lca):
    """Compute the path distance between two nodes in a tree."""
    return abs(tree.nodes[node1][depth_key] + tree.nodes[node2][depth_key] - 2 * tree.nodes[lca][depth_key])


def _tree_distance(tree, depth_key, metric, pairs=None):
    """Compute distances between pairs of nodes in a tree."""
    rows, cols, data = [], [], []
    root = get_root(tree)
    if depth_key not in tree.nodes[root]:
        raise ValueError(
            f"Tree does not have {depth_key} attribute. You can run `pycea.pp.add_depth` to add depth attribute."
        )
    lcas = dict(nx.tree_all_pairs_lowest_common_ancestor(tree, root=root, pairs=pairs))
    for (node1, node2), lca in lcas.items():
        rows.append(node1)
        cols.append(node2)
        data.append(metric(tree, depth_key, node1, node2, lca))
    return rows, cols, data


def tree_distance(
    tdata: td.TreeData,
    depth_key: str = "depth",
    obs: str | Sequence[str] | None = None,
    metric: str = "path",
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    key_added: str | None = None,
    tree: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None | np.array:
    """Computes tree distances between observations.

    Parameters
    ----------
    tdata
        The TreeData object.
    depth_key
        Key where depth is stored.
    obs
        The observations to use:
        - If `None`, pairwise distance for tree leaves is stored in `tdata.obsp[key_added]`.
        - If a string, distance to all other tree leaves is `tdata.obs[key_added]`.
        - If a sequence, pairwise distance is stored in `tdata.obsp[key_added]`.
        - If a sequence of pairs, distance between pairs is stored in `tdata.obsp[key_added]`.
    metric
        The type of tree distance to compute:
        - `'lca'`: lowest common ancestor depth.
        - `'path'`: abs(node1 depth + node2 depth - 2 * lca depth).
    sample_n
        If specified, randomly sample `sample_n` pairs of observations.
    connect_key
        If specified, compute distances only between connected observations specified by
         `tdata.obsp['{connect_key}_connectivities']`.
    random_state
        Random seed for sampling.
    key_added
        Distances are stored in `tdata.obsp['{key_added}_distances']` and
        connectivities in .obsp['{key_added}_connectivities']. Defaults to 'tree'.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`np.array` or :class:`scipy.sparse.csr_matrix` with distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`numpy.array` or :class:`scipy.sparse.csr_matrix`.
    Sets the following fields:

    `tdata.obsp['{key_added}_distances']` : :class:`numpy.array` or :class:`scipy.sparse.csr_matrix` (dtype `float`)
    if `obs` is `None` or a sequence.
    `tdata.obsp['{key_added}_connectivities']` : ::class:`scipy.sparse.csr_matrix` (dtype `float`)
    if distances is sparse.
    `tdata.obs['{key_added}_distances']` : :class:`pandas.Series` (dtype `float`) if `obs` is a string.
    """
    # Setup
    if random_state is not None:
        random.seed(random_state)
    key_added = key_added or "tree"
    if connect_key is not None:
        if "connectivities" not in connect_key:
            connect_key = f"{connect_key}_connectivities"
    tree_keys = tree
    trees = get_trees(tdata, tree_keys)
    if metric == "lca":
        metric_fn = _lca_distance
    elif metric == "path":
        metric_fn = _path_distance
    else:
        raise ValueError(f"Unknown metric {metric}. Valid metrics are 'lca' and 'path'.")
    if len(trees) > 1 and tdata.allow_overlap and len(tree_keys) != 1:
        raise ValueError("Must specify a singe tree if tdata.allow_overlap is True.")
    # All pairs
    if obs is None and connect_key is None:
        # Without sampling
        if sample_n is None:
            tree_pairs = {}
            for key, tree in trees.items():
                leaves = get_leaves(tree)
                tree_pairs[key] = [(i, j) for i in leaves for j in leaves]
        # With sampling
        else:
            tree_to_leaf = {key: get_leaves(tree) for key, tree in trees.items()}
            tree_keys = list(tree_to_leaf.keys())
            tree_n_pairs = np.array([len(leaves) ** 2 for leaves in tree_to_leaf.values()])
            tree_pairs = defaultdict(set)
            n_pairs = 0
            if sample_n > tree_n_pairs.sum():
                raise ValueError("Sample size is larger than the number of pairs.")
            k = 0
            while k < sample_n:
                tree = random.choices(tree_keys, tree_n_pairs, k=1)[0]
                i = random.choice(tree_to_leaf[tree])
                j = random.choice(tree_to_leaf[tree])
                if (i, j) not in tree_pairs[tree]:
                    tree_pairs[tree].add((i, j))
                    n_pairs += 1
                k += 1
            tree_pairs = {key: list(pairs) for key, pairs in tree_pairs.items()}
    # Selected pairs
    else:
        if connect_key is not None:
            if obs is not None:
                warnings.warn("`obs` is ignored when connectivity is specified.", stacklevel=2)
            if connect_key not in tdata.obsp.keys():
                raise ValueError(f"Connectivity key {connect_key} not found in `tdata.obsp`.")
            pairs = list(zip(*tdata.obsp[connect_key].nonzero()))
            pairs = [(tdata.obs_names[i], tdata.obs_names[j]) for i, j in pairs]
        elif isinstance(obs, str):
            pairs = [(i, obs) for i in tdata.obs_names]
        elif isinstance(obs, Sequence) and isinstance(obs[0], str):
            pairs = [(i, j) for i in obs for j in obs]
        elif isinstance(obs, Sequence) and isinstance(obs[0], tuple):
            pairs = obs
        else:
            raise ValueError("Invalid type for parameter `obs`.")
        # Assign pairs to trees
        leaf_to_tree = {leaf: key for key, tree in trees.items() for leaf in get_leaves(tree)}
        has_tree = set(leaf_to_tree.keys())
        tree_pairs = defaultdict(list)
        for i, j in pairs:
            if i in has_tree and j in has_tree and leaf_to_tree[i] == leaf_to_tree[j]:
                tree_pairs[leaf_to_tree[i]].append((i, j))
        # Sample pairs
        if sample_n is not None:
            pairs_to_tree = {pair: key for key, pairs in tree_pairs.items() for pair in pairs}
            if sample_n > len(pairs_to_tree):
                raise ValueError("Sample size is larger than the number of pairs.")
            sampled_pairs = random.sample(pairs_to_tree.keys(), sample_n)
            tree_pairs = {key: [pair for pair in pairs if pair in sampled_pairs] for key, pairs in tree_pairs.items()}
    # Compute distances
    if tree_pairs is not None:
        rows, cols, data = [], [], []
        for key, pairs in tree_pairs.items():
            tree_rows, tree_cols, tree_data = _tree_distance(trees[key], depth_key, metric_fn, pairs)
            rows.extend(tree_rows)
            cols.extend(tree_cols)
            data.extend(tree_data)
        # Point distances
        if isinstance(obs, str):
            key = list(tree_pairs.keys())[0]
            pairs = tree_pairs[key]
            rows, cols, data = _tree_distance(trees[key], depth_key, metric_fn, pairs)
            distances = pd.DataFrame({key_added: data}, index=rows)
            tdata.obs[f"{key_added}_distances"] = distances
        # Pairwise distances
        else:
            rows = [tdata.obs_names.get_loc(row) for row in rows]
            cols = [tdata.obs_names.get_loc(col) for col in cols]
            distances = sp.sparse.csr_matrix((data, (rows, cols)), shape=(tdata.n_obs, tdata.n_obs))
            if len(data) == tdata.n_obs**2:
                distances = distances.toarray()
            else:
                connectivities = sp.sparse.csr_matrix(
                    (np.ones(len(data)), (rows, cols)), shape=(tdata.n_obs, tdata.n_obs)
                )
                tdata.obsp[f"{key_added}_connectivities"] = connectivities
            tdata.obsp[f"{key_added}_distances"] = distances
    if copy:
        return distances
