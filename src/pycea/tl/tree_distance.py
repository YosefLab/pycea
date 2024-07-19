from __future__ import annotations

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
    key_added
        Distances are stored in `'tree_distances'` unless `key_added` is specified.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`np.array` or :class:`scipy.sparse.csr_matrix` with distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`numpy.array` or :class:`scipy.sparse.csr_matrix`.
    Sets the following fields:

    `tdata.obsp[key_added]` : :class:`numpy.array` or :class:`scipy.sparse.csr_matrix` (dtype `float`)
    if `obs` is `None` or a sequence.
    `tdata.obs[key_added]` : :class:`pandas.Series` (dtype `float`) if `obs` is a string.
    """
    # Setup
    key_added = key_added or "tree_distances"
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
    # Case 1: single obs
    if isinstance(obs, str):
        for _, tree in trees.items():
            leaves = get_leaves(tree)
            if obs in leaves:
                pairs = [(node, obs) for node in leaves]
                rows, cols, data = _tree_distance(tree, depth_key, metric_fn, pairs)
                distances = pd.DataFrame({key_added: data}, index=rows)
                tdata.obs[key_added] = distances[key_added]
    # Case 2: multiple obs
    else:
        tree_pairs = {}
        if obs is None:
            for key, tree in trees.items():
                leaves = get_leaves(tree)
                pairs = [(node1, node2) for node1 in leaves for node2 in leaves]
                tree_pairs[key] = pairs
        elif isinstance(obs, Sequence):
            if isinstance(obs[0], str):
                for key, tree in trees.items():
                    leaves = list(set(get_leaves(tree)).intersection(obs))
                    pairs = [(node1, node2) for node1 in leaves for node2 in leaves]
                    tree_pairs[key] = pairs
            elif isinstance(obs[0], tuple) and len(obs[0]):
                for key, tree in trees.items():
                    leaves = get_leaves(tree)
                    pairs = []
                    for node1, node2 in obs:
                        if node1 in leaves and node2 in leaves:
                            pairs.append((node1, node2))
                    tree_pairs[key] = pairs
            else:
                raise ValueError("Invalid type for parameter `obs`.")
        else:
            raise ValueError("Invalid type for parameter `obs`.")
        # Compute distances
        rows, cols, data = [], [], []
        for key, pairs in tree_pairs.items():
            tree_rows, tree_cols, tree_data = _tree_distance(trees[key], depth_key, metric_fn, pairs)
            rows.extend(tree_rows)
            cols.extend(tree_cols)
            data.extend(tree_data)
        # Convert to matrix
        rows = [tdata.obs_names.get_loc(row) for row in rows]
        cols = [tdata.obs_names.get_loc(col) for col in cols]
        distances = sp.sparse.csr_matrix((data, (rows, cols)), shape=(len(tdata.obs_names), len(tdata.obs_names)))
        if len(data) == len(tdata.obs_names) ** 2:
            distances = distances.toarray()
        tdata.obsp[key_added] = distances
    # Return
    if copy:
        return distances
