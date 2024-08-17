from __future__ import annotations

import heapq
import random
from collections.abc import Sequence

import networkx as nx
import scipy as sp
import treedata as td

from pycea.utils import check_tree_has_key, get_leaves, get_trees

from ._metrics import _get_tree_metric, _TreeMetric
from ._utils import (
    _assert_param_xor,
    _check_previous_params,
    _check_tree_overlap,
    _csr_data_mask,
    _set_distances_and_connectivities,
    _set_random_state,
)


def _bfs_by_distance(tree, start_node, n_neighbors, max_dist, metric, depth_key):
    """Breadth-first search with a maximum distance."""
    # Setup
    queue = []
    heapq.heappush(queue, (0, start_node))
    visited = set()
    visited.add(start_node)
    neighbors = []
    neighbor_distances = []
    # Breadth-first search
    while queue and (len(neighbors) < n_neighbors):
        distance, node = heapq.heappop(queue)
        # Add children to queue
        children = list(nx.descendants(tree, node))
        random.shuffle(children)
        for child in children:
            if child not in visited:
                if metric == "path":
                    child_distance = distance + abs(tree.nodes[node][depth_key] - tree.nodes[child][depth_key])
                elif metric == "lca":
                    child_distance = distance
                if child_distance <= max_dist:
                    # Check if child is a leaf
                    if tree.out_degree(child) == 0:
                        neighbors.append(child)
                        neighbor_distances.append(child_distance)
                        if len(neighbors) >= n_neighbors:
                            break
                    heapq.heappush(queue, (child_distance, child))
                    visited.add(child)
        # Add parents to queue
        for parent in nx.ancestors(tree, node):
            if parent not in visited:
                if metric == "path":
                    parent_distance = distance + abs(tree.nodes[node][depth_key] - tree.nodes[parent][depth_key])
                elif metric == "lca":
                    parent_distance = tree.nodes[parent][depth_key]
                if parent_distance <= max_dist:
                    heapq.heappush(queue, (parent_distance, parent))
                    visited.add(parent)
    return neighbors, neighbor_distances


def _tree_neighbors(tree, n_neighbors, max_dist, depth_key, metric, leaves=None):
    """Identify neighbors in a given tree."""
    rows, cols, distances = [], [], []
    if leaves is None:
        leaves = [node for node in tree.nodes() if tree.out_degree(node) == 0]
    for leaf in leaves:
        neighbors, neighbor_distances = _bfs_by_distance(tree, leaf, n_neighbors, max_dist, metric, depth_key)
        rows.extend(neighbors)
        cols.extend([leaf] * len(neighbors))
        distances.extend(neighbor_distances)
    return rows, cols, distances


def tree_neighbors(
    tdata: td.TreeData,
    n_neighbors: int | None = None,
    max_dist: float | None = None,
    depth_key: str = "depth",
    obs: str | Sequence[str] | None = None,
    metric: _TreeMetric = "path",
    random_state: int | None = None,
    key_added: str = "tree",
    update: bool = True,
    tree: str | Sequence[str] | None = None,
    copy: bool = False,
) -> None | tuple[sp.sparse.csr_matrix, sp.sparse.csr_matrix]:
    """Identify neighbors in the tree.

    Parameters
    ----------
    tdata
        The TreeData object.
    n_neighbors
        The number of neighbors to identify for each leaf. Ties are broken randomly.
    max_dist
        If n_neighbors is None, identify all neighbors within this distance.
    depth_key
        Attribute of `tdata.obst[tree].nodes` where depth is stored.
    obs
        The observations to use:

        - If `None`, neighbors for all leaves are stored in `tdata.obsp`.
        - If a string, neighbors of specified leaf are stored in `tdata.obs`.
        - If a sequence, neighbors within specified leaves are stored in `tdata.obsp`.
    metric
        The type of tree distance to compute:

        - `'lca'`: lowest common ancestor depth.
        - `'path'`: abs(node1 depth + node2 depth - 2 * lca depth).
    random_state
        Random seed for breaking ties.
    key_added
        Neighbor distances are stored in `tdata.obsp['{key_added}_distances']` and
        neighbors in .obsp['{key_added}_connectivities']. Defaults to 'tree'.
    update
        If True, updates existing distances instead of overwriting.
    tree
        The `tdata.obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a tuple of connectivities and distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns (connectivities, distances).

    Sets the following fields:

    * `tdata.obsp['{key_added}_distances']` : :class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if `obs` is `None` or a sequence.
        - Distances to neighboring observations.
    * `tdata.obsp['{key_added}_connectivities']` : :class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if distance is sparse.
        - Set of neighbors for each observation.
    * `tdata.obs['{key_added}_neighbors']` : :class:`Series <pandas.Series>` (dtype `bool`) if `obs` is a string.
        - Set of neighbors for specified observation.
    """
    # Setup
    _set_random_state(random_state)
    _assert_param_xor({"n_neighbors": n_neighbors, "max_dist": max_dist})
    _ = _get_tree_metric(metric)
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    if update:
        _check_previous_params(tdata, {"metric": metric}, key_added, ["neighbors", "distances"])
    # Neighbors of a single leaf
    if isinstance(obs, str):
        trees = get_trees(tdata, tree_keys)
        leaf_to_tree = {leaf: key for key, tree in trees.items() for leaf in get_leaves(tree)}
        if obs not in leaf_to_tree:
            raise ValueError(f"Leaf {obs} not found in any tree.")
        tree = trees[leaf_to_tree[obs]]
        connectivities, _, distances = _tree_neighbors(
            tree, n_neighbors or float("inf"), max_dist or float("inf"), depth_key, metric, leaves=[obs]
        )
        tdata.obs[f"{key_added}_neighbors"] = tdata.obs_names.isin(connectivities)
    # Neighbors for some or all leaves
    else:
        if isinstance(obs, Sequence):
            tdata_subset = tdata[obs]
            trees = get_trees(tdata_subset, tree_keys)
        elif obs is None:
            trees = get_trees(tdata, tree_keys)
        else:
            raise ValueError("obs must be a string, a sequence of strings, or None.")
        # For each tree, identify neighbors
        rows, cols, data = [], [], []
        for _, tree in trees.items():
            check_tree_has_key(tree, depth_key)
            tree_rows, tree_cols, tree_data = _tree_neighbors(
                tree, n_neighbors or float("inf"), max_dist or float("inf"), depth_key, metric
            )
            rows.extend([tdata.obs_names.get_loc(row) for row in tree_rows])
            cols.extend([tdata.obs_names.get_loc(col) for col in tree_cols])
            data.extend(tree_data)
        # Update tdata
        distances = sp.sparse.csr_matrix((data, (rows, cols)), shape=(tdata.n_obs, tdata.n_obs))
        connectivities = _csr_data_mask(distances)
        param_dict = {
            "connectivities_key": f"{key_added}_connectivities",
            "distances_key": f"{key_added}_distances",
            "params": {
                "n_neighbors": n_neighbors,
                "max_dist": max_dist,
                "metric": metric,
                "random_state": random_state,
            },
        }
        _set_distances_and_connectivities(tdata, key_added, distances, connectivities, update)
        tdata.uns[f"{key_added}_neighbors"] = param_dict
    if copy:
        return (connectivities, distances)
