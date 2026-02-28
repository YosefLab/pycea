from __future__ import annotations

import heapq
import random
from collections.abc import Sequence
from typing import Literal, overload

import scipy as sp
import treedata as td

from pycea.utils import _check_tree_overlap, check_tree_has_key, get_leaves, get_trees

from ._metrics import _get_tree_metric, _TreeMetric
from ._utils import (
    _assert_param_xor,
    _check_previous_params,
    _csr_data_mask,
    _set_distances_and_connectivities,
    _set_random_state,
)


def _lca_neighbors(tree, start_node, n_neighbors, max_dist, depth_key):
    """Find neighbors using LCA distance via a walk-up approach.

    Walks from start_node to root, collecting sibling subtree leaves at each level.
    All leaves in a sibling subtree share the same LCA distance (depth_key of their
    common ancestor with start_node). Processes closest relatives first.
    Time complexity: O(n) per leaf.
    """
    neighbors = []
    neighbor_distances = []
    seen = {start_node}
    node = start_node
    is_finite = n_neighbors != float("inf")

    while len(neighbors) < n_neighbors:
        parents = list(tree.predecessors(node))
        if not parents:
            break
        parent = parents[0]
        lca_dist = tree.nodes[parent][depth_key]
        seen.add(parent)

        if lca_dist <= max_dist:
            # Collect unseen leaves — their LCA with start_node is exactly parent
            sibling_leaves = []
            stack = list(tree.successors(parent))
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                if tree.out_degree(n) == 0:
                    sibling_leaves.append(n)
                else:
                    stack.extend(tree.successors(n))

            random.shuffle(sibling_leaves)
            take = sibling_leaves[: n_neighbors - len(neighbors)] if is_finite else sibling_leaves
            for leaf in take:
                neighbors.append(leaf)
                neighbor_distances.append(lca_dist)

        node = parent

    return neighbors, neighbor_distances


def _bfs_by_distance(tree, start_node, n_neighbors, max_dist, depth_key):
    """Breadth-first search for path distance neighbors."""
    queue = []
    heapq.heappush(queue, (0, start_node))
    visited = {start_node}
    neighbors = []
    neighbor_distances = []

    # Pre-compute ancestors and add to queue once
    node = start_node
    while True:
        parents = list(tree.predecessors(node))
        if not parents:
            break
        parent = parents[0]
        parent_distance = abs(tree.nodes[start_node][depth_key] - tree.nodes[parent][depth_key])
        if parent_distance <= max_dist:
            heapq.heappush(queue, (parent_distance, parent))
        visited.add(parent)
        node = parent

    # Breadth-first search using direct children only
    while queue and (len(neighbors) < n_neighbors):
        distance, node = heapq.heappop(queue)
        children = list(tree.successors(node))
        random.shuffle(children)
        for child in children:
            if child not in visited:
                child_distance = distance + abs(tree.nodes[node][depth_key] - tree.nodes[child][depth_key])
                if child_distance <= max_dist:
                    if tree.out_degree(child) == 0:
                        neighbors.append(child)
                        neighbor_distances.append(child_distance)
                        if len(neighbors) >= n_neighbors:
                            break
                    heapq.heappush(queue, (child_distance, child))
                visited.add(child)

    return neighbors, neighbor_distances


def _tree_neighbors(tree, n_neighbors, max_dist, depth_key, metric, leaves=None):
    """Identify neighbors in a given tree."""
    rows, cols, distances = [], [], []
    if leaves is None:
        leaves = [node for node in tree.nodes() if tree.out_degree(node) == 0]
    for leaf in leaves:
        if metric == "lca":
            leaf_neighbors, leaf_distances = _lca_neighbors(tree, leaf, n_neighbors, max_dist, depth_key)
        else:
            leaf_neighbors, leaf_distances = _bfs_by_distance(tree, leaf, n_neighbors, max_dist, depth_key)
        rows.extend([leaf] * len(leaf_neighbors))
        cols.extend(leaf_neighbors)
        distances.extend(leaf_distances)
    return rows, cols, distances


@overload
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
    copy: Literal[True, False] = True,
) -> tuple[sp.sparse.csr_matrix, sp.sparse.csr_matrix]: ...
@overload
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
    copy: Literal[True, False] = False,
) -> None: ...
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
    copy: Literal[True, False] = False,
) -> None | tuple[sp.sparse.csr_matrix, sp.sparse.csr_matrix]:
    """Identifies neighbors in the tree.

    For each leaf, this function identifies neighbors according to a chosen
    tree distance `metric` and either:

    * the top-``n_neighbors`` closest leaves (ties broken at random)

    * all leaves within a distance threshold ``max_dist``.

    Results are stored as sparse connectivities and distances, or returned when
    ``copy=True``. You can restrict the operation to a subset of leaves via
    ``obs`` and/or to specific trees via ``tree``.

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

    Examples
    --------
    Identify the 5 closest neighbors for each leaf based on path distance:

    >>> tdata = py.datasets.koblan25()
    >>> py.tl.tree_neighbors(tdata, n_neighbors=5, depth_key="time")
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
        t = trees[leaf_to_tree[obs]]
        connectivities, _, distances = _tree_neighbors(
            t, n_neighbors or float("inf"), max_dist or float("inf"), depth_key, metric, leaves=[obs]
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
        for _, t in trees.items():
            check_tree_has_key(t, depth_key)
            tree_rows, tree_cols, tree_data = _tree_neighbors(
                t, n_neighbors or float("inf"), max_dist or float("inf"), depth_key, metric
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
