from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, overload, Union
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import _check_tree_overlap, get_keyed_node_data, get_keyed_obs_data, get_trees, get_leaves_from_node

def _run_permutations(
    data: pd.DataFrame,
    n_permutations: int,
    reduction_fn: Callable[[np.ndarray], Union[np.ndarray, float]],
    difference_fn: Callable[[Union[np.ndarray, float], Union[np.ndarray, float]], float],
    n_right: int,
    n_left: int
) -> np.ndarray:
    """
    Randomly permute row assignments across two groups and record
    (right_stat - left_stat) for each permutation.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset to split each permutation.
    n_permutations : int
        Number of permutations to run.
    reduction_fn
        Callable function that can reduce the data from all the leaves of a given split to a vector or scalar.
    difference_fn
        Callable function that takes to outputs from reduction_fn (one from each side of a node) and returns a scalar.
    n_right : int
        Size of the "right" group in each permutation.
    n_left : int
        Size of the "left" group in each permutation.

    Returns
    -------
    np.ndarray
        Array of length n_permutations with permutation statistics.
    """
    n = len(data)

    permutation_vals = np.zeros(n_permutations, dtype=float)

    for i in range(n_permutations):
        # Randomly permute the row indices
        perm = np.random.permutation(n)

        # Take the first n_left as left, next n_right as right
        left_idx = perm[:n_left]
        right_idx = perm[n_left:n_left + n_right]

        left_df = data.iloc[left_idx]
        right_df = data.iloc[right_idx]
        left_stat = reduction_fn(left_df.to_numpy())
        right_stat = reduction_fn(right_df.to_numpy())

        permutation_vals[i] = difference_fn(left_stat, right_stat)

    return permutation_vals

def difference(a: float, b: float) -> float:
    return a - b

@overload
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    reduction_fn: Callable[[np.ndarray], np.ndarray],
    difference_fn: Callable[[np.ndarray, np.ndarray], float],
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 1000,
    min_required_permutations: int = 40,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True
) -> pd.DataFrame: ...
@overload
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    reduction_fn: Callable[[np.ndarray], np.ndarray],
    difference_fn: Callable[[np.ndarray, np.ndarray], float],
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 1000,
    min_required_permutations: int = 40,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False
) -> None: ...
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    reduction_fn: Callable[[np.ndarray], Union[np.ndarray, float]] = np.mean,
    difference_fn: Callable[[Union[np.ndarray, float], Union[np.ndarray, float]], float] = difference,
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 500,
    min_required_permutations: int = 50,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False
) -> pd.DataFrame | None:
    """
    Compute a split statistic across every internal split of each tree and (optionally)
    a two-sided permutation p-value.

    For each requested observation key, the function traverses every tree in ``tdata``
    (or only those specified via ``tree``). At each split node (a node with at least
    two children), the leaves under the left child and the leaves under the right child
    are identified. The user-supplied ``reduction_fn`` is applied separately to the data
    for the left and right leaf sets, producing two vectors or scalars. The **split statistic**
    is defined as ``difference_fn(left_stat, right_stat)``. `.

    If ``permutation_test`` is enabled and there are enough distinct labelings to
    justify resampling, a two-sided permutation test is performed by repeatedly
    shuffling the pooled rows (left+right) and recomputing the difference. The
    number of permutations executed is the minimum of the user-requested
    ``n_permutations`` and the **theoretical maximum** distinct labelings,
    ``comb(n_left + n_right, n_left)``. The p-value is computed with a standard
    +1 smoothing:

        p = ( #{ |perm_stat| >= |observed| } + 1 ) / ( permutations_performed + 1 )

    Results are written back to the tree(s):

    - On **edges**: the per-branch statistic for each child edge is stored under
      ``{key_added}`` for that edge:
        - ``(parent -> left_child)[key_added] = left_stat``
        - ``(parent -> right_child)[key_added] = right_stat``

    - On **nodes** (parent split node): when the permutation test runs, the node
      receives:
        - ``f"{key_added}_split"`` -> the observed split statistic
        - ``f"{key_added}_pval"``  -> the two-sided permutation p-value

    When ``copy=True``, a DataFrame of the node-level results is returned with
    columns for each ``key_added`` suffixed by ``"_split"`` and ``"_pval"``; otherwise
    the function returns ``None`` after mutating the graphs in-place.

    Notes
    -----
    - **Method signature.** ``method`` must accept a pandas object containing only
      the rows for one side of the split (typically a ``pd.Series`` or single-column
      ``pd.DataFrame``) and return a single ``float``. It should be invariant to
      row order because permutations will reshuffle rows.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to reconstruct.
    reduction_fn
        Callable function that can reduce the data from all the leaves of a given split to a vector or scalar. Defaults
        to np.mean.
    difference_fn
        Callable function that takes to outputs from reduction_fn (one from each side of a node) and returns a scalar.
        Defaults to left_stat - right_stat.
    permutation_test
        Whether to perform a permutation test to obtain a two-sided p-value for the
        split statistic at each split node (subject to size constraints below).
    n_permutations
        Upper bound on the number of permutations to run. The actually executed
        number is ``min(n_permutations, comb(n_left + n_right, n_left))`` per split.
    min_required_permutations
        Minimum number of **distinct** permutations required to run the test. If
        ``comb(n_left + n_right, n_left) <= min_required_permutations``, the test
        is skipped for that split.
    keys_added
        Attribute keys of `tdata.obst[tree].nodes` where split statistics will be stored. If `None`, `keys` are used.
    tree
        The `obst` key or keys of the trees to use. If `None`, all trees are used.
    copy
        If True, returns a :class:`DataFrame <pandas.DataFrame>` with split statistics.

    Returns
    -------
    pd.DataFrame or None
        If ``copy=True``, a DataFrame indexed by node identifiers with columns
        ``f"{key_added}_split"`` and ``f"{key_added}_pval"`` for each requested key.
        Nodes where the permutation test was skipped will have missing columns/values
        for that key. If ``copy=False``, returns ``None`` after writing attributes
        onto the graphs.
    """

    if isinstance(keys, str):
        keys = [keys]
    if keys_added is None:
        keys_added = keys
    if isinstance(keys_added, str):
        keys_added = [keys_added]
    if len(keys) != len(keys_added):
        raise ValueError("Length of keys must match length of keys_added.")
    if n_permutations < min_required_permutations:
        raise ValueError("n_permutations must at least be min_required_permutations.")
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)

    for key, key_added in zip(keys, keys_added):
        data, is_array, is_square = get_keyed_obs_data(tdata, key)
        if not(is_array or is_square):
            data = data[key]
        for _, t in trees.items():
            for parent in nx.topological_sort(t):
                children = list(t.successors(parent))

                # don't do anything if not a split
                if len(children) < 2:
                    continue

                # for each child, find all leaves
                left_child = children[0]
                right_child = children[1]

                left_leaves = get_leaves_from_node(t, left_child)
                right_leaves = get_leaves_from_node(t, right_child)

                left_leaves_in_data = [x for x in left_leaves if x in data.index]
                right_leaves_in_data = [x for x in right_leaves if x in data.index]

                if len(left_leaves_in_data) > 0 and len(right_leaves_in_data) > 0:
                    left_data = data.loc[left_leaves_in_data]
                    right_data = data.loc[right_leaves_in_data]
                else:
                    continue

                n_right = len(right_leaves_in_data)
                n_left = len(left_leaves_in_data)

                left_stat = reduction_fn(left_data.to_numpy())
                right_stat = reduction_fn(right_data.to_numpy())
                split_stat = difference_fn(left_stat, right_stat)
                nx.set_edge_attributes(t, {
                    (parent, left_child): {key_added: left_stat},
                    (parent, right_child): {key_added: right_stat},
                })
                # don't perform more than theoretical maximum number of permutations
                permutations_to_do = min(comb(n_left + n_right, n_left), n_permutations)

                if permutation_test and permutations_to_do > min_required_permutations:
                    lr_data = pd.concat([left_data, right_data])

                    permutation_stats = _run_permutations(
                        lr_data,
                        permutations_to_do,
                        reduction_fn,
                        difference_fn,
                        n_right,
                        n_left
                    )

                    two_sided_pval = (np.sum(np.abs(permutation_stats) >= abs(split_stat)) + 1) / (permutations_to_do + 1)
                    nx.set_node_attributes(t, {
                        parent: {
                            f"{key_added}_split" : split_stat,
                            f"{key_added}_pval": two_sided_pval
                        },
                    })

    if copy:
        # get _pval and _split for each node
        return get_keyed_node_data(
            tdata, [suffix for x in keys_added for suffix in (f"{x}_split", f"{x}_pval")], tree_keys
        )
