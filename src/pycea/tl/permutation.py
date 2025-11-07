from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import _check_tree_overlap, get_keyed_obs_data, get_trees, get_leaves_from_node
from sklearn.metrics import DistanceMetric
from collections.abc import Mapping
from ._metrics import _Metric, _MetricFn, MeanDiffMetric
from ._aggregators import _Aggregator, _AggregatorFn, Aggregator


def _run_permutations(
    data: pd.DataFrame,
    n_permutations: int,
    aggregate_fn: _AggregatorFn,
    metric_fn: _MetricFn,
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
    aggregate
        Callable function that can reduce the data from all the leaves of a given split to a vector or scalar.
    metric
        Callable function that takes to outputs from aggregate (one from each side of a node) and returns a scalar.
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
        left_stat = aggregate_fn(left_df.to_numpy())
        right_stat = aggregate_fn(right_df.to_numpy())

        permutation_vals[i] = float(metric_fn.pairwise(left_stat.reshape(1, -1), right_stat.reshape(1, -1)))

    return permutation_vals

@overload
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 500,
    min_required_permutations: int = 50,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True
) -> pd.DataFrame: ...
@overload
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 500,
    min_required_permutations: int = 50,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False
) -> None: ...
def split_permutation_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    permutation_test: Literal[True, False] = True,
    n_permutations: int = 500,
    min_required_permutations: int = 50,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True
) -> pd.DataFrame | None:
    """
    Compute a split statistic across every internal split of each tree and (optionally)
    a two-sided permutation p-value.

    For each requested observation key, the function traverses every tree in ``tdata``
    (or only those specified via ``tree``). At each internal node (a node with two or
    more children), **group_1 vs. group_2** comparisons are performed as follows:

      • **Binary splits (two children):**
        The first child is compared directly against the second child.

      • **Non-binary splits (three or more children):**
        A one-vs-rest scheme is used, where each child is compared individually
        against the pooled set of all other children at that node.

    For each comparison, the user-supplied ``aggregate`` function is applied
    separately to the data for group_1 and group_2 (each producing a vector or scalar),
    and the **split statistic** is computed as ``metric.pairwise(group_1_stat, group_2_stat)``.

    If ``permutation_test`` is enabled and there are enough distinct labelings to
    justify resampling, a two-sided permutation test is performed by repeatedly
    shuffling the pooled rows (group_1 + group_2) and recomputing the metric.
    The number of permutations executed is the minimum of the user-requested
    ``n_permutations`` and the **theoretical maximum** number of distinct labelings,
    ``comb(n_left + n_right, n_left)``. The p-value is computed with standard
    +1 smoothing:

        p = ( #{ |perm_stat| >= |observed| } + 1 ) / ( permutations_performed + 1 )

    Results are written back to the tree(s):

      - **Edges:** The per-branch aggregate value for each child edge is stored under
        ``{key_added}_value``.
      - **Edges (with permutation test):** Each edge also stores a permutation p-value
        under ``{key_added}_pvalue``.
      - **Nodes:** When ``copy=True``, a summary DataFrame is returned containing all
        comparisons performed at each internal node.

    When ``copy=True``, the function returns a :class:`pandas.DataFrame` summarizing
    the group_1 vs. group_2 comparisons for each split; otherwise, the trees are
    modified in-place and the function returns ``None``.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to reconstruct.
    aggregate
        Callable function that can reduce the data from all the leaves of a given split to a vector or scalar. Defaults
        to np.mean(,axis = 0).
    metric
        A metric to compare the children from both sides of the tree. Can be a known metric or a callable.
    metric_kwds
        Options for the metric.
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

    if metric == "mean_difference":
        metric_fn = MeanDiffMetric()
    else:
        metric_fn = DistanceMetric.get_metric(metric, **(metric_kwds or {}))

    aggregate_fn = Aggregator.get_aggregator(aggregate)
    df_dict = {}

    first_key = True

    for key, key_added in zip(keys, keys_added):

        # lists for dataframe if copy = True
        parent_list = []
        group1_list = []
        group2_list = []
        group1_value_list = []
        group2_value_list = []
        pvalue_list = []
        tree_list = []

        data, is_array, is_square = get_keyed_obs_data(tdata, key)
        data = data.dropna()
        index_set = set(data.index)
        if not(is_array or is_square):
            data = data[key]
        for tree_id, t in trees.items():
            for parent in nx.topological_sort(t):
                children = list(t.successors(parent))

                # don't do anything if not a split
                if len(children) < 2:
                    continue

                # get leaves that are in the data
                leaves_dict = {
                    child: [u for u in get_leaves_from_node(t, child) if u in index_set]
                    for child in children
                }
                for child, left_leaves in leaves_dict.items():
                    # All other leaves except those from the current child
                    right_leaves = [leaf for other_child, leaves in leaves_dict.items()
                                    if other_child != child
                                    for leaf in leaves]

                    if len(left_leaves) > 0 and len(right_leaves) > 0:
                        left_data = data.loc[left_leaves]
                        right_data = data.loc[right_leaves]
                    else:
                        continue

                    if copy and first_key:
                        tree_list.append(tree_id)
                        parent_list.append(parent)

                    n_right = len(right_leaves)
                    n_left = len(left_leaves)

                    left_stat = aggregate_fn(left_data.to_numpy())
                    right_stat = aggregate_fn(right_data.to_numpy())
                    split_stat = float(metric_fn.pairwise(left_stat.reshape(1, -1), right_stat.reshape(1, -1)))

                    nx.set_edge_attributes(t, {
                        (parent, child): {f"{key_added}_value": left_stat}
                    })

                    if copy:
                        group1_value_list.append(left_stat)
                        group2_value_list.append(right_stat)
                        if first_key:
                            group1_list.append(child)

                    if len(children) == 2:
                        # handle special case in which there are exactly two children
                        nx.set_edge_attributes(t, {
                            (parent, children[1]): {f"{key_added}_value": right_stat}
                        })

                        if copy and first_key:
                            group2_list.append(children[1])
                    else:
                        if copy and first_key:
                            group2_list.append(", ".join([x for x in children if x != child]))

                    # don't perform more than theoretical maximum number of permutations
                    permutations_to_do = min(comb(n_left + n_right, n_left), n_permutations)

                    if permutation_test:
                        if permutations_to_do > min_required_permutations:
                            lr_data = pd.concat([left_data, right_data])

                            permutation_stats = _run_permutations(
                                lr_data,
                                permutations_to_do,
                                aggregate_fn,
                                metric_fn,
                                n_right,
                                n_left
                            )

                            two_sided_pval = (np.sum(np.abs(permutation_stats) >= abs(split_stat)) + 1) / (
                                    permutations_to_do + 1)

                            nx.set_edge_attributes(t, {
                                (parent, child): {f"{key_added}_pvalue": two_sided_pval}
                            })

                            if len(children) == 2:
                                # handle special case in which there are exactly two children
                                nx.set_edge_attributes(t, {
                                    (parent, children[1]): {f"{key_added}_pvalue": two_sided_pval}
                                })

                            if copy:
                                pvalue_list.append(two_sided_pval)

                        else:
                            if copy:
                                pvalue_list.append(np.nan)

                    if len(children) == 2:
                        # only need to do one test if there are two children
                        break

        if copy:
            if first_key:
                # write off everything for the first key
                df_dict[key_added] = pd.DataFrame(
                    {
                        'tree': tree_list,
                        'parent': parent_list,
                        'group1': group1_list,
                        'group2': group2_list,
                        f'group1_{key_added}_value': group1_value_list,
                        f'group2_{key_added}_value': group2_value_list
                    }
                )
            else:
                # only write off values after the first key
                df_dict[key_added] = pd.DataFrame(
                    {
                        f'group1_{key_added}_value': group1_value_list,
                        f'group2_{key_added}_value': group2_value_list
                    }
                )

            if permutation_test:
                df_dict[key_added][f'{key_added}_pvalue'] = pvalue_list

        first_key = False

    if copy:
        # get _value and _pvalue for each node
        combined_df = pd.concat(df_dict.values(), axis=1)
        return combined_df
