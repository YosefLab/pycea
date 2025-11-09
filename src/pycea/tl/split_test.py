from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload
from math import comb

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import _check_tree_overlap, get_keyed_obs_data, get_trees, _get_descendant_leaves
from scipy.stats import ttest_ind
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
def split_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    n_permutations: int = 100,
    random_seed: int = 42,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True
) -> pd.DataFrame: ...
@overload
def split_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    n_permutations: int = 100,
    random_seed: int = 42,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False
) -> None: ...
def split_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    aggregate: _AggregatorFn | _Aggregator = 'mean',
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    n_permutations: int = 100,
    random_seed: int = 42,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy:Literal[True, False] = True
) -> pd.DataFrame | None:
    """
    Compute a split statistic across every internal split of each tree and (optionally)
    a two-sided permutation p-value.

    For each requested observation key, the function traverses every tree in ``tdata``
    (or only those specified via ``tree``). At each internal node (a node with two or
    more children), **group_1 vs. group_2** comparisons are performed. If comparison = "siblings",
    nodes are compared as follows:

      • **Binary splits (two children):**
        The first child is compared directly against the second child.

      • **Non-binary splits (three or more children):**
        A one-vs-rest scheme is used, where each child is compared individually
        against the pooled set of all other children at that node.

    If comparison = "rest", then the leaves of each node are compared against all other leaves in the tree.

    For each comparison, the user-supplied ``aggregate`` function is applied
    separately to the data for group_1 and group_2 (each producing a vector or scalar),
    and the **split statistic** is computed as ``metric.pairwise(group_1_stat, group_2_stat)``.

    If test="permutation" is set and there are enough distinct leaves in each group to
    justify resampling (based on the setting of min_group_leaves),
    a two-sided permutation test is performed by repeatedly
    shuffling the pooled rows (group_1 + group_2) and recomputing the metric.
    The number of permutations executed is the minimum of the user-requested
    ``n_permutations`` and the **theoretical maximum** number of distinct labelings,
    ``comb(n_left + n_right, n_left)``. The p-value is computed with standard
    +1 smoothing:

        p = ( #{ |perm_stat| >= |observed| } + 1 ) / ( permutations_performed + 1 )

    If test="t-test" is set and there are enough distinct leaves in each group, then a two-sided
    t-test is performed for each split. Note that for small numbers of leaves the p-value of this
    t-test can be unreliable.

    Results are written back to the tree(s):

      - **Nodes:** The per-branch aggregate value for each child edge is stored under
        ``{key_added}_value``.
      - **Edges (with test != None):** Each edge stores a p-value under ``{key_added}_pvalue``, and when
        test == "permutation", a metric value comparing the given child to all other children.

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
        to np.mean(,axis = 0). Only used for test="permutation".
    metric
        A metric to compare the children from both sides of the tree. Can be a known metric or a callable. Only used
        for test="permutation".
    metric_kwds
        Options for the metric.
    comparison
        String indicating type of comparison. "siblings" compares the leaves descending from a given node only to
        the leaves descending from its siblings (either a single test in the case of two siblings or multiple
        one vs. rest comparisons for multiple siblings). "rest" compares the leaves descending from a given node
        to all other leaves of the that the node is on.
    test
        Type of test to perform to compare the two groups. "t-test" can only be used for scalar quantities.
    equal_var
        Boolean indicating if the variance in the two groups should be assumed to be equal. Only used for
        test="t-test".
    n_permutations
        Upper bound on the number of permutations to run. The actually executed
        number is ``min(n_permutations, comb(n_left + n_right, n_left))`` per split.
    random_seed
        Random seed to ensure reproducibility of permutation test.
    min_group_leaves
        Minimum number of leaves required in each group to perform a statistical test. The t-test may be particularly
        unreliable with small sample sizes.
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
    if test is not None and test not in ["permutation", "t-test"]:
        raise ValueError("Test must either be None or set to one of 'permutation' or 't-test'.")
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)

    np.random.seed(random_seed)

    if metric == "mean_difference":
        metric_fn = MeanDiffMetric()
    else:
        metric_fn = DistanceMetric.get_metric(metric, **(metric_kwds or {}))

    aggregate_fn = Aggregator.get_aggregator(aggregate)
    df_dict = {}

    # for each tree, get dictionary with keys as nodes and values as leaves
    all_trees_leaves_dict = {tree_id: _get_descendant_leaves(t) for tree_id, t in trees.items()}
    # boolean indicating if the current key is the first key in the loop
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
        if (is_array or is_square) and test == "t-test":
            raise ValueError("t-test cannot be performed for vector valued keys.")

        if not(is_array or is_square):
            data = data[key]

        data = data.dropna()
        index_set = set(data.index)

        for tree_id, t in trees.items():

            tree_leaves_dict = all_trees_leaves_dict[tree_id]

            # filter out children not in data index
            tree_leaves_dict = {
                node: [u for u in leaves if u in index_set]
                for node, leaves in tree_leaves_dict.items()
            }

            for parent in nx.topological_sort(t):
                children = list(t.successors(parent))

                # don't do anything if not a split and comparing siblings
                if len(children) < 2 and comparison == "siblings":
                    continue

                # get leaves from children
                leaves_dict = {child: tree_leaves_dict.get(child, []) for child in children}

                for child, left_leaves in leaves_dict.items():

                    if comparison == "siblings":
                        # leaves of other children at split
                        right_leaves = [leaf for other_child, leaves in leaves_dict.items()
                                        if other_child != child
                                        for leaf in leaves]
                    else:
                        # all other leaves
                        child_leaf_set = set(left_leaves)
                        right_leaves = [v for vals in tree_leaves_dict.values() for v in vals if v not in child_leaf_set]

                    if copy and first_key:
                        tree_list.append(tree_id)
                        parent_list.append(parent)
                        group1_list.append(str(child))

                    if len(left_leaves) > 0 and len(right_leaves) > 0:
                        left_data = data.loc[left_leaves]
                        right_data = data.loc[right_leaves]
                    else:
                        # if copy, need to write something off here to avoid dataframes being of different
                        # lengths if there are different na patterns amongst the keys
                        if copy:
                            group1_value_list.append(np.nan)
                            group2_value_list.append(np.nan)
                            if test is not None:
                                pvalue_list.append(np.nan)
                            if first_key:
                                if comparison == "siblings":
                                    if len(right_leaves) > 0:
                                        group2_list.append(", ".join([x for x in children if x != child]))
                                    else:
                                        group2_list.append("")
                                elif comparison == "rest":
                                    group2_list.append("rest")

                        continue

                    n_right = len(right_leaves)
                    n_left = len(left_leaves)

                    left_stat = aggregate_fn(left_data.to_numpy())
                    right_stat = aggregate_fn(right_data.to_numpy())
                    split_stat = float(metric_fn.pairwise(left_stat.reshape(1, -1), right_stat.reshape(1, -1)))

                    nx.set_node_attributes(t, {
                        child: {f"{key_added}_value": left_stat}
                    })

                    if copy:
                        group1_value_list.append(left_stat)
                        group2_value_list.append(right_stat)

                    if len(children) == 2 and comparison == "siblings":
                        # handle special case in which there are exactly two children and comparing siblings
                        nx.set_node_attributes(t, {
                            children[1]: {f"{key_added}_value": right_stat}
                        })

                        if copy and first_key:
                            group2_list.append(children[1])
                    else:
                        if copy and first_key:
                            if comparison == "siblings":
                                group2_list.append(", ".join([x for x in children if x != child]))
                            elif comparison == "rest":
                                group2_list.append("rest")

                    if test is not None:
                        if n_right >= min_group_leaves and n_left >= min_group_leaves:
                            if test == "permutation":
                                lr_data = pd.concat([left_data, right_data])

                                # don't perform more than theoretical maximum number of permutations
                                permutations_to_do = min(comb(n_left + n_right, n_left), n_permutations)

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

                            elif test == "t-test":
                                _, two_sided_pval = ttest_ind(
                                    left_data.to_numpy(),
                                    right_data.to_numpy(),
                                    axis=None,
                                    equal_var=equal_var
                                )

                            nx.set_edge_attributes(t, {
                                (parent, child): {f"{key_added}_pvalue": two_sided_pval}
                            })

                            if test == "permutation":
                                nx.set_edge_attributes(t, {
                                    (parent, child): {f"{key_added}_metric": split_stat}
                                })

                            if len(children) == 2 and comparison == "siblings":
                                # handle special case in which there are exactly two children and comparing siblings
                                nx.set_edge_attributes(t, {
                                    (parent, children[1]): {f"{key_added}_pvalue": two_sided_pval}
                                })
                                if test == "permutation":
                                    if metric == "mean_difference":
                                        # if mean difference metric, multiply by -1 before writing off value
                                        nx.set_edge_attributes(t, {
                                            (parent, children[1]): {f"{key_added}_metric": -split_stat}
                                        })
                                    else:
                                        nx.set_edge_attributes(t, {
                                            (parent, children[1]): {f"{key_added}_metric": split_stat}
                                        })

                            if copy:
                                pvalue_list.append(two_sided_pval)

                        else:
                            if copy:
                                pvalue_list.append(np.nan)

                    if len(children) == 2 and comparison == "siblings":
                        # only need to do one test if there are two children and comparing siblings
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

            if test is not None:
                df_dict[key_added][f'{key_added}_pvalue'] = pvalue_list

        first_key = False

    if copy:
        # get _value and _pvalue for each node
        combined_df = pd.concat(df_dict.values(), axis=1)
        return combined_df
