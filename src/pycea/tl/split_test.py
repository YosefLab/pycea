from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import comb
from typing import Literal, overload

import networkx as nx
import numpy as np
import pandas as pd
import treedata as td
from scipy.stats import ttest_ind
from sklearn.metrics import DistanceMetric

from pycea.utils import _check_tree_overlap, _get_descendant_leaves, get_keyed_obs_data, get_trees

from ._aggregators import _Aggregator, _AggregatorFn, _get_aggregator
from ._metrics import MeanDiffMetric, _Metric, _MetricFn
from ._utils import _set_random_state


def _run_permutations(
    data: pd.DataFrame,
    n_permutations: int,
    aggregate_fn: _AggregatorFn,
    metric_fn: _MetricFn,
    n_right: int,
    n_left: int,
) -> np.ndarray:
    """
    Randomly permute row assignments across two groups and record (right_stat - left_stat) for each permutation.

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
        right_idx = perm[n_left : n_left + n_right]

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
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    aggregate: _AggregatorFn | _Aggregator = "mean",
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    n_permutations: int = 100,
    random_state: int | None = None,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...
@overload
def split_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    aggregate: _AggregatorFn | _Aggregator = "mean",
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    n_permutations: int = 100,
    random_state: int | None = None,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = False,
) -> None: ...
def split_test(
    tdata: td.TreeData,
    keys: str | Sequence[str],
    comparison: Literal["siblings", "rest"] = "siblings",
    test: Literal["permutation", "t-test"] | None = "permutation",
    aggregate: _AggregatorFn | _Aggregator = "mean",
    metric: _MetricFn | _Metric | Literal["mean_difference"] = "mean_difference",
    metric_kwds: Mapping | None = None,
    n_permutations: int = 100,
    random_state: int | None = None,
    equal_var: bool = True,
    min_group_leaves: int = 10,
    keys_added: str | Sequence[str] | None = None,
    tree: str | Sequence[str] | None = None,
    copy: Literal[True, False] = True,
) -> pd.DataFrame | None:
    r"""
    Compute permutation or t test for splits in a tree.

    For each requested observation key, the function traverses every tree in ``tdata``
    (or only those specified via ``tree``). At each internal node, the key's value in the descendant
    leaves is compared between two groups defined by the ``comparison`` setting:

      • **comparison='siblings':**
        For binary splits, compare the two children's descendant leaves directly against each other.
        For non-binary splits, use a one-vs-rest scheme comparing each child individually against
        the pooled set of all other children at that node.

      • **comparison='rest':**
        Compare the descendant leaves of each child to all other leaves in the tree.

    The `test` parameter defines how **group1** leaves are compared to **group2** leaves:

      • **test='permutation':**
        a two-sided permutation test is performed by repeatedly
        shuffling the pooled rows (group_1 + group_2), applying the ``aggregate`` function, and
        then recomputing the split statistic using the provided `metric` function.
        The number of permutations executed is the minimum of the user-requested
        ``n_permutations`` and the **theoretical maximum** number of distinct labelings,
        ``comb(n_left + n_right, n_left)``. The p-value is computed with standard
        +1 smoothing:

            p = ( #{ \|perm_stat\| >= \|observed\| } + 1 ) / ( permutations_performed + 1 )

      • **test='test-t':**
        a two-sided t-test is performed for each split. Note that for small numbers of leaves the p-value of this
        t-test can be unreliable.

      • **test=None:**
        no statistical test is performed; only the split statistic is computed.

    P-values are calculated as long as both groups have at least ``min_group_leaves`` leaves;
    otherwise, no test is performed for that split and the p-value is set to NaN.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to reconstruct.
    comparison
        Set of leaves to compare to:

        * 'siblings' : leaves descending from a given node are compared to leaves descending from its siblings.
        * 'rest' : leaves descending from a given node are compared to all other leaves of the tree.
    test
        Type of test to perform to compare the two groups. "t-test" can only be used for scalar keys.
    aggregate
        Function to reduce the data from all the leaves of a given split to a vector or scalar. Can be a known
        aggregator or a callable. Only used for test="permutation".
    metric
        A metric to compare the children from both sides of the tree. Can be a known metric or a callable. Only used
        for test="permutation".
    metric_kwds
        Options for the metric.
    equal_var
        Boolean indicating if the variance in the two groups should be assumed to be equal. Only used for
        test="t-test".
    n_permutations
        Upper bound on the number of permutations to run. The actually executed
        number is ``min(n_permutations, comb(n_left + n_right, n_left))`` per split.
    random_state
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
    Returns `None` if `copy=False`, else returns :class:`DataFrame <pandas.DataFrame>` with columns:
        - `'tree'` - Tree name.
        - `'key'` - Observation key.
        - `'parent'` - Parent node of the split.
        - `'group1'` - Child node being compared.
        - `'group2'` - Other child node(s) or "rest" being compared against.
        - `'value1'` - Aggregate value for `group1`.
        - `'value2'` - Aggregate value for `group2`.
        - `'pval'` - p-value from the statistical test (if performed).

    Sets the following fields:

    * `tdata.obst[tree].nodes[f"{key_added}_value"]` : `float`/:class:`ndarray <numpy.ndarray>`
        - Aggregate value for each child at each split.
    * `tdata.obst[tree].edges[f"{key_added}_pval"]` : `float`
        - p-value from statistical test for each split.
    * `tdata.obst[tree].edges[f"{key_added}_metric"]` : `float`
        - Metric value for each split (only if test="permutation").
    """
    _set_random_state(random_state)
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

    if metric == "mean_difference":
        metric_fn = MeanDiffMetric()
    else:
        metric_fn = DistanceMetric.get_metric(metric, **(metric_kwds or {}))

    aggregate_fn = _get_aggregator(aggregate)

    # for each tree, get dictionary with keys as nodes and values as leaves
    all_trees_leaves_dict = {tree_id: _get_descendant_leaves(t) for tree_id, t in trees.items()}
    # Record lists for dataframe if copy
    records = []

    for key, key_added in zip(keys, keys_added, strict=False):
        data, is_array, is_square = get_keyed_obs_data(tdata, key)
        if (is_array or is_square) and test == "t-test":
            raise ValueError("t-test cannot be performed for vector valued keys.")

        if not (is_array or is_square):
            data = data[key]

        data = data.dropna()
        index_set = set(data.index)

        for tree_id, t in trees.items():
            tree_leaves_dict = all_trees_leaves_dict[tree_id]

            # filter out children not in data index
            tree_leaves_dict = {
                node: [u for u in leaves if u in index_set] for node, leaves in tree_leaves_dict.items()
            }

            for parent in nx.topological_sort(t):
                children = list(t.successors(parent))

                # don't do anything if not a split and comparing siblings
                if len(children) < 2 and comparison == "siblings":
                    continue

                # get leaves from children
                leaves_dict = {child: tree_leaves_dict.get(child, []) for child in children}

                for child, left_leaves in leaves_dict.items():
                    # initialize record
                    record = {"tree": tree_id, "key": key, "parent": parent, "group1": str(child)}
                    record.update(dict.fromkeys(["group2", "value1", "value2", "pval"], np.nan))

                    if comparison == "siblings":
                        # leaves of other children at split
                        right_leaves = [
                            leaf
                            for other_child, leaves in leaves_dict.items()
                            if other_child != child
                            for leaf in leaves
                        ]
                    else:
                        # all other leaves
                        child_leaf_set = set(left_leaves)
                        right_leaves = [
                            v for vals in tree_leaves_dict.values() for v in vals if v not in child_leaf_set
                        ]

                    if len(left_leaves) > 0 and len(right_leaves) > 0:
                        left_data = data.loc[left_leaves]
                        right_data = data.loc[right_leaves]
                    elif len(children) == 2 and comparison == "siblings":
                        break  # special case where there are two children and we're comparing siblings

                    n_right = len(right_leaves)
                    n_left = len(left_leaves)

                    left_stat = aggregate_fn(left_data.to_numpy())
                    right_stat = aggregate_fn(right_data.to_numpy())
                    split_stat = float(metric_fn.pairwise(left_stat.reshape(1, -1), right_stat.reshape(1, -1)))

                    nx.set_node_attributes(t, {child: {f"{key_added}_value": left_stat}})

                    record["value1"] = left_stat
                    record["value2"] = right_stat

                    if len(children) == 2 and comparison == "siblings":
                        # handle special case in which there are exactly two children and comparing siblings
                        nx.set_node_attributes(t, {children[1]: {f"{key_added}_value": right_stat}})
                        record["group2"] = str(children[1])
                    else:
                        record["group2"] = (
                            ", ".join([x for x in children if x != child]) if comparison == "siblings" else "rest"
                        )

                    if test is not None:
                        if n_right >= min_group_leaves and n_left >= min_group_leaves:
                            if test == "permutation":
                                lr_data = pd.concat([left_data, right_data])

                                # don't perform more than theoretical maximum number of permutations
                                permutations_to_do = min(comb(n_left + n_right, n_left), n_permutations)

                                permutation_stats = _run_permutations(
                                    lr_data, permutations_to_do, aggregate_fn, metric_fn, n_right, n_left
                                )

                                two_sided_pval = (np.sum(np.abs(permutation_stats) >= abs(split_stat)) + 1) / (
                                    permutations_to_do + 1
                                )

                            elif test == "t-test":
                                _, two_sided_pval = ttest_ind(
                                    left_data.to_numpy(), right_data.to_numpy(), axis=None, equal_var=equal_var
                                )

                            nx.set_edge_attributes(t, {(parent, child): {f"{key_added}_pval": two_sided_pval}})

                            if test == "permutation":
                                nx.set_edge_attributes(t, {(parent, child): {f"{key_added}_metric": split_stat}})

                            if len(children) == 2 and comparison == "siblings":
                                # handle special case in which there are exactly two children and comparing siblings
                                nx.set_edge_attributes(
                                    t, {(parent, children[1]): {f"{key_added}_pval": two_sided_pval}}
                                )
                                if test == "permutation":
                                    if metric == "mean_difference":
                                        # if mean difference metric, multiply by -1 before writing off value
                                        nx.set_edge_attributes(
                                            t, {(parent, children[1]): {f"{key_added}_metric": -split_stat}}
                                        )
                                    else:
                                        nx.set_edge_attributes(
                                            t, {(parent, children[1]): {f"{key_added}_metric": split_stat}}
                                        )
                            record["pval"] = two_sided_pval

                    records.append(record)

                    if len(children) == 2 and comparison == "siblings":
                        # only need to do one test if there are two children and comparing siblings
                        break

    if copy:
        return pd.DataFrame.from_records(records)
