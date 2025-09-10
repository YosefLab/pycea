from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import get_keyed_node_data, get_trees
from ._utils import _check_tree_overlap


@overload

def n_extant(
    tdata: td.TreeData,
    depth_key: str,
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    copy: Literal[True, False] = True,
) -> pd.DataFrame:
    ...


@overload

def n_extant(
    tdata: td.TreeData,
    depth_key: str,
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    copy: Literal[True, False] = False,
) -> None:
    ...


def n_extant(
    tdata: td.TreeData,
    depth_key: str,
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    copy: Literal[True, False] = False,
) -> pd.DataFrame | None:
    """
    Count extant branches over time.

    Computes the number of extant branches for each depth bin of the tree, optionally stratified by a `obst` grouping variable(s).

    Parameters
    ----------
    tdata
        TreeData object.
    depth_key
        Attribute of `tdata.obst[tree].nodes` storing node depth.
    groupby
        obst key(s) used to group counts. If None, counts across all branches.
    bins
        Number of histogram bins or explicit bin edges.
    tree
        tdata.obst key or keys of trees to use. If None, all trees are used.
    key_added
        Key under which to store results in `tdata.uns`.
    copy
        If True, return a DataFrame with extant counts.

    Returns
    -------
    counts - DataFrame with columns `time`, `n_extant`, grouping variables, and `tree`.
    """
    # Validate tree keys and get trees
    tree_keys = tree
    _check_tree_overlap(tdata, tree_keys)
    trees = get_trees(tdata, tree_keys)

    # Determine grouping variables
    if groupby is None:
        groupby_names: list[str] = ["_all"]
    elif isinstance(groupby, str):
        groupby_names = [groupby]
    else:
        groupby_names = list(groupby)

    results = []
    for key, t in trees.items():
        # Retrieve node data
        node_keys = [depth_key] + groupby_names
        nodes = get_keyed_node_data(tdata, keys=node_keys, tree=key)
        nodes.index = nodes.index.droplevel("tree")
        if groupby is None:
            nodes["_all"] = 1

        # Build time bins
        timepoints = np.histogram_bin_edges(nodes[depth_key], bins=bins)

        # Initialize counts per group
        groups = nodes[groupby_names].drop_duplicates().itertuples(index=False, name=None)
        group_counts = {g: np.zeros(len(timepoints)) for g in groups}

        # Iterate edges
        for u, v in t.edges:
            birth = nodes.loc[u, depth_key] - 1e-4
            birth_idx = np.searchsorted(timepoints, birth, side="right")
            if len(list(t.successors(v))) == 0:
                death_idx = len(timepoints)
            else:
                death = nodes.loc[v, depth_key] + 1e-4
                death_idx = np.searchsorted(timepoints, death, side="left")
            g = tuple(nodes.loc[u, groupby_names])
            group_counts[g][birth_idx:death_idx] += 1

        # Assemble DataFrame
        for g, counts in group_counts.items():
            data = {"time": timepoints, "n_extant": counts, "tree": key}
            if groupby is not None:
                for name, value in zip(groupby_names, g, strict=False):
                    data[name] = value
            result_df = pd.DataFrame(data)
            results.append(result_df)

    extant = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if groupby is None and "_all" in extant.columns:
        extant = extant.drop(columns="_all")

    # Store and return
    tdata.uns[key_added] = extant
    if copy:
        return extant
    return None
