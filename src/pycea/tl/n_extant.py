from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
import pandas as pd
import treedata as td

from pycea.utils import check_tree_has_key, get_keyed_node_data, get_root, get_trees

from ._utils import _check_tree_overlap


@overload
def n_extant(
    tdata: td.TreeData,
    depth_key: str = "depth",
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    extend_branches: bool = False,
    dropna: bool = False,
    slot: Literal["obst", "obs"] = "obst",
    copy: Literal[True, False] = True,
) -> pd.DataFrame: ...


@overload
def n_extant(
    tdata: td.TreeData,
    depth_key: str = "depth",
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    extend_branches: bool = False,
    dropna: bool = False,
    slot: Literal["obst", "obs"] = "obst",
    copy: Literal[True, False] = False,
) -> None: ...


def n_extant(
    tdata: td.TreeData,
    depth_key: str = "depth",
    groupby: Sequence[str] | str | None = None,
    bins: int | Sequence[float] = 20,
    tree: str | Sequence[str] | None = None,
    key_added: str = "n_extant",
    extend_branches: bool = False,
    dropna: bool = False,
    slot: Literal["obst", "obs"] = "obst",
    copy: Literal[True, False] = False,
) -> pd.DataFrame | None:
    """
    Counts extant branches over time.

    Computes the number of extant branches at each depth bin in the tree,
    optionally stratified by node grouping variable(s).

    Parameters
    ----------
    tdata
        TreeData object.
    depth_key
        Attribute of `tdata.obst[tree].nodes` storing depth.
    groupby
        Attribute of `tdata.obst[tree].nodes` storing grouping variable(s).
        If None, counts across all branches.
    bins
        Number of histogram bins or explicit bin edges.
    tree
        The `obst` key or keys of trees to use. If None, number of extant branches
        is computed for all trees in `obst`.
    key_added
        Key under which to store results in `tdata.uns`.
    extend_branches
        If True, leaf branches are extended to the maximum depth of the tree.
    dropna
        If True, drop rows with NaN values in grouping variables.
    slot
        Slot in TreeData object containing the depth and grouping keys.
    copy
        If True, return a DataFrame with extant counts.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`DataFrame <pandas.DataFrame>``.

    Sets the following fields:

    * `tdata.uns[key_added]` : :class:`DataFrame <pandas.DataFrame>` with columns
        depth_key, `n_extant`, grouping variables, and `tree`.
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
        # Check that depth is present
        if slot == "obst":
            check_tree_has_key(t, depth_key)

        # Retrieve node data
        node_keys = [depth_key] + groupby_names if groupby is not None else [depth_key]
        nodes = get_keyed_node_data(tdata, keys=node_keys, tree=key, slot=slot)
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
            birth = nodes.loc[u, depth_key]
            death = nodes.loc[v, depth_key]
            birth_idx = np.searchsorted(timepoints, birth + 1e-6, side="left")  # type: ignore
            if (len(list(t.successors(v))) == 0) and extend_branches:
                death_idx = len(timepoints)
            else:
                death_idx = np.searchsorted(timepoints, death + 1e-6, side="left")  # type: ignore
            g = tuple(nodes.loc[v, groupby_names])  # type: ignore
            group_counts[g][birth_idx:death_idx] += 1

        # Add count for root node
        root = get_root(t)
        root_idx = np.searchsorted(timepoints, nodes.loc[root, depth_key], side="left")  # type: ignore
        g = tuple(nodes.loc[root, groupby_names])  # type: ignore
        group_counts[g][root_idx] += 1

        # Assemble DataFrame
        for g, counts in group_counts.items():
            data = {depth_key: timepoints, "n_extant": counts, "tree": key}
            if groupby is not None:
                for name, value in zip(groupby_names, g, strict=False):
                    data[name] = value
            result_df = pd.DataFrame(data)
            results.append(result_df)

    extant = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    extant["n_extant"] = extant["n_extant"].astype(int)
    if dropna and groupby is not None:
        extant = extant.dropna(subset=groupby_names)
    for col in groupby_names:
        if col in tdata.obs.columns and tdata.obs[col].dtype.name == "category":
            extant[col] = pd.Categorical(extant[col], categories=tdata.obs[col].cat.categories)

    # Store and return
    tdata.uns[key_added] = extant
    if copy:
        return extant
    return None
