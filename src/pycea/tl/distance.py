from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast, overload

import numpy as np
import pandas as pd
import scipy as sp
import treedata as td
from sklearn.metrics import DistanceMetric

from pycea.utils import get_keyed_obsm_data

from ._metrics import _Metric, _MetricFn
from ._utils import (
    _check_previous_params,
    _csr_data_mask,
    _format_as_list,
    _format_keys,
    _set_distances_and_connectivities,
    _set_random_state,
)


def _sample_pairs(pairs: Any, sample_n: int | None, n_obs: int) -> Any:
    """Sample pairs"""
    if sample_n is None:
        pass
    elif pairs is None:
        if sample_n > n_obs**2:
            raise ValueError("Sample size is larger than the number of pairs.")
        pairs = set()
        while len(pairs) < sample_n:
            i = random.randint(0, n_obs - 1)
            j = random.randint(0, n_obs - 1)
            pairs.add((i, j))
    else:
        if sample_n > len(pairs):
            raise ValueError("Sample size is larger than the number of pairs.")
        pairs = random.sample(pairs, sample_n)
    return pairs


def _pairwise_with_nans(X, metric_fn):
    """Compute pairwise distances with NaNs"""
    n = X.shape[0]
    distances = np.full((n, n), np.nan)
    # rows without any NaN
    clean_idx = np.where(~np.isnan(X).any(axis=1))[0]
    if clean_idx.size == 0:
        return distances  # all rows have NaNs
    D = metric_fn.pairwise(X[clean_idx], X[clean_idx])
    distances[np.ix_(clean_idx, clean_idx)] = D
    return distances


@overload
def distance(
    tdata: td.TreeData,
    key: str,
    obs: str | int | Sequence[Any] | None = None,
    metric: _MetricFn | _Metric = "euclidean",
    metric_kwds: Mapping | None = None,
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    update: bool = True,
    key_added: str | None = None,
    copy: Literal[True, False] = True,
) -> np.ndarray | sp.sparse.csr_matrix: ...
@overload
def distance(
    tdata: td.TreeData,
    key: str,
    obs: str | int | Sequence[Any] | None = None,
    metric: _MetricFn | _Metric = "euclidean",
    metric_kwds: Mapping | None = None,
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    update: bool = True,
    key_added: str | None = None,
    copy: Literal[True, False] = False,
) -> None: ...
def distance(
    tdata: td.TreeData,
    key: str,
    obs: str | int | Sequence[Any] | None = None,
    metric: _MetricFn | _Metric = "euclidean",
    metric_kwds: Mapping | None = None,
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    update: bool = True,
    key_added: str | None = None,
    copy: Literal[True, False] = False,
) -> None | np.ndarray | sp.sparse.csr_matrix:
    """Computes distances between observations.

    Supports full pairwise distances, distances from a single observation to all others,
    distances within a specified subset, or distances for an explicit list of pairs.
    Distances can be computed using a named metric (e.g. ``"euclidean"``, ``"cosine"``,
    ``"manhattan"``) or a user-supplied callable.

    Parameters
    ----------
    tdata
        The TreeData object.
    key
        Use the indicated key. `'X'` or any `tdata.obsm` key is valid.
    obs
        The observations to use:

        - If `None`, pairwise distance for all observations is stored in `tdata.obsp`.
        - If a string, distance to all other observations is `tdata.obs`.
        - If a sequence, pairwise distance is stored in `tdata.obsp`.
        - If a sequence of pairs, distance between pairs is stored in `tdata.obsp`.

    metric
        A known metric’s name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    sample_n
        If specified, randomly sample `sample_n` pairs of observations.
    connect_key
        If specified, compute distances only between connected observations specified by
        `tdata.obsp[{connect_key}_connectivities]`.
    random_state
        Random seed for sampling.
    key_added
        Distances are stored in `tdata.obsp['{key_added}_distances']` and
        connectivities in `tdata.obsp['{key_added}_connectivities']`. Defaults to `key`.
    update
        If True, updates existing distances instead of overwriting.
    copy
        If True, returns a the distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns distances.

    Sets the following fields:

    * `tdata.obsp['{key_added}_distances']` : :class:`ndarray <numpy.ndarray>`/:class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if `obs` is `None` or a sequence.
        - Distances between observations.
    * `tdata.obsp['{key_added}_connectivities']` : :class:`csr_matrix <scipy.sparse.csr_matrix>` (dtype `float`) if distance is sparse.
        - Connectivity between observations.
    * `tdata.obs['{key_added}_distances']` : :class:`Series <pandas.Series>` (dtype `float`) if `obs` is a string.
        - Distance from specified observation to others.

    Notes
    -----
    * When both ``connect_key`` and ``sample_n`` are provided, sampling is performed
      **within** the connected pairs induced by the connectivity.
    * If you pass a callable metric, it must accept two 1D vectors and return a scalar.

    Examples
    --------
    Calculate pairwise spatial distance between all observations:

    >>> tdata = py.datasets.koblan25()
    >>> py.tl.distance(tdata, key="spatial")

    Calculate spatial distance between closely related observations:

    >>> py.tl.tree_neighbors(tdata, n_neighbors=10, depth_key="time")
    >>> py.tl.distance(tdata, key="spatial", connect_key="tree_connectivities")

    Calculate distance from a single observation to all others:

    >>> py.tl.distance(tdata, key="spatial", obs="M3-1-19")
    """
    # Setup
    _set_random_state(random_state)
    metric_fn = DistanceMetric.get_metric(metric, **(metric_kwds or {}))  # type: ignore
    key_added = key_added or key
    if connect_key not in tdata.obsp.keys():
        connect_key = _format_keys(connect_key, "connectivities")
    single_obs = False
    X = get_keyed_obsm_data(tdata, key)
    if isinstance(X, pd.DataFrame):
        X = X.values
    if update:
        _check_previous_params(
            tdata, {"metric": metric, "metric_kwds": metric_kwds}, key_added, ["neighbors", "distances"]
        )
    # Distance to single observation
    if isinstance(obs, str):
        idx = tdata.obs_names.get_loc(obs)
        distances = metric_fn.pairwise(X[idx].reshape(1, -1), X).flatten()
        single_obs = True
    # Distance given pairs
    elif connect_key or sample_n or (isinstance(obs, Sequence) and isinstance(obs[0], tuple)):
        # Generate pairs
        if connect_key:
            pairs = list(zip(*tdata.obsp[connect_key].nonzero(), strict=True))  # type: ignore
        elif obs:
            pairs = [(tdata.obs_names.get_loc(i), tdata.obs_names.get_loc(j)) for i, j in obs]  # type: ignore
        else:
            pairs = None
        pairs = _sample_pairs(pairs, sample_n, tdata.n_obs)
        # Compute distances
        distances = [metric_fn.pairwise(X[i : i + 1, :], X[j : j + 1, :])[0, 0] for i, j in pairs]
        distances = sp.sparse.csr_matrix(
            (distances, tuple(map(list, zip(*pairs, strict=False)))), shape=(tdata.n_obs, tdata.n_obs)
        )
    # Distance given indices
    elif obs is None or (isinstance(obs, Sequence) and isinstance(obs[0], str)):
        if obs is None:
            distances = _pairwise_with_nans(X, metric_fn) if np.isnan(X).any() else metric_fn.pairwise(X)
        else:
            idx = [tdata.obs_names.get_loc(o) for o in obs]
            distances = metric_fn.pairwise(X[idx])  # type: ignore
            distances = sp.sparse.csr_matrix(
                (distances.flatten(), (np.repeat(idx, len(idx)), np.tile(idx, len(idx)))),  # type: ignore
                shape=(tdata.n_obs, tdata.n_obs),
            )
    else:
        raise ValueError("Invalid type for parameter `obs`.")
    # Update tdata
    if single_obs:
        tdata.obs[f"{key_added}_distances"] = distances
    else:
        param_dict = {
            "connectivities_key": f"{key_added}_connectivities",
            "distances_key": f"{key_added}_distances",
            "params": {
                "metric": metric,
                "random_state": random_state,
                "metric_kwds": metric_kwds,
            },
        }
        tdata.uns[f"{key_added}_distances"] = param_dict
        if isinstance(distances, np.ndarray):
            _set_distances_and_connectivities(tdata, key_added, distances, None, update)
        else:
            _set_distances_and_connectivities(tdata, key_added, distances, _csr_data_mask(distances), update)
    if copy:
        return distances


def _determine_shared_pairs(tdata, dist_keys):
    """Determine shared pairs of observations"""
    shared, dense = None, True
    for dist_key in dist_keys:
        if dist_key not in tdata.obsp.keys():
            raise ValueError(f"Distance key {dist_keys} not found in `tdata.obsp`.")
        dist_matrix = tdata.obsp[dist_key]
        if isinstance(dist_matrix, sp.sparse.csr_matrix):
            dense = False
            mask = _csr_data_mask(dist_matrix)
            shared = mask.copy() if shared is None else shared.multiply(mask)
    return shared, dense


def _get_group_indices(tdata, groupby, groups) -> tuple[list[np.ndarray], list[str]]:
    """Get indices for groups"""
    if groupby is not None:
        if groups is None:
            groups = tdata.obs[groupby].unique()
        indices = [np.where(tdata.obs[groupby] == group)[0] for group in groups]
    else:
        indices = [np.arange(tdata.n_obs)]
    return indices, groups


def _get_pairs_for_group(idx, shared, dense, sample_n):
    """Get pairs for group given a shared boolean matrix"""
    if dense:
        if sample_n is not None:
            if sample_n > len(idx) ** 2:
                raise ValueError("Sample size is larger than the number of pairs.")
            pairs = set()
            while len(pairs) < sample_n:
                pair = (np.random.choice(idx), np.random.choice(idx))
                pairs.add(pair)
        else:
            pairs = [(i, j) for i in idx for j in idx]
    else:
        row, col = shared[idx, :][:, idx].nonzero()
        pairs = list(zip(idx[row], idx[col], strict=False))
        if sample_n is not None:
            if sample_n > len(pairs):
                raise ValueError("Sample size is larger than the number of pairs.")
            pairs = random.sample(pairs, sample_n)
    return pairs


def compare_distance(
    tdata: td.TreeData,
    dist_keys: str | Sequence[str] | None = None,
    sample_n: int | None = None,
    groupby: str | None = None,
    groups: str | Sequence[str] | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Get pairwise observation distances.

    This function gathers distances between the same observation pairs from one or
    more entries in ``tdata.obsp`` and returns them side-by-side in a tidy
    :class:`pandas.DataFrame`. Only pairs for which **all** requested distance
    matrices have defined values are included. Optionally, comparisons can be
    restricted within groups and/or randomly subsampled.

    Parameters
    ----------
    tdata
        The TreeData object.
    dist_key
        One or more `tdata.obsp` distance keys to compare. Only pairs where all distances are
        available are returned.
    sample_n
        If specified, randomly sample `sample_n` pairs of observations. If groupby is specified,
        the sample is taken within each group.
    groupby
        If specified, only compare distances within groups.
    groups
        Restrict the comparison to these groups.
    random_state
        Random seed for sampling.

    Returns
    -------
    Returns a :class:`DataFrame <pandas.DataFrame>` with the following columns:

    * `obs1` and `obs2` are the observation names.
    * `{dist_key}_distances` are the distances between the observations.

    Examples
    --------
    Compare spatial and tree distances for 1000 random pairs of observations:

    >>> tdata = py.datasets.koblan25()
    >>> py.tl.distance(tdata, key="spatial", sample_n=1000)
    >>> py.tl.tree_distance(tdata, key="tree", connect_key="spatial_connectivities")
    >>> df = py.tl.compare_distance(tdata, dist_keys=["spatial_distances", "tree_distances"])

    """
    # Setup
    _set_random_state(random_state)
    groups = _format_as_list(groups)
    dist_keys = _format_keys(dist_keys, "distances")
    dist_keys = cast(Sequence[str], _format_as_list(dist_keys))
    # Get set of shared pairs
    shared, dense = _determine_shared_pairs(tdata, dist_keys)
    # Get distances for each group
    indices, groups = _get_group_indices(tdata, groupby, groups)
    distances = []
    for i, idx in enumerate(indices):
        # Get pairs for group
        pairs = _get_pairs_for_group(idx, shared, dense, sample_n)
        # Get distances for pairs
        pair_names = [(tdata.obs_names[i], tdata.obs_names[j]) for i, j in pairs]
        group_distances = pd.DataFrame(pair_names, columns=["obs1", "obs2"])
        for dist_key in dist_keys:
            group_distances[dist_key] = [tdata.obsp[dist_key][i, j] for i, j in pairs]
        if groupby is not None:
            group_distances[groupby] = groups[i]
        distances.append(group_distances)
    distances = pd.concat(distances)
    return distances
