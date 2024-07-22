from __future__ import annotations

import random
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy as sp
import treedata as td
from sklearn.metrics import DistanceMetric

from ._metrics import _Metric, _MetricFn


def distance(
    tdata: td.TreeData,
    key: str,
    obs: str | Sequence[str] | None = None,
    metric: _MetricFn | _Metric = "euclidean",
    metric_kwargs: Mapping | None = None,
    sample_n: int | None = None,
    connect_key: str | None = None,
    random_state: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
):
    """Computes distances between observations.

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
        connectivities in .obsp['{key_added}_connectivities']. Defaults to `key`.
    copy
        If True, returns a :class:`np.array` or :class:`scipy.sparse.csr_matrix` with distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`numpy.array` or :class:`scipy.sparse.csr_matrix`.
    Sets the following fields:

    `tdata.obsp['{key_added}_distances']` : :class:`numpy.array` or :class:`scipy.sparse.csr_matrix` (dtype `float`)
    if `obs` is `None` or a sequence.
    `tdata.obsp['{key_added}_connectivities']` : ::class:`scipy.sparse.csr_matrix` (dtype `float`)
    if distances is sparse.
    `tdata.obs['{key_added}_distances']` : :class:`pandas.Series` (dtype `float`) if `obs` is a string.

    """
    # Setup
    metric_fn = DistanceMetric.get_metric(metric, **(metric_kwargs or {}))
    key_added = key_added or key
    if connect_key is not None:
        if "connectivities" not in connect_key:
            connect_key = f"{connect_key}_connectivities"
    connectivities = None
    if key == "X":
        X = tdata.X
    elif key in tdata.obsm:
        X = tdata.obsm[key]
    else:
        raise ValueError(f"Key {key} not found in `tdata.obsm`.")
    # Get pairs
    pairs = None
    if connect_key is not None:
        if obs is not None:
            warnings.warn("`obs` is ignored when connectivity is specified.", stacklevel=2)
        if connect_key not in tdata.obsp.keys():
            raise ValueError(f"Connectivity key {connect_key} not found in `tdata.obsp`.")
        pairs = list(zip(*tdata.obsp[connect_key].nonzero()))
    else:
        if isinstance(obs, Sequence) and isinstance(obs[0], tuple):
            pairs = [(tdata.obs_names.get_loc(i), tdata.obs_names.get_loc(j)) for i, j in obs]
        elif obs is None and sample_n is not None:
            pairs = [(i, j) for i in range(tdata.n_obs) for j in range(tdata.n_obs)]
    # Compute distances from pairs
    if pairs is not None:
        if sample_n is not None:
            if sample_n > len(pairs):
                raise ValueError("Sample size is larger than the number of pairs.")
            if random_state is not None:
                random.seed(random_state)
            pairs = random.sample(pairs, sample_n)
        distances = [metric_fn.pairwise(X[i : i + 1, :], X[j : j + 1, :])[0, 0] for i, j in pairs]
        distances = sp.sparse.csr_matrix((distances, zip(*pairs)), shape=(tdata.n_obs, tdata.n_obs))
        connectivities = sp.sparse.csr_matrix((np.ones(len(pairs)), zip(*pairs)), shape=(tdata.n_obs, tdata.n_obs))
    # Compute point distances
    elif isinstance(obs, str):
        idx = tdata.obs_names.get_loc(obs)
        distances = metric_fn.pairwise(X[idx].reshape(1, -1), X).flatten()
    # Compute all pairwise distances
    elif obs is None:
        distances = metric_fn.pairwise(X)
    # Compute subset pairwise distances
    elif isinstance(obs, Sequence) and isinstance(obs[0], str):
        idx = [tdata.obs_names.get_loc(o) for o in obs]
        rows = np.repeat(idx, len(idx))
        cols = np.tile(idx, len(idx))
        distances = metric_fn.pairwise(X[idx])
        distances = sp.sparse.csr_matrix((distances.flatten(), (rows, cols)), shape=(tdata.n_obs, tdata.n_obs))
        connectivities = sp.sparse.csr_matrix((np.ones(len(idx) ** 2), (rows, cols)), shape=(tdata.n_obs, tdata.n_obs))
    else:
        raise ValueError("Invalid type for parameter `obs`.")
    # Store distances and connectivities
    if distances.ndim == 2:
        tdata.obsp[f"{key_added}_distances"] = distances
    else:
        tdata.obs[f"{key_added}_distances"] = distances
    if connectivities is not None:
        tdata.obsp[f"{key_added}_connectivities"] = connectivities
    if copy:
        return distances


def compare_distance(
    tdata: td.TreeData,
    dist_keys: str | Sequence[str] | None = None,
    sample_n: int | None = None,
    random_state: int | None = None,
):
    """Get pairwise observation distances.

    Parameters
    ----------
    tdata
        The TreeData object.
    dist_key
        One or more `tdata.obsp` distance keys to compare. Only pairs where all distances are
        available are returned.
    sample_n
        If specified, randomly sample `sample_n` pairs of observations.
    random_state
        Random seed for sampling.

    Returns
    -------
    Returns a :class:`pandas.DataFrame` with the following columns:
    - `obs1` and `obs2` are the observation names.
    - `{dist_key}_distances` are the distances between the observations.

    """
    # Setup
    if random_state is not None:
        random.seed(random_state)
    if isinstance(dist_keys, str):
        dist_keys = [dist_keys]
    dist_keys = [key.replace("_distances", "") for key in dist_keys]
    # Get shared pairs
    shared = None
    dense = False
    sparse = False
    for key in dist_keys:
        if f"{key}_distances" not in tdata.obsp.keys():
            raise ValueError(f"Distance key {key} not found in `tdata.obsp`.")
        if isinstance(tdata.obsp[f"{key}_distances"], sp.sparse.csr_matrix):
            sparse = True
            if shared is None:
                shared = tdata.obsp[f"{key}_connectivities"].copy()
            else:
                shared = shared.multiply(tdata.obsp[f"{key}_connectivities"])
        else:
            dense = True
    if sparse:
        row, col = shared.nonzero()
        pairs = list(zip(row, col))
        if sample_n is not None:
            pairs = random.sample(pairs, sample_n)
    elif dense:
        if sample_n is not None:
            if sample_n > tdata.n_obs**2:
                raise ValueError("Sample size is larger than the number of pairs.")
            pairs = set()
            while len(pairs) < sample_n:
                pairs.add((random.randint(0, tdata.n_obs - 1), random.randint(0, tdata.n_obs - 1)))
        else:
            pairs = [(i, j) for i in range(tdata.n_obs) for j in range(tdata.n_obs)]
    # Get distances
    pair_names = [(tdata.obs_names[i], tdata.obs_names[j]) for i, j in pairs]
    distances = pd.DataFrame(pair_names, columns=["obs1", "obs2"])
    for key in dist_keys:
        distances[f"{key}_distances"] = [tdata.obsp[f"{key}_distances"][i, j] for i, j in pairs]
    return distances
