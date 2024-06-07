from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
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
        - If `None`, pairwise distance for all observations is stored in `tdata.obsp[key_added]`.
        - If a string, distance to all other observations is `tdata.obs[key_added]`.
        - If a sequence, pairwise distance is stored in `tdata.obsp[key_added]`.
        - If a sequence of pairs, distance between pairs is stored in `tdata.obsp[key_added]`.
    metric
        A known metric’s name or a callable that returns a distance.
    metric_kwds
        Options for the metric.
    key_added
        Distances are stored in `key + '_distances'` unless `key_added` is specified.
    copy
        If True, returns a :class:`np.array` or :class:`scipy.sparse.csr_matrix` with distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`numpy.array` or :class:`scipy.sparse.csr_matrix`.
    Sets the following fields:

    `tdata.obsp[key_added]` : :class:`numpy.array` or :class:`scipy.sparse.csr_matrix` (dtype `float`)
    if `obs` is `None` or a sequence.
    `tdata.obs[key_added]` : :class:`pandas.Series` (dtype `float`) if `obs` is a string.

    """
    # Setup
    metric_fn = DistanceMetric.get_metric(metric, **(metric_kwargs or {}))
    key_added = key_added or key + "_distances"
    if key == "X":
        X = tdata.X
    elif key in tdata.obsm:
        X = tdata.obsm[key]
    else:
        raise ValueError(f"Key {key} not found in `tdata.obsm`.")
    # Compute distances
    if obs is None:
        distances = metric_fn.pairwise(X)
        tdata.obsp[key_added] = distances
    elif isinstance(obs, str):
        idx = tdata.obs_names.get_loc(obs)
        distances = metric_fn.pairwise(X[idx].reshape(1, -1), X).flatten()
        tdata.obs[key_added] = distances
    elif isinstance(obs, Sequence):
        if isinstance(obs[0], str):
            idx = [tdata.obs_names.get_loc(o) for o in obs]
            rows = np.repeat(idx, len(idx))
            cols = np.tile(idx, len(idx))
            select_distances = metric_fn.pairwise(X[idx]).flatten()
        elif isinstance(obs[0], tuple) and len(obs[0]):
            rows = [tdata.obs_names.get_loc(i) for i, _ in obs]
            cols = [tdata.obs_names.get_loc(j) for _, j in obs]
            select_distances = [metric_fn.pairwise(X[i : i + 1, :], X[j : j + 1, :])[0, 0] for i, j in zip(rows, cols)]
        else:
            raise ValueError("Invalid type for parameter `obs`.")
        distances = sp.sparse.csr_matrix((select_distances, (rows, cols)), shape=(len(X), len(X)))
        tdata.obsp[key_added] = distances
    else:
        raise ValueError("Invalid type for parameter `obs`.")
    if copy:
        return distances
