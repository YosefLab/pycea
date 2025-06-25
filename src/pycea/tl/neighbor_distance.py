from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import scipy as sp
import treedata as td

from ._utils import _csr_data_mask, _format_keys


def _get_agg_func(method):
    """Returns aggregation function."""
    agg_funcs = {"mean": np.mean, "max": np.max, "min": np.min, "median": np.median}
    if method in agg_funcs:
        return agg_funcs[method]
    elif callable(method):
        return method
    else:
        raise ValueError(f"Invalid method: {method}")


def _assert_distance_specified(dist, mask):
    """Asserts that distance is specified for where connected"""
    if isinstance(dist, sp.sparse.csr_matrix):
        dist_mask = _csr_data_mask(dist)
        if not dist_mask[mask].sum() == mask.sum():
            raise ValueError("Distance must be specified for all connected observations.")
    return


def neighbor_distance(
    tdata: td.TreeData,
    connect_key: str | None = None,
    dist_key: str | None = None,
    method: str | Callable = "mean",
    key_added: str = "neighbor_distances",
    copy: bool = False,
) -> None | pd.Series:
    """Aggregates distance to neighboring observations.

    Parameters
    ----------
    tdata
        The TreeData object.
    connect_key
        `tdata.obsp` connectivity key specifying set of neighbors for each observation.
    dist_key
        `tdata.obsp` distances key specifying distances between observations.
    method
        Method to calculate neighbor distances:

        * 'mean' : The mean distance to neighboring observations.
        * 'median' : The median distance to neighboring observations.
        * 'min' : The minimum distance to neighboring observations.
        * 'max' : The maximum distance to neighboring observations.
        * Any function that takes a list of values and returns a single value.

    key_added
        `tdata.obs` key to store neighbor distances.
    copy
        If True, returns a :class:`Series <pandas.Series>` with neighbor distances.

    Returns
    -------
    Returns `None` if `copy=False`, else returns a :class:`Series <pandas.Series>`.

    Sets the following fields:

    * `tdata.obs[key_added]` : :class:`Series <pandas.Series>` (dtype `float`)
        - Neighbor distances for each observation.
    """
    # Setup
    if connect_key is None:
        raise ValueError("connect_key must be specified.")
    if dist_key is None:
        raise ValueError("dist_key must be specified.")
    _format_keys(connect_key, "connectivities")
    _format_keys(dist_key, "distances")
    agg_func = _get_agg_func(method)
    mask = tdata.obsp[connect_key] > 0
    dist = tdata.obsp[dist_key]
    _assert_distance_specified(dist, mask)
    # Calculate neighbor distances
    agg_dist = []
    for i in range(dist.shape[0]):
        if isinstance(mask, sp.sparse.csr_matrix):
            indices = mask[i].indices
        else:
            indices = np.nonzero(mask[i])[0]
        row_dist = dist[i, indices]
        if row_dist.size > 0:
            agg_dist.append(agg_func(row_dist))
        else:
            agg_dist.append(np.nan)
    # Update tdata and return
    tdata.obs[key_added] = agg_dist
    if copy:
        return tdata.obs[key_added]
