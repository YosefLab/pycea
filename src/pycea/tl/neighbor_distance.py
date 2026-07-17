from __future__ import annotations

from typing import Literal, overload

import numpy as np
import pandas as pd
import scipy as sp
import treedata as td

from ._aggregators import _Aggregator, _AggregatorFn, _get_aggregator
from ._utils import _csr_data_mask, _format_keys


def _assert_distance_specified(dist, mask):
    """Asserts that distance is specified for where connected"""
    if isinstance(dist, sp.sparse.csr_matrix):
        dist_mask = _csr_data_mask(dist)
        if not dist_mask[mask].sum() == mask.sum():
            raise ValueError("Distance must be specified for all connected observations.")
    return


@overload
def neighbor_distance(
    tdata: td.TreeData,
    connect_key: str | None = None,
    dist_key: str | None = None,
    method: _AggregatorFn | _Aggregator = "mean",
    key_added: str = "neighbor_distances",
    copy: Literal[True, False] = True,
) -> pd.Series: ...
@overload
def neighbor_distance(
    tdata: td.TreeData,
    connect_key: str | None = None,
    dist_key: str | None = None,
    method: _AggregatorFn | _Aggregator = "mean",
    key_added: str = "neighbor_distances",
    copy: Literal[True, False] = False,
) -> None: ...
def neighbor_distance(
    tdata: td.TreeData,
    connect_key: str | None = None,
    dist_key: str | None = None,
    method: _AggregatorFn | _Aggregator = "mean",
    key_added: str = "neighbor_distances",
    copy: Literal[True, False] = False,
) -> None | pd.Series:
    r"""Aggregates distance to neighboring observations.

    For each observation :math:`i`, this function collects the distances
    :math:`\\{ D_{ij} : j \in \mathcal{N}(i) \\}` to its neighbors (as defined by a
    binary/weighted connectivity in ``tdata.obsp[connect_key]``) and reduces them to a
    single value via an aggregation function :math:`g`:

    .. math::

        d_i = g\big( \{ D_{ij} : j \in \mathcal{N}(i) \} \big)

    The aggregator :math:`g` can be the mean, median, min, max, or a user-supplied
    callable. If an observation has no neighbors, the result for that observation is
    ``NaN``.

    Parameters
    ----------
    tdata
        The TreeData object.
    connect_key
        `tdata.obsp` connectivity key specifying set of neighbors for each observation.
    dist_key
        `tdata.obsp` distances key specifying distances between observations.
    method
        Aggregation function used to calculate neighbor distances.
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

    Examples
    --------
    Calculate mean spatial distance to tree neighbors:

    >>> tdata = py.datasets.koblan25()
    >>> py.tl.tree_neighbors(tdata, n_neighbors=5, depth_key="time")
    >>> py.tl.distance(tdata, key="spatial", connect_key="tree_connectivities")
    >>> py.tl.neighbor_distance(tdata, dist_key="spatial_distances", connect_key="tree_connectivities", method="mean")

    """
    # Setup
    if connect_key is None:
        raise ValueError("connect_key must be specified.")
    if dist_key is None:
        raise ValueError("dist_key must be specified.")
    if connect_key not in tdata.obsp.keys():
        _format_keys(connect_key, "connectivities")
    if dist_key not in tdata.obsp.keys():
        _format_keys(dist_key, "distances")
    agg_func = _get_aggregator(method)
    mask = tdata.obsp[connect_key] > 0
    dist = tdata.obsp[dist_key]
    _assert_distance_specified(dist, mask)
    # Calculate neighbor distances
    agg_dist = []
    for i in range(dist.shape[0]):  # type: ignore
        if sp.sparse.issparse(mask):
            indices = mask[i].indices
        else:
            indices = np.nonzero(mask[i])[0]
        if sp.sparse.issparse(dist):
            row_dist = dist[i, indices].toarray().ravel()
        else:
            row_dist = dist[i, indices]
        if row_dist.size > 0:
            agg_dist.append(agg_func(row_dist))
        else:
            agg_dist.append(np.nan)
    # Update tdata and return
    tdata.obs[key_added] = agg_dist
    if copy:
        return tdata.obs[key_added]
