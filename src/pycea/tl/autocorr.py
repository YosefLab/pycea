from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import treedata as td

from pycea.utils import get_keyed_obs_data


def _g_moments(w: sp.sparse.spmatrix | np.ndarray) -> tuple[float, float, float]:
    """
    Compute moments of adjacency matrix for analytic p-value calculation.

    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()
    # s1
    t = w.transpose() + w
    t2 = t.multiply(t) if isinstance(t, sp.sparse.spmatrix) else t * t
    s1 = t2.sum() / 2.0
    # s2
    s2 = (np.array(w.sum(1) + w.sum(0).transpose()) ** 2).sum()
    return s0, s1, s2


def _analytic_pval(score: float, g: sp.sparse.spmatrix | np.ndarray, method: str) -> tuple[float, float]:
    """
    Analytic p-value computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    n2 = n * n
    v_num = n2 * s1 - n * s2 + 3 * s02
    v_den = (n - 1) * (n + 1) * s02

    Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)
    if method == "moran":
        expected = -1.0 / (g.shape[0] - 1)
    elif method == "geary":
        expected = 1.0

    z_norm = (score - expected) / seScore_norm
    p_norm = np.empty(score.shape)
    p_norm[z_norm > 0] = 1 - sp.stats.norm.cdf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = sp.stats.norm.cdf(z_norm[z_norm <= 0])

    return p_norm, Vscore_norm


def autocorr(
    tdata: td.TreeData,
    keys: str | Sequence[str] = None,
    connect_key: str = None,
    method: str = "moran",
    layer: str = None,
    copy: bool = False,
) -> None | pd.DataFrame:
    """Calculate autocorrelation statistic.

    Parameters
    ----------
    tdata
        TreeData object.
    keys
        One or more `obs_keys`, `var_names`, `obsm_keys`, or `obsp_keys` to calculate autocorrelation for. Defaults to all 'var_names'.
    connect_key
        `tdata.obsp` connectivity key specifying set of neighbors for each observation.
    method
        Method to calculate autocorrelation. Options are:

        * 'moran' : `Moran's I autocorrelation <https://en.wikipedia.org/wiki/Moran%27s_I>`.
        * 'geary' : `Geary's C autocorrelation <https://en.wikipedia.org/wiki/Geary%27s_C>`.
    layer
        Name of the TreeData object layer to use. If `None`, `tdata.X` is used.
    copy
        If True, returns a :class:`DataFrame <pandas.DataFrame>` with autocorrelation.

    Returns
    -------
    Returns `None` if `copy=False`, else return :class:`DataFrame <pandas.DataFrame>` columns.

        - `'autocorr'` - Moran's I or Geary's C statistic.
        - `'pval_norm'` - p-value under normality assumption.
        - `'var_norm'` - variance of `'score'` under normality assumption.

    Sets the following fields for each key:

    * `tdata.uns['moranI']` : Above DataFrame for if method is `'moran'`.
    * `tdata.uns['gearyC']` : Above DataFrame for if method is `'geary'`.
    """
    # Setup
    if keys is None:
        keys = tdata.var_names
        if layer is None:
            data = tdata.X
        else:
            data = tdata.layers[layer]
    else:
        if isinstance(keys, str):
            keys = [keys]
        data, _ = get_keyed_obs_data(tdata, keys, layer=layer)
        keys = data.columns
    method_names = {"moran": "moranI", "geary": "gearyC"}
    # Calculate autocorrelation
    if method == "moran":
        corr = sc.metrics.morans_i(tdata.obsp[connect_key], data.T)
        tdata.uns["moranI"] = corr
    elif method == "geary":
        corr = sc.metrics.gearys_c(tdata.obsp[connect_key], data.T)
        tdata.uns["gearyC"] = corr
    else:
        raise ValueError(f"Method {method} not recognized. Must be 'moran' or 'geary'.")
    # Calculate p-value
    p_norm, var_norm = _analytic_pval(corr, tdata.obsp[connect_key], method)
    corr = pd.DataFrame({"autocorr": corr, "pval_norm": p_norm, "var_norm": var_norm}, index=keys)
    tdata.uns[method_names[method]] = corr
    if copy:
        return corr
