from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import treedata as td
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from pycea.tl.ancestral_linkage import _symmetrize_matrix

_STATS_SUFFIX = "_linkage_stats"
_PARAMS_SUFFIX = "_linkage_params"


def _cluster_order(matrix: pd.DataFrame, method: str, negate: bool) -> list[int]:
    """Optimal leaf order for a square matrix via SciPy hierarchical clustering.

    Clustering needs a distance (larger = less related). When ``negate`` is `True` the
    matrix holds *similarities* (larger = more related), so it is negated via ``max - value``;
    otherwise it already holds *dissimilarities* and is used directly (shifted to be
    non-negative). The result is symmetrized so it can be condensed with
    :func:`scipy.spatial.distance.squareform`, then clustered and reordered with
    :func:`scipy.cluster.hierarchy.optimal_leaf_ordering`.
    """
    arr = matrix.fillna(0).to_numpy(dtype=float)
    dist = (float(np.nanmax(arr)) - arr) if negate else (arr - float(np.nanmin(arr)))
    dist = (dist + dist.T) / 2  # enforce exact symmetry required by squareform
    np.fill_diagonal(dist, 0.0)
    condensed = ssd.squareform(dist, checks=False)
    linkage = sch.linkage(condensed, method=method)
    linkage = sch.optimal_leaf_ordering(linkage, condensed)
    return list(sch.leaves_list(linkage))


def ancestral_linkage(
    tdata: td.TreeData,
    groupby: str | None = None,
    symmetrize: Literal["mean", "max", "min", False, None] = None,
    normalize: bool | None = None,
    data: pd.DataFrame | None = None,
    tree: str | None = None,
    cluster: str | None = "average",
    cluster_mode: Literal["similarity", "dissimilarity", None] = None,
    order: Sequence[str] | None = None,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = 0,
    xticklabels: bool = True,
    yticklabels: bool = True,
    labelsize: float | None = None,
    cbar: bool = True,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """
    Plot ancestral linkage as a clustered heatmap.

    Displays the category × category linkage computed by
    :func:`pycea.tl.ancestral_linkage` as a heatmap, with rows and columns ordered by
    hierarchical clustering. The matrix is rebuilt from the long-form statistics table so
    that `symmetrize` and `normalize` can be chosen independently of how
    :func:`pycea.tl.ancestral_linkage` was called.

    Parameters
    ----------
    tdata
        The TreeData object.
    groupby
        The ``groupby`` / ``key_added`` used when running :func:`pycea.tl.ancestral_linkage`;
        statistics are read from ``tdata.uns[f"{groupby}_linkage_stats"]``. If `None`, it is
        inferred from ``tdata.uns`` (requires exactly one ``*_linkage_stats`` entry).
    symmetrize
        How to symmetrize the directional (source → target) matrix before plotting:
        `'mean'`, `'max'`, `'min'`, or `False` to leave it asymmetric. If `None` (default),
        use the value recorded by :func:`pycea.tl.ancestral_linkage` in ``tdata.uns``
        (defaults to `False` when it cannot be determined).
    normalize
        If `True`, color by the enrichment ``value - permuted_value`` (centered at 0); if
        `False`, color by the raw linkage ``value``. If `None` (default), use the value
        recorded by :func:`pycea.tl.ancestral_linkage` in ``tdata.uns`` (defaults to `True`
        when it cannot be determined).
    data
        Long-form statistics to plot. Uses ``tdata.uns[f"{groupby}_linkage_stats"]`` if `None`.
    tree
        If the statistics are per-tree (computed with ``by_tree=True``), the tree to plot.
        If `None`, values are averaged across trees.
    cluster
        Linkage method (e.g. `'ward'`, `'average'`) used to order rows/columns by
        hierarchical clustering. If `None`, the categories are left in their original order.
        Ignored if `order` is given.
    cluster_mode
        Whether the matrix values are a `'similarity'` (larger = more related, negated to a
        distance before clustering) or a `'dissimilarity'` (larger = less related, used
        directly). If `None` (default), inferred from the recorded `metric`:
        `'dissimilarity'` for `metric='path'` and `'similarity'` otherwise.
    order
        Explicit order of categories (rows and columns). Overrides `cluster`.
    cmap
        Colormap for the heatmap. If `None`, defaults to `'RdBu_r'` when `normalize=True`
        (diverging) and `'viridis'` when `normalize=False` (sequential).
    vmin, vmax
        Lower / upper bounds of the color scale. If `None`, derived from the largest
        absolute off-diagonal deviation from `center` (diverging) or from the off-diagonal
        min / max (sequential).
    center
        The value at which to center the colormap. As with :func:`seaborn.heatmap`, this is
        only applied to the diverging (`normalize=True`) heatmap via
        :class:`~matplotlib.colors.TwoSlopeNorm`; it is ignored for the sequential
        (`normalize=False`) heatmap. Set to `None` to disable centering.
    xticklabels, yticklabels
        Whether to draw the column / row tick labels.
    labelsize
        Font size for the tick labels.
    cbar
        Whether to draw a colorbar.
    ax
        Axes on which to draw the plot. Creates new axes when `None`.
    kwargs
        Additional keyword arguments for :meth:`~matplotlib.axes.Axes.pcolormesh`.

    Returns
    -------
    ax - Axes containing the heatmap.
    """
    # ── load statistics ───────────────────────────────────────────────────────
    if data is not None:
        stats = data.copy()
    else:
        if groupby is None:
            candidates = [k[: -len(_STATS_SUFFIX)] for k in tdata.uns if k.endswith(_STATS_SUFFIX)]
            if not candidates:
                raise KeyError(f"No {'*' + _STATS_SUFFIX!r} found in tdata.uns. Run pycea.tl.ancestral_linkage first.")
            if len(candidates) > 1:
                raise ValueError(f"Multiple linkage results found: {sorted(candidates)}. Specify groupby.")
            groupby = candidates[0]
        stats = tdata.uns.get(f"{groupby}{_STATS_SUFFIX}")
        if stats is None:
            raise KeyError(
                f"{groupby + _STATS_SUFFIX!r} not found in tdata.uns. "
                f"Run pycea.tl.ancestral_linkage(..., key_added={groupby!r}) first."
            )
        stats = stats.copy()

    if "tree" in stats.columns and tree is not None:
        stats = stats[stats["tree"] == tree]
        if stats.empty:
            raise ValueError(f"tree {tree!r} not found in the linkage statistics.")

    # ── resolve normalize / symmetrize from the parameters tl recorded ──────────
    # When left as None, mirror how pycea.tl.ancestral_linkage was called so the plot
    # matches the stored linkage matrix instead of silently re-normalizing.
    params = tdata.uns.get(f"{groupby}{_PARAMS_SUFFIX}", {}) if groupby is not None else {}
    if normalize is None:
        normalize = bool(params.get("normalize", True))
    if symmetrize is None:
        symmetrize = params.get("symmetrize", False)
    if cluster_mode is None:
        cluster_mode = "dissimilarity" if params.get("metric") == "path" else "similarity"
    if cluster_mode not in ("similarity", "dissimilarity"):
        raise ValueError(f"cluster_mode must be 'similarity', 'dissimilarity', or None; got {cluster_mode!r}.")

    # ── select the quantity to plot ───────────────────────────────────────────
    if normalize:
        if "permuted_value" not in stats.columns:
            raise ValueError(
                "normalize=True requires a 'permuted_value' column. Recompute with "
                "pycea.tl.ancestral_linkage or pass normalize=False."
            )
        values = stats["value"] - stats["permuted_value"]
    else:
        values = stats["value"]

    stats = stats.assign(_value=values.to_numpy())
    matrix = stats.pivot_table(index="source", columns="target", values="_value", aggfunc="mean")

    # square & align (source and target share the same category set in pairwise mode)
    cats = matrix.index.union(matrix.columns)
    matrix = matrix.reindex(index=cats, columns=cats)

    # ── symmetrize ────────────────────────────────────────────────────────────
    if symmetrize:
        matrix = _symmetrize_matrix(matrix, symmetrize)

    # ── order rows / columns ──────────────────────────────────────────────────
    if order is not None:
        missing = [c for c in order if c not in matrix.index]
        if missing:
            raise ValueError(f"order contains categories not in the linkage matrix: {missing}.")
        matrix = matrix.loc[list(order), list(order)]
    elif cluster is not None and matrix.shape[0] > 2:
        idx = _cluster_order(matrix, method=cluster, negate=cluster_mode == "similarity")
        matrix = matrix.iloc[idx, idx]

    # ── color scale ────────────────────────────────────────────────────────────
    if cmap is None:
        cmap = "RdBu_r" if normalize else "viridis"

    arr = matrix.to_numpy(dtype=float)
    offdiag = arr.copy()
    np.fill_diagonal(offdiag, np.nan)
    finite = offdiag[np.isfinite(offdiag)]
    if finite.size == 0:  # fall back to the full matrix (e.g. degenerate 1-2 category case)
        finite = arr[np.isfinite(arr)]

    # ``center`` produces a diverging scale and, mirroring seaborn, is only applied to the
    # (normalized) diverging heatmap; the sequential heatmap uses a plain linear scale.
    if normalize and center is not None:
        if vmin is None or vmax is None:
            span = float(np.max(np.abs(finite - center))) if finite.size else 1.0
            if not np.isfinite(span) or span == 0:
                span = 1.0
            vmin = center - span if vmin is None else vmin
            vmax = center + span if vmax is None else vmax
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax) if vmin < center < vmax else Normalize(vmin, vmax)
    else:
        if vmin is None:
            vmin = float(np.min(finite)) if finite.size else 0.0
        if vmax is None:
            vmax = float(np.max(finite)) if finite.size else 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

    # ── draw ──────────────────────────────────────────────────────────────────
    if ax is None:
        _, ax = plt.subplots()
    ax = cast(Axes, ax)

    mesh = ax.pcolormesh(matrix.to_numpy(dtype=float), cmap=cmap, norm=norm, **kwargs)
    n = matrix.shape[0]
    if xticklabels:
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_xticklabels(matrix.columns, rotation=90)
    else:
        ax.set_xticks([])
    if yticklabels:
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels(matrix.index)
    else:
        ax.set_yticks([])
    # Drop the tick marks and pull the labels in close to the axes.
    tick_kwargs: dict[str, Any] = {"length": 0, "pad": 3}
    if labelsize is not None:
        tick_kwargs["labelsize"] = labelsize
    ax.tick_params(axis="both", **tick_kwargs)
    ax.set_aspect("equal")
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()  # first row at the top, like a conventional heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")

    if cbar:
        ax.get_figure().colorbar(mesh, ax=ax)

    return ax
