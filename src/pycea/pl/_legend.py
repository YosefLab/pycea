from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib as mpl
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.axes import Axes
from matplotlib.font_manager import FontProperties

from ._handlers import HandlerColorbar

_colorbar_proxy = mpatches.Rectangle((0, 0), 1, 1, fc="none", ec="none")


def _cbar_legend(title: str, color_map: Any, norm: Any) -> dict[str, Any]:
    return {
        "title": title,
        "handles": [_colorbar_proxy],
        "labels": [""],
        "handler_map": {_colorbar_proxy: HandlerColorbar(color_map, norm)},
        "handletextpad": 0,
        "handlelength": "dynamic",
        "handleheight": 1,
    }


def _categorical_legend(
    title: str,
    color_map: Mapping[str, str] | None = None,
    size_map: Mapping[str, float] | None = None,
    marker_map: Mapping[str, Any] | None = None,  # new parameter
    type: str = "patch",
) -> dict[str, Any]:
    """
    Create a legend dictionary for categorical data.

    Parameters
    ----------
        title: The legend title.
        color_map: A mapping from category to color.
        size_map: A mapping from category to size.
        marker_map: A mapping from category to marker style.
        type: The legend type; one of "marker", "line", or "patch".

    Returns
    -------
        A dictionary with keys "title", "handles", and "labels".

    Raises
    ------
        ValueError: If the required mapping(s) are not provided or if type is invalid.
    """
    if (color_map is None) and (size_map is None) and (marker_map is None):
        raise ValueError("At least one of mapping must be provided.")
    mapping = color_map or size_map or marker_map
    if mapping is None:
        raise ValueError("Mapping cannot be None. Provide at least one of color_map or size_map.")
    if type == "marker":
        handles = [
            mlines.Line2D(
                [0],
                [0],
                marker=marker_map[cat] if marker_map else "o",  # updated to use marker_map
                color=color_map[cat] if color_map else "black",
                markersize=size_map[cat] / 6 if size_map else 6,
                linestyle="",
                label=str(cat),
            )
            for cat in mapping
        ]
    elif type == "line":
        handles = [
            mlines.Line2D(
                [0],
                [0],
                color=color_map[cat] if color_map else "black",
                linewidth=size_map[cat] if size_map else 2,
                label=str(cat),
            )
            for cat in mapping
        ]
    elif type == "patch":
        if color_map is None:
            raise ValueError("For 'patch' type, color_map must be provided.")
        handles = [mpatches.Patch(color=color_map[cat], label=str(cat)) for cat in color_map.keys()]
    else:
        raise ValueError(f"Invalid type '{type}' for categorical legend. Use 'line' or 'patch'.")
    return {
        "title": title,
        "handles": handles,
        "labels": [str(cat) for cat in mapping.keys()],
    }


def _size_legend(title: str, sizes: Sequence[Any]) -> dict[str, Any]:
    """Create a legend dictionary for point sizes."""
    uniq_sizes = sorted(set(sizes))
    if len(uniq_sizes) > 6:
        # Evenly sample 6 representative sizes (similar to seaborn behaviour)
        idx = np.linspace(0, len(uniq_sizes) - 1, 6, dtype=int)
        uniq_sizes = [uniq_sizes[i] for i in idx]
    handles = [
        mlines.Line2D([], [], linestyle="none", marker="o", color="black", markersize=np.sqrt(s)) for s in uniq_sizes
    ]
    labels = [str(s) for s in uniq_sizes]
    return {"title": title, "handles": handles, "labels": labels}


def _place_legend(
    ax: Axes,
    legend_kwargs: dict[str, Any],
    shared_kwargs: dict[str, Any],
    at_x: float,
    at_y: float,
) -> mlegend.Legend:
    """Place a legend on the axes at the specified position.

    Parameters
    ----------
    ax
        The axes on which to place the legend.
    legend_kwargs
        A dictionary of keyword arguments to pass to the legend.
    shared_kwargs
        A dictionary of shared keyword arguments for all legends.
    at_x
        The x offset in pixels from the top-right corner of the axes.
    at_y
        The y offset in pixels from the top-right corner of the axes.
    """
    handlelength = legend_kwargs.get("handlelength", 2.0)
    fontsize = shared_kwargs.get("fontsize", mpl.rcParams["legend.fontsize"])
    if isinstance(fontsize, str):
        fontsize = FontProperties(size=fontsize).get_size_in_points()
    if handlelength == "dynamic":
        handlelength = 100 / fontsize
    offset_trans = mtransforms.ScaledTranslation(at_x / ax.figure.dpi, at_y / ax.figure.dpi, ax.figure.dpi_scale_trans)
    opts: dict[str, Any] = {
        "handlelength": handlelength,
        "loc": legend_kwargs.get("loc", "upper left"),
        "bbox_to_anchor": (1, 1),
        "bbox_transform": ax.transAxes + offset_trans,
    }
    opts.update({k: v for k, v in legend_kwargs.items() if k not in ("loc", "handlelength")})
    opts.update(shared_kwargs)
    leg: mlegend.Legend = ax.legend(**opts)
    return leg


def _render_legends(
    ax: Axes,
    legends: list[dict],
    anchor_x: float = 1.02,
    spacing: float = 0.02,
    shared_kwargs: dict[str, Any] | None = None,
):
    """Render legends on the given axes, stacking them in columns without overlap.

    Parameters
    ----------
    ax
        The axes on which to render the legends.
    legends
        A dict where keys are legend titles and values are dicts of legend kwargs
        (e.g. 'loc', 'frameon', etc.).
    anchor_x
        The initial x-coordinate (in axes fraction) to anchor the first column.
    spacing
        Spacing (in axes fraction) between legends vertically and between columns horizontally.
    shared_kwargs
        Additional keyword arguments passed to `matplotlib.pyplot.legend`.
    """
    if shared_kwargs is None:
        shared_kwargs = {}
    fig = ax.figure
    fig.canvas.draw()  # make sure transforms are current
    ax_height = ax.bbox.height
    ax_width = ax.bbox.width
    spacing *= ax_height  # convert to pixels

    if not hasattr(ax, "_attrs"):
        ax._attrs = {}  # type: ignore
    x_offset = ax._attrs.get("x_offset", (anchor_x - 1) * ax_width)  # type: ignore
    y_offset = ax._attrs.get("y_offset", 0.0)  # type: ignore
    column_max_width = ax._attrs.get("column_max_width", 0.0)  # type: ignore

    for legend_kwargs in legends:
        # 1) if previous legend exists, add it to the axes
        if ax.get_legend() is not None:
            ax.add_artist(ax.get_legend())
        # 2) place normally to measure its natural size
        legend = _place_legend(ax, legend_kwargs, shared_kwargs, x_offset, y_offset)
        # 3) measure in pixels
        renderer = fig.canvas.get_renderer()  # type: ignore
        bbox_disp = legend.get_window_extent(renderer=renderer)
        width = bbox_disp.width
        height = bbox_disp.height
        # 4) if first in column, initialize max width
        if column_max_width == 0.0:
            column_max_width = width
        # 5) if it overflows vertically, start new column
        if (height - y_offset > ax_height) & (y_offset != 0.0):
            legend.remove()
            x_offset += column_max_width + spacing
            y_offset = 0.0
            # place again and re-measure
            legend = _place_legend(ax, legend_kwargs, shared_kwargs, x_offset, y_offset)
            bbox_disp = legend.get_window_extent(renderer=renderer)
            column_max_width = bbox_disp.width
            height = bbox_disp.height
        # 6) update offsets for next legend
        y_offset -= height + spacing
        if width > column_max_width:
            column_max_width = width
        ax._attrs.update({"y_offset": y_offset})  # type: ignore
        ax._attrs.update({"x_offset": x_offset})  # type: ignore
        ax._attrs.update({"column_max_width": column_max_width})  # type: ignore
