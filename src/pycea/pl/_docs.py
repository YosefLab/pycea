"""Shared docstrings for plotting functions."""

from __future__ import annotations

doc_common_plot_args = """\
color_map
    Color map to use for continous variables. Can be a name or a
    :class:`~matplotlib.colors.Colormap` instance (e.g. `"magma`", `"viridis"`
    or `mpl.cm.cividis`), see :func:`~matplotlib.cm.get_cmap`.
    If `None`, the value of `mpl.rcParams["image.cmap"]` is used.
    The default `color_map` can be set using :func:`~scanpy.set_figure_params`.
palette
    Colors to use for plotting categorical annotation groups.
    The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
    (`'Set2'`, `'tab20'`, …), a :class:`~cycler.Cycler` object, a dict mapping
    categories to colors, or a sequence of colors. Colors must be valid to
    matplotlib. (see :func:`~matplotlib.colors.is_color_like`).
    If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
    variable already has colors stored in `tdata.uns["{var}_colors"]`.
    If provided, values of `tdata.uns["{var}_colors"]` will be set.
ax
    A matplotlib axes object. If `None`, a new figure and axes will be created.
"""
