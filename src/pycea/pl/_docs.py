"""Shared docstrings for plotting functions."""

from __future__ import annotations

from textwrap import dedent


def _doc_params(**kwds):
    r"""Docstrings should start with ``\\`` in the first line for proper formatting"""

    def dec(obj):
        obj.__orig_doc__ = obj.__doc__
        obj.__doc__ = dedent(obj.__doc__).format_map(kwds)
        return obj

    return dec


doc_common_plot_args = """\
cmap
    Color map to use for continous variables. Can be a name or a
    :class:`~matplotlib.colors.Colormap` instance (e.g. `"magma`", `"viridis"`
    or `mpl.cm.cividis`), see :func:`~matplotlib.cm.get_cmap`.
    If `None`, the value of `mpl.rcParams["image.cmap"]` is used.
    The default `cmap` can be set using :func:`~scanpy.set_figure_params`.
palette
    Colors to use for plotting categorical annotation groups.
    The palette can be a valid :class:`~matplotlib.colors.ListedColormap` name
    (`'Set2'`, `'tab20'`, …), a :class:`~cycler.Cycler` object, a dict mapping
    categories to colors, or a sequence of colors. Colors must be valid to
    matplotlib. (see :func:`~matplotlib.colors.is_color_like`).
    If `None`, `mpl.rcParams["axes.prop_cycle"]` is used unless the categorical
    variable already has colors stored in `tdata.uns["{var}_colors"]`.
    If provided, values of `tdata.uns["{var}_colors"]` will be set.
vmax
    The maximum value for the colormap.
vmin
    The minimum value for the colormap.
ax
    A matplotlib axes object. If `None`, a new figure and axes will be created.
"""
