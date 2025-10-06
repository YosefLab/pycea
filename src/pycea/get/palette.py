from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cycler
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import treedata as td

from pycea.pl._utils import _get_categorical_colors
from pycea.utils import get_keyed_obs_data


def _colors_from_cmap(
    cmap: str | mcolors.Colormap | cycler.Cycler, categories: list, sort_frequency: bool
) -> dict[Any, Any]:
    """Generates a color mapping from a colormap or cycler for a list of categories."""
    n = len(categories)
    # If cmap is a string or a Matplotlib colormap, sample uniformly from it.
    if isinstance(cmap, str | mcolors.Colormap):
        if isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap)
        else:
            cmap_obj = cmap
        if sort_frequency:
            # Square-root scaling: most frequent (i=0) gets position 0, while last gets position 1.
            indices = [(i / max(n - 1, 1)) ** 0.5 for i in range(n)]
        else:
            indices = [i / max(n - 1, 1) for i in range(n)]
        sampled_colors = [mcolors.to_hex(cmap_obj(i)) for i in indices]
        colors = dict(zip(categories, sampled_colors, strict=True))
    # If cmap is a Cycler, extract its colors and sample uniformly.
    elif isinstance(cmap, cycler.Cycler):
        color_list = list(cmap.by_key().get("color", []))
        if len(color_list) < n:
            indices = [int(i * len(color_list) / n) for i in range(n)]
            sampled_colors = [color_list[i] for i in indices]
        else:
            sampled_colors = color_list[:n]
        colors = dict(zip(categories, sampled_colors, strict=True))
    else:
        raise ValueError("Unsupported type for cmap; must be str, Colormap, or Cycler.")
    return colors


def _adjust_colors_for_priors(colors: dict[Any, Any], priors: Mapping[Any, float]) -> dict[Any, Any]:
    """Adjusts colors based on prior probabilities."""
    max_prior = max(priors.values())
    for category, prior in priors.items():
        if category in colors:
            normalized_prior = prior / max_prior
            original_rgb = mcolors.to_rgb(colors[category])
            gray = (0.5, 0.5, 0.5)
            adjusted_rgb = tuple(
                (1 - normalized_prior) * c + normalized_prior * g for c, g in zip(original_rgb, gray, strict=False)
            )
            colors[category] = mcolors.to_hex(adjusted_rgb)  # type: ignore
    return colors


def palette(
    tdata: td.TreeData,
    key: str,
    custom: Mapping[Any, Any] | None = None,
    cmap: str | mcolors.Colormap | cycler.Cycler | None = None,
    priors: Mapping[Any, float] | None = None,
    sort: str | None = None,
    random_state: int | None = None,
) -> dict[Any, Any]:
    """
    Get color palette for a given key.

    Parameters
    ----------
    tdata
        The `treedata.TreeData` object.
    key
        A key from `obs_keys`, `obsm_keys`, or `obsp_keys` to generate a color palette for.
    custom
        A dictionary mapping specific values to colors (e.g., `{"category1": "red"}`).
    cmap
        A colormap to use for generating colors. If `None`, `mpl.rcParams["axes.prop_cycle"]` is used.
    priors
        A dictionary mapping values to their prior probabilities. Values with higher priors
        will be assigned less saturated colors.
    sort
        How to sort the categories. One of `None`, `"alphabetical"`, `"frequency"`, or `"random"`.
        If `None`, existing categories order is used or natural sorting.
    random_state
        Random seed for sampling. Only used if `sort` is `"random"`.

    Returns
    -------
    palette - Color palette for the given key
    """
    # Setup
    tdata._sanitize()
    if not isinstance(key, str):
        raise TypeError("Key must be a string.")
    if random_state is not None:
        np.random.seed(random_state)
    data, is_array, is_square = get_keyed_obs_data(tdata, [key], sort=sort)
    if is_array:
        key = data.columns[0]
    if len(data.select_dtypes(exclude="category").columns) > 0:
        raise ValueError(f"Data for key '{key}' cannot be converted to categorical.")
    # Palette exists
    if not any([custom, cmap, priors, sort]) and f"{key}_colors" in tdata.uns:
        return dict(zip(data[key].cat.categories, tdata.uns[f"{key}_colors"], strict=True))
    # Create palette from cmap
    if cmap is not None:
        colors = _colors_from_cmap(cmap, data[key].cat.categories.tolist(), False)
    # Create palette from defaults
    else:
        colors = _get_categorical_colors(tdata, key, data[key], save=False)
    if priors is not None:
        colors = _adjust_colors_for_priors(colors, priors)
    if custom is not None:
        colors.update(custom)
    return colors
