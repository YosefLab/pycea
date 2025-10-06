import cycler
import matplotlib.colors as mcolors
import pandas as pd
import treedata as td

from pycea.get.palette import _adjust_colors_for_priors, _colors_from_cmap, palette


def test_colors_from_cmap_string_and_sorting():
    categories = ["a", "b", "c", "d"]
    # Without sort_frequency
    cmap = "tab10"
    colors = _colors_from_cmap(cmap, categories, sort_frequency=False)
    assert isinstance(colors, dict)
    assert set(colors.keys()) == set(categories)
    # Values should be 7-character hex strings
    for hex_ in colors.values():
        assert mcolors.is_color_like(hex_)
    # With sort_frequency
    colors_sf = _colors_from_cmap(cmap, categories, sort_frequency=True)
    assert set(colors_sf.keys()) == set(categories)
    # Ensure at least one color differs between sorted and unsorted
    assert any(colors_sf[k] != colors[k] for k in categories)


def test_colors_from_cycler_and_insufficient_colors():
    categories = ["x", "y", "z", "w", "v"]
    base_colors = ["#ff0000", "#00ff00"]
    cyc = cycler.cycler(color=base_colors)
    colors = _colors_from_cmap(cyc, categories, sort_frequency=False)
    assert set(colors.keys()) == set(categories)
    # Since base list shorter, colors should repeat or sample
    for hex_ in colors.values():
        assert hex_ in base_colors


def test_adjust_colors_for_priors():
    colors = {"a": "#ff0000", "b": "#00ff00"}
    priors = {"a": 1.0, "b": 2.0}
    adjusted = _adjust_colors_for_priors(colors.copy(), priors)
    # For b, normalized_prior=1 -> should be gray
    assert adjusted["b"] == mcolors.to_hex((0.5, 0.5, 0.5))
    # For a, normalized_prior=0.5 -> mix original and gray
    orig_rgb = mcolors.to_rgb("#ff0000")
    expected_a = [(1 - 0.5) * c + 0.5 * g for c, g in zip(orig_rgb, (0.5, 0.5, 0.5), strict=True)]
    assert mcolors.to_hex(expected_a) == adjusted["a"]  # type: ignore


def test_palette_uses_uns_colors():
    tdata = td.TreeData()
    # Create categorical obs
    tdata.obs["fruit"] = pd.Categorical(["apple", "banana", "cherry"], categories=["apple", "banana", "cherry"])
    # Predefine uns
    tdata.uns["fruit_colors"] = ["#111111", "#222222", "#333333"]
    result = palette(tdata, "fruit")
    assert result == {"apple": "#111111", "banana": "#222222", "cherry": "#333333"}


def test_palette_with_custom_and_priors():
    tdata = td.TreeData()
    tdata.obs["animal"] = pd.Categorical(["cat", "dog", "fish"], categories=["cat", "dog", "fish"])
    # Test cmap
    result_cmap = palette(tdata, "animal", cmap="coolwarm", random_state=42)
    assert set(result_cmap.keys()) == {"cat", "dog", "fish"}
    assert all(mcolors.is_color_like(c) for c in result_cmap.values())
    # Test priors adjust
    priors = {"cat": 0.0, "dog": 0.5, "fish": 1.0}
    result_prior = palette(tdata, "animal", cmap="coolwarm", priors=priors)
    # fish should be gray
    assert result_prior["fish"] == mcolors.to_hex((0.5, 0.5, 0.5))
    # Test custom overrides
    custom = {"cat": "#abcdef"}
    result_custom = palette(tdata, "animal", custom=custom)
    assert result_custom["cat"] == "#abcdef"
