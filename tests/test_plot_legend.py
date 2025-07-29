import pytest

# ruff: noqa
import matplotlib.colors as mcolors
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from pycea.pl._handlers import HandlerColorbar
from pycea.pl._legend import _categorical_legend, _cbar_legend, _place_legend, _render_legends, _size_legend


def test_cbar_legend():
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=0, vmax=1)
    legend = _cbar_legend("title", cmap, norm)
    assert legend["title"] == "title"
    handle = legend["handles"][0]
    assert isinstance(handle, mpatches.Patch)
    assert legend["labels"] == [""]
    handler_map = legend["handler_map"]
    assert handle in handler_map
    assert isinstance(handler_map[handle], HandlerColorbar)
    assert legend["handletextpad"] == 0
    assert legend["handlelength"] == "dynamic"
    assert legend["handleheight"] == 1


@pytest.mark.parametrize(
    "type_, color_map, size_map, marker_map, expected_type",
    [
        ("patch", {"a": "red", "b": "blue"}, None, None, mpatches.Patch),
        ("line", {"a": "red", "b": "blue"}, {"a": 1.0, "b": 2.0}, None, mlines.Line2D),
        ("marker", {"a": "red", "b": "blue"}, {"a": 9.0, "b": 16.0}, {"a": "s", "b": "o"}, mlines.Line2D),
    ],
)
def test_categorical_legend(type_, color_map, size_map, marker_map, expected_type):
    legend = _categorical_legend("t", color_map=color_map, size_map=size_map, marker_map=marker_map, type=type_)
    assert legend["title"] == "t"
    handles = legend["handles"]
    labels = legend["labels"]
    mapping = color_map or size_map or marker_map
    assert labels == [str(k) for k in mapping.keys()]
    assert all(isinstance(h, expected_type) for h in handles)


def test_categorical_legend_errors():
    with pytest.raises(ValueError):
        _categorical_legend("t")
    with pytest.raises(ValueError):
        # patch type requires color_map
        _categorical_legend("t", color_map=None, size_map={"a": 1}, marker_map=None, type="patch")
    with pytest.raises(ValueError):
        _categorical_legend("t", color_map={"a": "x"}, type="invalid")


def test_size_legend():
    sizes = [1, 4, 9]
    legend = _size_legend("s", sizes)
    assert legend["title"] == "s"
    handles = legend["handles"]
    labels = legend["labels"]
    assert labels == ["1", "4", "9"]
    assert all(isinstance(h, mlines.Line2D) for h in handles)
    # too many unique sizes
    sizes = list(range(10))
    legend2 = _size_legend("s2", sizes)
    assert len(legend2["handles"]) == 6
    assert len(legend2["labels"]) == 6


def test_place_legend_default_and_expand():
    fig, ax = plt.subplots()
    l1 = mlines.Line2D([], [], color="red", label="a")
    legend_kwargs = {"title": "t1", "handles": [l1], "labels": ["a"]}
    shared_kwargs = {"fontsize": 10}
    leg = _place_legend(ax, legend_kwargs, shared_kwargs, at_x=0.5, at_y=0.5)
    assert isinstance(leg, mlegend.Legend)
    # Expand case
    fig2, ax2 = plt.subplots()
    p1 = mpatches.Patch(color="blue", label="b")
    legend_kwargs2 = {"title": "t2", "handles": [p1], "labels": ["b"], "handlelength": 2}
    shared_kwargs2 = {"fontsize": 12}
    leg2 = _place_legend(ax2, legend_kwargs2, shared_kwargs2, at_x=0.2, at_y=0.8, box_width=0.3, expand=True)
    assert isinstance(leg2, mlegend.Legend)
    assert hasattr(leg2, "_bbox_to_anchor")


def test_render_legends():
    fig, ax = plt.subplots()
    ax._attrs = {}  # type: ignore
    l = mlines.Line2D([], [], color="red", label="r")
    p = mpatches.Patch(color="green", label="g")
    legends = [
        {"title": "l1", "handles": [l], "labels": ["r"]},
        {"title": "l2", "handles": [p], "labels": ["g"]},
    ]
    # Should not raise
    _render_legends(ax, legends, anchor_x=1.0, spacing=0.05, shared_kwargs={"fontsize": 8})
    # Last legend should be on axes
    assert ax.get_legend() is not None
