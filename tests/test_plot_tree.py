from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import treedata as td

import pycea

plot_path = Path(__file__).parent / "plots"


@pytest.fixture
def tdata() -> td.TreeData:
    return td.read_h5td("tests/data/tdata.h5ad")


def test_polar_with_clades(tdata):
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(
        tdata, tree="2", polar=True, color="clade", depth_key="time", palette="Set1", na_color="black", ax=ax
    )
    pycea.pl.nodes(tdata, color="clade", palette="Set1", style="clade", ax=ax)
    pycea.pl.annotation(tdata, keys="clade", ax=ax)
    plt.savefig(plot_path / "polar_clades.png", bbox_inches="tight")
    plt.close()


def test_angled_numeric_annotations(tdata):
    pycea.pl.branches(
        tdata,
        polar=False,
        color="length",
        cmap="hsv",
        linewidth="length",
        depth_key="time",
        angled_branches=True,
        vmax=2,
    )
    pycea.pl.nodes(tdata, nodes="all", color="time", style="s", size=20)
    pycea.pl.nodes(tdata, nodes=["2"], tree="1", color="black", style="*", size=200)
    pycea.pl.annotation(
        tdata,
        keys=["x", "y"],
        cmap="jet",
        width=0.1,
        gap=0.05,
        label=["x position", "y position"],
        border_width=2,
        legend=False,
    )
    pycea.pl.annotation(tdata, keys=["0", "1", "2", "3", "4", "5"], label="genes", border_width=2, share_cmap=True)
    plt.savefig(plot_path / "angled_numeric.png", dpi=300, bbox_inches="tight")
    plt.close()


def test_matrix_annotation(tdata):
    fig, ax = plt.subplots(dpi=300, figsize=(7, 3))
    pycea.pl.tree(
        tdata,
        nodes="internal",
        node_color="clade",
        node_size="time",
        depth_key="time",
        keys=["spatial_distances"],
        ax=ax,
    )
    pycea.tl.tree_neighbors(tdata, max_dist=5, depth_key="time", update=False)
    pycea.pl.annotation(tdata, keys="tree_connectivities", ax=ax, palette={True: "black", False: "white"}, legend=False)
    plt.savefig(plot_path / "matrix_annotation.png", bbox_inches="tight")
    plt.close()


def test_character_annotation(tdata):
    tdata.obsm["characters"] = pd.DataFrame(tdata.obsm["characters"], index=tdata.obs_names).astype(str)
    tdata.obsm["characters"].replace("-1", pd.NA, inplace=True)
    palette = {"0": "lightgray"}
    palette.update({str(i + 1): plt.cm.rainbow(i / 7) for i in range(8)})  # type: ignore
    palette = pycea.get.palette(tdata, key="characters", custom={"0": "lightgray"}, cmap="rainbow")
    pycea.pl.tree(
        tdata,
        depth_key="time",
        keys="characters",
        palette=palette,
    )
    assert "characters_colors" in tdata.uns.keys()
    plt.savefig(plot_path / "character_annotation.png", bbox_inches="tight")
    plt.close()


def test_polar_angle_range(tdata):
    # Semicircle
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(tdata, polar=True, angle_range=(0, 180), depth_key="time", ax=ax)
    import numpy as np

    assert np.isclose(ax._attrs["start_angle"], 0.0)
    assert np.isclose(ax._attrs["end_angle"], np.pi)
    pycea.pl.annotation(tdata, keys="clade", ax=ax)
    plt.savefig(plot_path / "polar_semicircle.png", bbox_inches="tight")
    plt.close()

    # Quarter circle
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(tdata, polar=True, angle_range=(0, 90), depth_key="time", ax=ax)
    assert np.isclose(ax._attrs["start_angle"], 0.0)
    assert np.isclose(ax._attrs["end_angle"], np.pi / 2)
    plt.savefig(plot_path / "polar_quarter.png", bbox_inches="tight")
    plt.close()

    # Offset arc
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(tdata, polar=True, angle_range=(45, 315), depth_key="time", ax=ax)
    assert np.isclose(ax._attrs["start_angle"], np.deg2rad(45))
    assert np.isclose(ax._attrs["end_angle"], np.deg2rad(315))
    plt.savefig(plot_path / "polar_offset.png", bbox_inches="tight")
    plt.close()

    # Default (0, 360) unchanged
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(tdata, polar=True, depth_key="time", ax=ax)
    assert np.isclose(ax._attrs["start_angle"], 0.0)
    assert np.isclose(ax._attrs["end_angle"], 2 * np.pi)
    plt.close()


def test_branches_bad_input(tdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, color="bad", depth_key="time")
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, linewidth="bad", depth_key="time")
    # Warns about polar
    with pytest.warns(match="Polar"):
        pycea.pl.branches(tdata, polar=True, ax=ax, depth_key="time")
    plt.close()


def test_nodes_bad_input(tdata):
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, depth_key="time", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.nodes(tdata, nodes="bad", color="clade", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.nodes(tdata, nodes="all", color="bad", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.nodes(tdata, nodes="all", style="bad", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.nodes(tdata, nodes="all", size="bad", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.nodes(tdata, nodes="all", tree="bad", ax=ax)
    plt.close()


def test_annotation_bad_input(tdata):
    # Need to plot branches first
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys="clade")
    pycea.pl.branches(tdata, ax=ax, depth_key="time")
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, tree="bad", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys="clade", label=None, ax=ax)
    plt.close()


def test_annotation_vmin_vmax_label_false(tdata):
    """annotation must not raise when label=False and vmax/vmin are given."""
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, depth_key="time", ax=ax)
    # vmax only
    pycea.pl.annotation(tdata, keys="x", label=False, legend=False, vmax=1.0, ax=ax)
    # vmin + vmax (triggers share_cmap=True, which previously hit labels[0] on empty list)
    pycea.pl.annotation(tdata, keys="x", label=False, legend=False, vmin=0.0, vmax=1.0, ax=ax)
    plt.close()


def test_hex_color_branches(tdata):
    """Branches colored by a per-edge hex attribute use the raw hex values directly."""
    import matplotlib.colors as mcolors
    hex_colors = {"1": "#e41a1c", "2": "#377eb8"}
    for tree_key, tree in tdata.obst.items():
        for u, v, data in tree.edges(data=True):
            data["hex_color"] = hex_colors[tree_key]
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, color="hex_color", depth_key="time", ax=ax)
    edge_colors = ax.collections[0].get_colors()
    expected = {mcolors.to_rgba(c) for c in hex_colors.values()}
    actual = {tuple(row) for row in edge_colors}
    assert actual == expected
    plt.close()


def test_hex_color_nodes(tdata):
    """Nodes colored by a per-node hex attribute use the raw hex values directly."""
    import matplotlib.colors as mcolors
    hex_color = "#4daf4a"
    for node, data in tdata.obst["1"].nodes(data=True):
        data["hex_color"] = hex_color
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, tree="1", depth_key="time", ax=ax)
    pycea.pl.nodes(tdata, nodes="leaves", color="hex_color", ax=ax)
    node_colors = ax.collections[1].get_facecolors()
    expected = mcolors.to_rgba(hex_color)
    assert all(tuple(row) == expected for row in node_colors)
    plt.close()


def test_nodes_outline_width(tdata):
    """outline_width draws a black edge only around visible (non-na) nodes."""
    import matplotlib.colors as mcolors
    # Default (no outline): does not error
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, depth_key="time", ax=ax)
    pycea.pl.nodes(tdata, nodes="internal", ax=ax)
    plt.close()

    # With outline: visible nodes get black edge at given width
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, depth_key="time", ax=ax)
    pycea.pl.nodes(tdata, nodes="internal", outline_width=1.5, ax=ax)
    sc = ax.collections[1]
    assert all(mcolors.to_rgba(c) == (0.0, 0.0, 0.0, 1.0) for c in sc.get_edgecolors())
    assert all(w == pytest.approx(1.5) for w in sc.get_linewidths())
    plt.close()

    # na nodes (alpha=0) must not get a black outline
    fig, ax = plt.subplots()
    pycea.pl.branches(tdata, depth_key="time", ax=ax)
    # color="clade" → some nodes will be missing data and get na_color="#FFFFFF00"
    pycea.pl.nodes(tdata, nodes="all", color="clade", outline_width=1.0, ax=ax)
    sc = ax.collections[1]
    face_rgba = mcolors.to_rgba_array(sc.get_facecolors())
    edge_rgba = mcolors.to_rgba_array(sc.get_edgecolors())
    for face, edge in zip(face_rgba, edge_rgba):
        if face[3] == 0:  # transparent (na) node
            assert edge[3] == 0, "na nodes must not have a visible outline"
        else:
            assert tuple(edge) == (0.0, 0.0, 0.0, 1.0), "visible nodes must have a black outline"
    plt.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
