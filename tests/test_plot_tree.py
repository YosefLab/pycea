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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
