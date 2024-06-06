from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import pycea

plot_path = Path(__file__).parent / "plots"


def test_polar_with_clades(tdata):
    fig, ax = plt.subplots(dpi=300, subplot_kw={"polar": True})
    pycea.pl.branches(
        tdata, tree="clone1", polar=True, color="clade", depth_key="time", palette="Set1", na_color="black", ax=ax
    )
    pycea.pl.nodes(tdata, color="clade", palette="Set1", style="clade", ax=ax)
    pycea.pl.annotation(tdata, keys="clade", ax=ax)
    plt.savefig(plot_path / "polar_clades.png")
    plt.close()


def test_angled_numeric_annotations(tdata):
    pycea.pl.branches(
        tdata, polar=False, color="length", cmap="hsv", linewidth="length", depth_key="time", angled_branches=True
    )
    pycea.pl.nodes(tdata, nodes="all", color="time", style="s", size=20)
    pycea.pl.nodes(tdata, nodes=["5"], tree="clone2", color="black", style="*", size=200)
    pycea.pl.annotation(tdata, keys=["x", "y"], cmap="magma", width=0.1, gap=0.05, border_width=2)
    pycea.pl.annotation(tdata, keys=["0", "1", "2", "3", "4", "5"], label="genes", border_width=2)
    plt.savefig(plot_path / "angled_numeric.png")
    plt.close()


def test_matrix_annotation(tdata):
    fig, ax = plt.subplots(dpi=300)
    pycea.pl.tree(
        tdata,
        nodes="internal",
        node_color="clade",
        node_size="time",
        depth_key="time",
        keys=["spatial_distance"],
        ax=ax,
    )
    plt.savefig(plot_path / "matrix_annotation.png")
    plt.close()


def test_branches_bad_input(tdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, color=["bad"] * 5, depth_key="time")
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, linewidth=["bad"] * 5, depth_key="time")
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
        pycea.pl.annotation(tdata, keys="clade", label={}, ax=ax)
    plt.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
