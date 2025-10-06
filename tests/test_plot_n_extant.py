import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import pytest
import treedata as td

import pycea

plot_path = Path(__file__).parent / "plots"


@pytest.fixture
def tdata():
    tree = nx.DiGraph(
        [
            ("root", "A"),
            ("root", "B"),
            ("A", "C"),
            ("A", "D"),
        ]
    )
    nx.set_node_attributes(tree, {"root": 0, "A": 1, "B": 1, "C": 2, "D": 2}, "depth")
    nx.set_node_attributes(tree, {"root": "r", "A": "g1", "B": "g2", "C": "g1", "D": "g1"}, "clade")
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "C", "D"]), obst={"tree": tree})
    return tdata


def test_plot_n_extant(tdata):
    tdata = td.read_h5td("tests/data/tdata.h5ad")
    fig, ax = plt.subplots(dpi=300)
    pycea.tl.n_extant(tdata, "time", groupby="clade", dropna=False)
    pycea.pl.n_extant(tdata, ax=ax)
    plt.savefig(plot_path / "n_extant.png", bbox_inches="tight")
    plt.close()
    assert len(ax.collections) == 10


def test_plot_n_extant_with_data(tdata):
    counts = pycea.tl.n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=False)
    ax = pycea.pl.n_extant(tdata, data=counts, color=None, legend=False)
    assert len(ax.collections) == 3


def test_plot_n_extant_color_order(tdata):
    counts = pycea.tl.n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=True)
    tdata.uns["clade_colors"] = ["green", "blue", "red"]
    ax = pycea.pl.n_extant(tdata, color="clade", data=counts, order=["r", "g1", "g2"], legend=False)
    colors = [tuple(poly.get_facecolor()[0]) for poly in ax.collections]  # type: ignore
    expected = [
        mcolors.to_rgba("red"),
        mcolors.to_rgba("green"),
        mcolors.to_rgba("blue"),
    ]
    assert colors == expected


if __name__ == "__main__":
    pytest.main(["-v", __file__])
