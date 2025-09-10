import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl import n_extant as tl_n_extant
from pycea.pl import n_extant as pl_n_extant


@pytest.fixture
def tdata():
    tree = nx.DiGraph([
        ("root", "A"),
        ("root", "B"),
        ("A", "C"),
        ("A", "D"),
    ])
    nx.set_node_attributes(tree, {"root": 0, "A": 1, "B": 1, "C": 2, "D": 2}, "depth")
    nx.set_node_attributes(tree, {"root": "r", "A": "g1", "B": "g2", "C": "g1", "D": "g1"}, "clade")
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "C", "D"]), obst={"tree": tree})
    return tdata


def test_plot_n_extant_uses_uns(tdata):
    tl_n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=False)
    ax = pl_n_extant(tdata, legend=False)
    assert len(ax.collections) == 3


def test_plot_n_extant_ax_return(tdata):
    counts = tl_n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=True)
    fig, ax = plt.subplots()
    returned = pl_n_extant(tdata, group_key="clade", data=counts, ax=ax, legend=False)
    assert returned is ax
    plt.close(fig)


def test_plot_n_extant_color_order(tdata):
    counts = tl_n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=True)
    tdata.uns["clade_colors"] = ["green", "blue", "red"]
    ax = pl_n_extant(tdata, group_key="clade", data=counts, order=["r", "g1", "g2"], legend=False)
    colors = [tuple(poly.get_facecolor()[0]) for poly in ax.collections]
    expected = [
        mpl.colors.to_rgba("red"),
        mpl.colors.to_rgba("green"),
        mpl.colors.to_rgba("blue"),
    ]
    assert colors == expected
    plt.close(ax.figure)
