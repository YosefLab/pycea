import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl import n_extant


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
    nx.set_node_attributes(tree, {"root": "L1", "A": "L1", "B": "L1", "C": "L1", "D": "L1"}, "lineage")
    tdata = td.TreeData(obs=pd.DataFrame(index=["B", "C", "D"]), obst={"tree": tree})
    return tdata


def test_n_extant_uns_and_copy(tdata):
    counts = n_extant(tdata, "depth", bins=[0, 1, 2, 3], copy=True)
    assert "n_extant" in tdata.uns
    assert counts.equals(tdata.uns["n_extant"])
    assert counts["n_extant"].tolist() == [1, 2, 2, 0]


def test_n_extant_no_copy(tdata):
    result = n_extant(tdata, "depth", bins=[0, 1, 2, 3], copy=False)
    assert result is None
    assert "n_extant" in tdata.uns


def test_n_extant_extend_branches(tdata):
    counts = n_extant(tdata, "depth", bins=[0, 1, 2, 3], copy=True, extend_branches=True)
    assert "n_extant" in tdata.uns
    assert counts.equals(tdata.uns["n_extant"])
    assert counts["n_extant"].tolist() == [1, 2, 3, 3]


def test_n_extant_groupby(tdata):
    counts = n_extant(tdata, "depth", groupby="clade", bins=[0, 1, 2, 3], copy=True)
    assert set(counts["clade"]) == {"r", "g1", "g2"}
    r = counts[counts["clade"] == "r"]["n_extant"].tolist()
    g1 = counts[counts["clade"] == "g1"]["n_extant"].tolist()
    g2 = counts[counts["clade"] == "g2"]["n_extant"].tolist()
    assert r == [1, 0, 0, 0]
    assert g1 == [0, 1, 2, 0]
    assert g2 == [0, 1, 0, 0]


def test_n_extant_multi_groupby(tdata):
    counts = n_extant(tdata, "depth", groupby=["clade", "lineage"], bins=[0, 1, 2, 3], copy=True)
    assert set(counts["clade"]) == {"r", "g1", "g2"}
    assert set(counts["lineage"]) == {"L1"}
    r = counts[counts["clade"] == "r"]["n_extant"].tolist()
    g1 = counts[counts["clade"] == "g1"]["n_extant"].tolist()
    g2 = counts[counts["clade"] == "g2"]["n_extant"].tolist()
    assert r == [1, 0, 0, 0]
    assert g1 == [0, 1, 2, 0]
    assert g2 == [0, 1, 0, 0]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
