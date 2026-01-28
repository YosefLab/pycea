import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.pp.setup_tree import add_depth


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D")])
    tree2 = nx.DiGraph([("root", "E"), ("root", "F")])
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["A", "C", "D", "E", "F"]), obst={"tree1": tree1, "tree2": tree2, "empty": nx.DiGraph()}
    )
    yield tdata


@pytest.fixture
def tdata_with_overlap():
    tree = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D")])
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["A", "C", "D"]), obst={"tree1": tree, "tree2": tree}, allow_overlap=True
    )
    yield tdata


def test_add_depth(tdata):
    depths = add_depth(tdata, key_added="depth", copy=True)
    assert isinstance(depths, pd.DataFrame)
    assert depths.loc[("tree1", "root"), "depth"] == 0
    assert depths.loc[("tree1", "C"), "depth"] == 2
    assert tdata.obst["tree1"].nodes["root"]["depth"] == 0
    assert tdata.obst["tree1"].nodes["C"]["depth"] == 2
    assert tdata.obs.loc["C", "depth"] == 2


def test_add_depth_overlap(tdata_with_overlap):
    with pytest.raises(ValueError):
        add_depth(tdata_with_overlap, key_added="depth", copy=True)
    depths = add_depth(tdata_with_overlap, key_added="depth", tree="tree1", copy=True)
    assert isinstance(depths, pd.DataFrame)
    assert depths.loc[("tree1", "C"), "depth"] == 2
    depths = add_depth(tdata_with_overlap, key_added="depth", tree="tree2", copy=True)
    assert isinstance(depths, pd.DataFrame)
    assert depths.loc[("tree2", "C"), "depth"] == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
