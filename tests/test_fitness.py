import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.pp.setup_tree import add_depth
from pycea.tl.fitness import fitness


@pytest.fixture
def tdata():
    tree1 = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D")])
    tdata = td.TreeData(obs=pd.DataFrame(index=["A", "C", "D"]), obst={"tree1": tree1})
    return tdata


def test_fitness_returns_df_and_sets_attributes(tdata):
    add_depth(tdata, key_added="depth")
    df = fitness(tdata, tree="tree1", depth_key="depth", key_added="fitness", copy=True)
    assert isinstance(df, pd.DataFrame)
    assert "fitness" in df.columns
    for leaf in ["A", "C", "D"]:
        assert leaf in df.index
        assert not pd.isna(tdata.obs.loc[leaf, "fitness"])
        assert df.loc[leaf, "fitness"] == tdata.obst["tree1"].nodes[leaf]["fitness"]
    for node in ["root", "B"]:
        assert "fitness" in tdata.obst["tree1"].nodes[node]


def test_fitness_copy_false(tdata):
    add_depth(tdata, key_added="depth")
    result = fitness(tdata, tree="tree1", depth_key="depth", key_added="lbi", copy=False)
    assert result is None
    for leaf in ["A", "C", "D"]:
        assert "lbi" in tdata.obst["tree1"].nodes[leaf]
        assert not pd.isna(tdata.obs.loc[leaf, "lbi"])
