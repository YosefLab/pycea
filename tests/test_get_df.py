import networkx as nx
import treedata as td

from pycea.get import edge_df, node_df


def _make_tree():
    t = nx.DiGraph()
    t.add_edge("A", "B", weight=1, length=5)
    t.add_edge("A", "C", weight=3)
    t.nodes["A"].update(depth=0, label="root")
    t.nodes["B"].update(depth=1, value=42)
    t.nodes["C"].update(depth=1)
    return t


def test_edge_df_single_tree():
    tdata = td.TreeData(obst={"t1": _make_tree()})
    df = edge_df(tdata)
    assert set(df.columns) == {"weight", "length"}
    assert df.index.names == ["edge"]
    assert ("A", "B") in df.index


def test_edge_df_multiple_trees_and_tree_param():
    t1 = _make_tree()
    t2 = nx.DiGraph()
    t2.add_edge("X", "Y", capacity=3)
    tdata = td.TreeData(obst={"t1": t1, "t2": t2})

    df = edge_df(tdata)
    assert set(df.columns) == {"weight", "length", "capacity"}
    assert df.index.names == ["tree", "edge"]
    assert ("t2", ("X", "Y")) in df.index

    df_t1 = edge_df(tdata, tree="t1")
    assert df_t1.index.names == ["edge"]


def test_node_df_single_tree():
    tdata = td.TreeData(obst={"t1": _make_tree()})
    df = node_df(tdata)
    assert set(df.columns) == {"depth", "label", "value"}
    assert df.index.names == ["node"]
    assert df.loc["B", "depth"].item() == 1


def test_node_df_multiple_trees_and_tree_param():
    t1 = _make_tree()
    t2 = nx.DiGraph()
    t2.add_edge("X", "Y")
    t2.nodes["X"].update(color="red")
    t2.nodes["Y"].update(color="blue")
    tdata = td.TreeData(obst={"t1": t1, "t2": t2})

    df = node_df(tdata)
    assert set(df.columns) == {"depth", "label", "value", "color"}
    assert df.index.names == ["tree", "node"]
    assert ("t2", "X") in df.index

    df_t1 = node_df(tdata, tree="t1")
    assert df_t1.index.names == ["node"]
