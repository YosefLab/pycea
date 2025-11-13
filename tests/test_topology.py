import networkx as nx
import pandas as pd
import pytest
import treedata as td

import pycea as py
from pycea.tl.topology import expansion_test


@pytest.fixture
def test_tree():
    """Create a test TreeData object with a tree topology."""
    # Create tree topology
    tree = nx.DiGraph()
    tree.add_edges_from(
        [
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("1", "4"),
            ("1", "5"),
            ("2", "6"),
            ("2", "7"),
            ("3", "8"),
            ("3", "9"),
            ("3", "16"),
            ("7", "10"),
            ("7", "11"),
            ("8", "12"),
            ("8", "13"),
            ("9", "14"),
            ("9", "15"),
            ("16", "17"),
            ("16", "18"),
        ]
    )

    # Create TreeData object
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["4", "5", "6", "10", "11", "12", "13", "14", "15", "17", "18"]),
        obst={"tree": tree},
    )

    return tdata


def test_expansion_test_min_clade(test_tree):
    """Test that min_clade_size=20 filters out all clades."""
    expansion_test(test_tree, min_clade_size=20)
    node_data = py.get.node_df(test_tree)
    assert (node_data["expansion_pvalue"] == 1.0).all(), "All nodes should be filtered with min_clade_size=20"


def test_expansion_test_basic(test_tree):
    """Test expansion p-values with min_clade_size=2."""
    expansion_test(test_tree, min_clade_size=2)
    expected_basic = {
        "0": 1.0,
        "1": 0.3,
        "2": 0.8,
        "3": 0.047,
        "4": 1.0,
        "5": 1.0,
        "6": 1.0,
        "7": 0.5,
        "8": 0.6,
        "9": 0.6,
        "10": 1.0,
        "11": 1.0,
        "12": 1.0,
        "13": 1.0,
        "14": 1.0,
        "15": 1.0,
        "16": 0.6,
        "17": 1.0,
        "18": 1.0,
    }
    node_data = py.get.node_df(test_tree)
    for node, expected in expected_basic.items():
        actual = node_data.loc[node, "expansion_pvalue"]
        assert abs(actual - expected) < 0.01, f"Basic: Node {node} expected {expected}, got {actual}"


def test_expansion_test_depth_filter(test_tree):
    """Test filtering with min_depth=3."""
    expansion_test(test_tree, min_clade_size=2, min_depth=3)
    expected_depth = {
        "0": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
        "4": 1.0,
        "5": 1.0,
        "6": 1.0,
        "7": 1.0,
        "8": 0.6,
        "9": 0.6,
        "10": 1.0,
        "11": 1.0,
        "12": 1.0,
        "13": 1.0,
        "14": 1.0,
        "15": 1.0,
        "16": 0.6,
        "17": 1.0,
        "18": 1.0,
    }
    node_data = py.get.node_df(test_tree)
    for node, expected in expected_depth.items():
        actual = node_data.loc[node, "expansion_pvalue"]
        assert abs(actual - expected) < 0.01, f"Depth filter: Node {node} expected {expected}, got {actual}"


def test_expansion_test_copy_behavior(test_tree):
    """Test that copy=True returns DataFrane without modifying original."""
    result_df = expansion_test(test_tree, min_clade_size=2, min_depth=1, copy=True)
    assert isinstance(result_df, pd.DataFrame)
    # Check copy has correct values
    expected = {
        "0": 1.0,
        "1": 0.3,
        "2": 0.8,
        "3": 0.047,
        "4": 1.0,
        "5": 1.0,
        "6": 1.0,
        "7": 0.5,
        "8": 0.6,
        "9": 0.6,
        "10": 1.0,
        "11": 1.0,
        "12": 1.0,
        "13": 1.0,
        "14": 1.0,
        "15": 1.0,
        "16": 0.6,
        "17": 1.0,
        "18": 1.0,
    }
    for node, exp in expected.items():
        actual = result_df.loc[("tree", node), "expansion_pvalue"]
        assert abs(actual - exp) < 0.01, f"Copy: Node {node} expected {exp}, got {actual}"

    # Check original was NOT modified
    original_tree_graph = test_tree.obst["tree"]
    for node in original_tree_graph.nodes():
        assert "expansion_pvalue" not in original_tree_graph.nodes[node], (
            f"Original tree modified at node {node} when copy=True"
        )


def test_expansion_test_multiple_trees():
    """Test multiple trees."""
    tree1 = nx.DiGraph()
    tree1.add_edges_from([("0", "1"), ("0", "2")])
    tree2 = nx.DiGraph()
    tree2.add_edges_from([("A", "B"), ("A", "C")])
    tdata_multi = td.TreeData(
        obs=pd.DataFrame(index=["1", "2", "B", "C"]),
        obst={"tree1": tree1, "tree2": tree2},
    )
    expansion_test(tdata_multi, min_clade_size=2)
    assert "expansion_pvalue" in tdata_multi.obst["tree1"].nodes["0"]
    assert "expansion_pvalue" in tdata_multi.obst["tree2"].nodes["A"]

    tdata_multi2 = td.TreeData(
        obs=pd.DataFrame(index=["1", "2", "3", "4", "B", "C"]),
        obst={"tree1": tree1.copy(), "tree2": tree2.copy()},
    )
    expansion_test(tdata_multi2, min_clade_size=2, tree="tree1")
    assert "expansion_pvalue" in tdata_multi2.obst["tree1"].nodes["0"]
    assert "expansion_pvalue" not in tdata_multi2.obst["tree2"].nodes["A"]


def test_expansion_test_custom_key(test_tree):
    """Test using custom key_added parameter."""
    expansion_test(test_tree, min_clade_size=2, key_added="custom_pvalue")
    assert "custom_pvalue" in test_tree.obst["tree"].nodes["0"]
    assert "expansion_pvalue" not in test_tree.obst["tree"].nodes["0"]
