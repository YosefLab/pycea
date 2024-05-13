import networkx as nx
import pandas as pd


def get_root(tree: nx.DiGraph):
    """Finds the root of a tree"""
    if not tree.nodes():
        return None  # Handle empty graph case.
    node = next(iter(tree.nodes))
    while True:
        parent = list(tree.predecessors(node))
        if not parent:
            return node  # No predecessors, this is the root
        node = parent[0]


def _get_keyed_edge_data(tree: nx.DiGraph, key: str) -> pd.Series:
    """Gets edge data for a given key from a tree."""
    edge_data = {
        (parent, child): data.get(key)
        for parent, child, data in tree.edges(data=True)
        if key in data and data[key] is not None
    }
    return pd.Series(edge_data, name=key)
