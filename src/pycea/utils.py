from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import pandas as pd
import treedata as td


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


def get_keyed_edge_data(tree: nx.DiGraph, key: str) -> pd.Series:
    """Gets edge data for a given key from a tree."""
    edge_data = {
        (parent, child): data.get(key)
        for parent, child, data in tree.edges(data=True)
        if key in data and data[key] is not None
    }
    return pd.Series(edge_data, name=key)


def get_keyed_node_data(tree: nx.DiGraph, key: str) -> pd.Series:
    """Gets node data for a given key from a tree."""
    node_data = {node: data.get(key) for node, data in tree.nodes(data=True) if key in data and data[key] is not None}
    return pd.Series(node_data, name=key)


def get_keyed_obs_data(tdata: td.TreeData, keys: Sequence[str], layer: str = None) -> pd.DataFrame:
    """Gets observation data for a given key from a tree."""
    data = []
    column_keys = False
    array_keys = False
    for key in keys:
        if key in tdata.obs_keys():
            if tdata.obs[key].dtype.kind in ["b", "O", "S"]:
                tdata.obs[key] = tdata.obs[key].astype("category")
            data.append(tdata.obs[key])
            column_keys = True
        elif key in tdata.var_names:
            data.append(pd.Series(tdata.obs_vector(key, layer=layer), index=tdata.obs_names))
            column_keys = True
        elif "obsm" in dir(tdata) and key in tdata.obsm.keys():
            data.append(tdata.obsm[key])
            array_keys = True
        elif "obsp" in dir(tdata) and key in tdata.obsp.keys():
            data.append(tdata.obsp[key])
            array_keys = True
        else:
            raise ValueError(
                f"Key {key!r} is invalid! You must pass a valid observation annotation. "
                f"One of obs_keys, var_names, obsm_keys, obsp_keys."
            )
    if column_keys and array_keys:
        raise ValueError("Cannot mix column and matrix keys.")
    if array_keys and len(keys) > 1:
        raise ValueError("Cannot request multiple matrix keys.")
    # Convert to DataFrame
    if column_keys:
        data = pd.concat(data, axis=1)
        data.columns = keys
    elif array_keys:
        data = pd.DataFrame(data[0], index=tdata.obs_names)

        if data.shape[0] == data.shape[1]:
            data.columns = tdata.obs_names
    return data, array_keys