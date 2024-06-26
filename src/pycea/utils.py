from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def get_leaves(tree: nx.DiGraph):
    """Finds the leaves of a tree"""
    return [node for node in nx.dfs_postorder_nodes(tree, get_root(tree)) if tree.out_degree(node) == 0]


def get_keyed_edge_data(
    tdata: td.TreeData, keys: str | Sequence[str], tree_keys: str | Sequence[str] = None
) -> pd.DataFrame:
    """Gets edge data for a given key from a tree or set of trees."""
    if isinstance(tree_keys, str):
        tree_keys = [tree_keys]
    if isinstance(keys, str):
        keys = [keys]
    trees = get_trees(tdata, tree_keys)
    data = []
    for name, tree in trees.items():
        edge_data = {key: nx.get_edge_attributes(tree, key) for key in keys}
        edge_data = pd.DataFrame(edge_data)
        edge_data["tree"] = name
        edge_data["edge"] = edge_data.index
        data.append(edge_data)
    data = pd.concat(data)
    data = data.set_index(["tree", "edge"])
    return data


def get_keyed_node_data(
    tdata: td.TreeData, keys: str | Sequence[str], tree_keys: str | Sequence[str] = None
) -> pd.DataFrame:
    """Gets node data for a given key a tree or set of trees."""
    if isinstance(tree_keys, str):
        tree_keys = [tree_keys]
    if isinstance(keys, str):
        keys = [keys]
    trees = get_trees(tdata, tree_keys)
    data = []
    for name, tree in trees.items():
        tree_data = {key: nx.get_node_attributes(tree, key) for key in keys}
        tree_data = pd.DataFrame(tree_data)
        tree_data["tree"] = name
        data.append(tree_data)
    data = pd.concat(data)
    data["node"] = data.index
    data = data.set_index(["tree", "node"])
    return data


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
    if not column_keys and not array_keys:
        raise ValueError("No valid keys found.")
    # Convert to DataFrame
    if column_keys:
        data = pd.concat(data, axis=1)
        data.columns = keys
    elif array_keys:
        data = pd.DataFrame(data[0], index=tdata.obs_names)
        if data.shape[0] == data.shape[1]:
            data.columns = tdata.obs_names
    return data, array_keys


def get_trees(tdata: td.TreeData, tree_keys: str | Sequence[str] | None) -> Mapping[str, nx.DiGraph]:
    """Gets tree data for a given key from a tree."""
    trees = {}
    if tree_keys is None:
        tree_keys = tdata.obst.keys()
    elif isinstance(tree_keys, str):
        tree_keys = [tree_keys]
    elif isinstance(tree_keys, Sequence):
        if tdata.allow_overlap:
            raise ValueError("Cannot request multiple trees when tdata.allow_overlap is True.")
        tree_keys = list(tree_keys)
    else:
        raise ValueError("Tree keys must be a string, list of strings, or None.")
    for key in tree_keys:
        if key not in tdata.obst.keys():
            raise ValueError(f"Key {key!r} is not present in obst.")
        trees[key] = tdata.obst[key]
    return trees
