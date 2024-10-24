"""Tool utilities"""

from __future__ import annotations

import random
import warnings
from collections.abc import Mapping, Sequence

import numpy as np
import scipy as sp
import treedata as td


def _check_previous_params(tdata: td.TreeData, params: Mapping, key: str, suffixes: Sequence[str]) -> None:
    """When a function is updating previous results, check that the parameters are the same."""
    for suffix in suffixes:
        if f"{key}_{suffix}" in tdata.uns:
            prev_params = tdata.uns[f"{key}_{suffix}"]["params"]
            for param, value in params.items():
                if param not in prev_params or prev_params[param] != value:
                    raise ValueError(
                        f"{param} value does not match previous call. "
                        f"Previous: {prev_params}. Current: {params}. "
                        f"Set `update=False` to avoid this error."
                    )
    return None


def _csr_data_mask(csr):
    """Boolean mask of explicit data in a csr matrix including zeros"""
    return sp.sparse.csr_matrix((np.ones(len(csr.data), dtype=bool), csr.indices, csr.indptr), shape=csr.shape)


def _set_random_state(random_state):
    """Set random state"""
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)
    return


def _format_keys(keys, suffix):
    """Ensures that keys are formatted correctly"""
    if keys is None:
        pass
    elif isinstance(keys, str):
        if not keys.endswith(suffix):
            keys = f"{keys}_{suffix}"
    elif isinstance(keys, Sequence):
        keys = [f"{key}_{suffix}" if not key.endswith(suffix) else key for key in keys]
    else:
        raise ValueError("keys must be a string or a sequence of strings.")
    return keys


def _format_as_list(obj):
    """Ensures that obj is a list"""
    if obj is None:
        pass
    elif not isinstance(obj, Sequence):
        obj = [obj]
    return obj


def _set_distances_and_connectivities(tdata, key_added, dist, connect, update):
    """Set distances and connectivities in tdata"""
    dist_key = f"{key_added}_distances"
    connect_key = f"{key_added}_connectivities"
    if update and (dist_key in tdata.obsp.keys()):
        if isinstance(dist, np.ndarray):
            tdata.obsp[dist_key] = dist
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = _csr_data_mask(dist)
                tdata.obsp[dist_key][mask] = dist[mask]
    else:
        if dist_key in tdata.obsp.keys():
            del tdata.obsp[dist_key]
        if f"{key_added}_neighbors" in tdata.uns.keys():
            del tdata.uns[f"{key_added}_neighbors"]
        tdata.obsp[dist_key] = dist
    if connect is not None:
        tdata.obsp[connect_key] = connect
    return None


def _assert_param_xor(params):
    """Assert that only one of the parameters is set"""
    n_set = sum([value is not None for key, value in params.items()])
    param_text = ", ".join(params.keys())
    if n_set > 1:
        raise ValueError(f"Only one of {param_text} can be set.")
    if n_set == 0:
        raise ValueError(f"At least one of {param_text} must be set.")
    return None


def _check_tree_overlap(tdata, tree_keys):
    """Check single tree is requested when allow_overlap is True"""
    if tree_keys is None:
        tree_keys = tdata.obst.keys()
        if tdata.allow_overlap and len(tree_keys) > 1:
            raise ValueError("Must specify a tree when tdata.allow_overlap is True.")
    elif isinstance(tree_keys, str):
        pass
    elif isinstance(tree_keys, Sequence):
        if tdata.allow_overlap:
            raise ValueError("Cannot request multiple trees when tdata.allow_overlap is True.")
    else:
        raise ValueError("Tree keys must be a string, list of strings, or None.")
