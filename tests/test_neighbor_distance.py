import numpy as np
import pandas as pd
import pytest
import scipy as sp
import treedata as td

from pycea.tl.neighbor_distance import neighbor_distance


@pytest.fixture
def tdata():
    distances = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
    neighbors = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    tdata = td.TreeData(
        obs=pd.DataFrame({"group": ["1", "1", "2"]}, index=["A", "B", "C"]),
        obsp={
            "connectivities": neighbors,
            "sparse_connectivities": sp.sparse.csr_matrix(neighbors),
            "distances": distances,
            "sparse_distances": sp.sparse.csr_matrix(distances),
        },
    )
    yield tdata


@pytest.mark.parametrize("connect_key", ["connectivities", "sparse_connectivities"])
@pytest.mark.parametrize("dist_key", ["distances", "sparse_distances"])
def test_neighbor_distance(tdata, connect_key, dist_key):
    distances = neighbor_distance(tdata, connect_key=connect_key, dist_key=dist_key, copy=True)
    assert tdata.obs["neighbor_distances"].equals(distances)
    assert isinstance(distances, pd.Series)
    assert np.allclose(distances.values.tolist(), [2.5, 2, np.nan], equal_nan=True)


def test_neighbor_distance_methods(tdata):
    distances = neighbor_distance(tdata, connect_key="connectivities", dist_key="distances", method="min", copy=True)
    assert np.allclose(distances.values.tolist(), [2, 2, np.nan], equal_nan=True)
    distances = neighbor_distance(tdata, connect_key="connectivities", dist_key="distances", method="max", copy=True)
    assert np.allclose(distances.values.tolist(), [3, 2, np.nan], equal_nan=True)
    distances = neighbor_distance(tdata, connect_key="connectivities", dist_key="distances", method="median", copy=True)
    assert np.allclose(distances.values.tolist(), [2.5, 2, np.nan], equal_nan=True)
    distances = neighbor_distance(tdata, connect_key="connectivities", dist_key="distances", method=np.mean, copy=True)
    assert np.allclose(distances.values.tolist(), [2.5, 2, np.nan], equal_nan=True)


def test_neighbor_distance_missing(tdata):
    tdata.obsp["missing_distances"] = sp.sparse.csr_matrix(([1, 1], ([0, 0], [0, 1])), shape=(3, 3))
    with pytest.raises(ValueError):
        neighbor_distance(tdata, connect_key="connectivities", dist_key="missing_distances", copy=True)
    with pytest.raises(ValueError):
        neighbor_distance(tdata, connect_key="sparse_connectivities", dist_key="missing_distances", copy=True)


def test_neighbor_distance_invalid(tdata):
    with pytest.raises(ValueError):
        neighbor_distance(tdata, connect_key=None, dist_key="distances", copy=True)
    with pytest.raises(ValueError):
        neighbor_distance(tdata, connect_key="connectivities", dist_key=None, copy=True)
    with pytest.raises(ValueError):
        neighbor_distance(tdata, connect_key="connectivities", dist_key="distances", method="invalid", copy=True)
    with pytest.raises(KeyError):
        neighbor_distance(tdata, connect_key="invalid", dist_key="distances", copy=True)
    with pytest.raises(KeyError):
        neighbor_distance(tdata, connect_key="connectivities", dist_key="invalid", copy=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
