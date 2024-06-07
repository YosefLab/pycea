import numpy as np
import pandas as pd
import pytest
import scipy as sp
import treedata as td

from pycea.tl.distance import distance


@pytest.fixture
def tdata():
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["A", "B", "C"]),
        obsm={"spatial": np.array([[0, 0], [1, 1], [2, 2]]), "characters": np.array([[0, 0], [1, 1], [0, 1]])},
    )
    yield tdata


def test_pairwise_distance(tdata):
    dist = distance(tdata, "spatial", metric="euclidean", key_added="euclidean", copy=True)
    np.testing.assert_array_equal(tdata.obsp["euclidean"], dist)
    assert tdata.obsp["euclidean"].shape == (3, 3)
    assert tdata.obsp["euclidean"][0, 1] == pytest.approx(np.sqrt(2))
    assert tdata.obsp["euclidean"][1, 2] == pytest.approx(np.sqrt(2))
    assert tdata.obsp["euclidean"][0, 2] == pytest.approx(np.sqrt(8))
    metric = lambda x, y: np.abs(x - y).sum()
    distance(tdata, "characters", metric=metric, key_added="manhatten")
    assert tdata.obsp["manhatten"][0, 1] == 2
    assert tdata.obsp["manhatten"][1, 2] == 1
    assert tdata.obsp["manhatten"][0, 2] == 1


def test_obs_distance(tdata):
    distance(tdata, "spatial", obs="A", metric="manhattan")
    assert tdata.obs["spatial_distances"].tolist() == [0, 2, 4]


def test_select_obs_distance(tdata):
    distance(tdata, "spatial", obs=["A", "C"], metric="cityblock")
    assert isinstance(tdata.obsp["spatial_distances"], sp.sparse.csr_matrix)
    assert tdata.obsp["spatial_distances"][0, 2] == 4
    assert tdata.obsp["spatial_distances"][0, 0] == 0
    dist = distance(tdata, "spatial", obs=[("A", "C")], metric="cityblock", copy=True)
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert len(dist.data) == 1
    assert dist[0, 2] == 4


def test_distance_invalid(tdata):
    with pytest.raises(ValueError):
        distance(tdata, "bad", metric="cityblock")
    with pytest.raises(ValueError):
        distance(tdata, "spatial", obs=1, metric="cityblock")
    with pytest.raises(ValueError):
        distance(tdata, "spatial", obs=[1], metric="cityblock")
    with pytest.raises(ValueError):
        distance(tdata, "spatial", obs=[("A",)], metric="cityblock")
    with pytest.raises(ValueError):
        distance(tdata, "spatial", obs=[("A", "B", "C")], metric="cityblock")
    with pytest.raises(ValueError):
        distance(tdata, "spatial", metric="bad")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
