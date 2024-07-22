import numpy as np
import pandas as pd
import pytest
import scipy as sp
import treedata as td

from pycea.tl.distance import distance, compare_distance


@pytest.fixture
def tdata():
    tdata = td.TreeData(
        obs=pd.DataFrame(index=["A", "B", "C"]),
        obsm={"spatial": np.array([[0, 0], [1, 1], [2, 2]]), "characters": np.array([[0, 0], [1, 1], [0, 1]])},
        obsp={"connectivities": sp.sparse.csr_matrix(([1,1], ([0, 0], [1, 2])), shape=(3, 3))},
    )
    yield tdata


def test_pairwise_distance(tdata):
    dist = distance(tdata, "spatial", metric="euclidean", key_added="euclidean", copy=True)
    np.testing.assert_array_equal(tdata.obsp["euclidean_distances"], dist)
    assert tdata.obsp["euclidean_distances"].shape == (3, 3)
    assert tdata.obsp["euclidean_distances"][0, 1] == pytest.approx(np.sqrt(2))
    assert tdata.obsp["euclidean_distances"][1, 2] == pytest.approx(np.sqrt(2))
    assert tdata.obsp["euclidean_distances"][0, 2] == pytest.approx(np.sqrt(8))
    assert "euclidean_connectivities" not in tdata.obsp.keys()
    metric = lambda x, y: np.abs(x - y).sum()
    distance(tdata, "characters", metric=metric, key_added="manhatten")
    assert tdata.obsp["manhatten_distances"][0, 1] == 2
    assert tdata.obsp["manhatten_distances"][1, 2] == 1
    assert tdata.obsp["manhatten_distances"][0, 2] == 1


def test_obs_distance(tdata):
    distance(tdata, "spatial", obs="A", metric="manhattan")
    assert tdata.obs["spatial_distances"].tolist() == [0, 2, 4]


def test_select_obs_distance(tdata):
    distance(tdata, "spatial", obs=["A", "C"], metric="cityblock")
    assert isinstance(tdata.obsp["spatial_distances"], sp.sparse.csr_matrix)
    assert tdata.obsp["spatial_distances"][0, 2] == 4
    assert tdata.obsp["spatial_distances"][0, 0] == 0
    assert isinstance(tdata.obsp["spatial_connectivities"], sp.sparse.csr_matrix)
    assert tdata.obsp["spatial_connectivities"][0, 2] == 1
    assert len(tdata.obsp["spatial_connectivities"].data) == 4
    dist = distance(tdata, "spatial", obs=[("A", "C")], metric="cityblock", copy=True)
    assert isinstance(dist, sp.sparse.csr_matrix)
    assert len(dist.data) == 1
    assert dist[0, 2] == 4
    assert isinstance(tdata.obsp["spatial_connectivities"], sp.sparse.csr_matrix)
    assert tdata.obsp["spatial_connectivities"][0, 2] == 1
    assert len(tdata.obsp["spatial_connectivities"].data) == 1


def test_sampled_distance(tdata):
    distance(tdata, "spatial", sample_n=2, metric="cityblock",random_state=0)
    assert tdata.obsp["spatial_distances"].shape == (3, 3)
    assert len(tdata.obsp["spatial_distances"].data) == 2
    assert tdata.obsp["spatial_distances"].data.tolist() == [4, 0]
    assert tdata.obsp["spatial_connectivities"].shape == (3, 3)
    assert len(tdata.obsp["spatial_connectivities"].data) == 2


def test_connected_distance(tdata):
    dist = distance(tdata, "spatial", connect_key="connectivities", metric="cityblock", copy=True)
    assert dist.shape == (3, 3)
    assert len(dist.data) == 2
    assert dist.data.tolist() == [2, 4]
    np.testing.assert_equal(tdata.obsp["spatial_connectivities"].data,tdata.obsp["connectivities"].data)


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
    with pytest.raises(ValueError):
        distance(tdata, "spatial", connect_key="bad", metric="cityblock")


def test_compare_distance(tdata):
    distance(tdata, "spatial", metric="euclidean", key_added="euclidean")
    distance(tdata, "spatial", metric="cityblock", key_added="cityblock")
    dist = compare_distance(tdata, dist_keys=["euclidean", "cityblock"])
    assert dist.shape == (9, 4)
    assert dist.query("obs1 == obs2")["euclidean_distances"].to_list() == [0, 0, 0]
    assert dist.query("obs1 == obs2")["cityblock_distances"].to_list() == [0, 0, 0]
    # sampled
    dist = compare_distance(tdata, dist_keys=["euclidean", "cityblock"], sample_n=2, random_state=0)
    assert dist.shape == (2, 4)
    assert dist["cityblock_distances"].to_list() == [2, 0]


def test_compare_sparse_distance(tdata):
    distance(tdata, "spatial", metric="euclidean", key_added="euclidean", sample_n=4, random_state=1)
    distance(tdata, "spatial", metric="cityblock", key_added="cityblock", connect_key="euclidean")
    dist = compare_distance(tdata, dist_keys=["euclidean", "cityblock"])
    assert dist.shape == (4, 4)
    assert dist["cityblock_distances"].to_list() == [0, 2, 4, 0]
    # sampled
    dist = compare_distance(tdata, dist_keys=["euclidean", "cityblock"], sample_n=2, random_state=0)
    assert dist.shape == (2, 4)
    assert dist["cityblock_distances"].to_list() == [0, 2]


def test_compare_distance_invalid(tdata):
    distance(tdata, "spatial", metric="euclidean", key_added="euclidean")
    with pytest.raises(ValueError):
        compare_distance(tdata, dist_keys=["bad"])
    with pytest.raises(ValueError):
        compare_distance(tdata, dist_keys=["euclidean", "bad"])
    with pytest.raises(ValueError):
        compare_distance(tdata, dist_keys=["euclidean"], sample_n=100)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
