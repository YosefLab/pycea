import pytest
import treedata as td

from pycea.tl.fitness import fitness


@pytest.fixture
def tdata() -> td.TreeData:
    return td.read_h5td("tests/data/tdata.h5ad")


def test_fitness_attaches_attribute(tdata):
    fitness(tdata, depth_key="time", method="lbi", copy=False)
    G = tdata.obst["1"]
    assert all("fitness" in data for _, data in G.nodes(data=True))
    assert "fitness" in tdata.obs
