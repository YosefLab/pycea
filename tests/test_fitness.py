import pandas as pd
import pytest
import treedata as td

from pycea.tl.fitness import fitness


@pytest.fixture
def tdata() -> td.TreeData:
    return td.read_h5td("tests/data/tdata.h5ad")


def test_fitness_lbi(tdata):
    result = fitness(tdata, depth_key="time", method="lbi", copy=True)
    assert isinstance(result, pd.DataFrame)
    assert result.loc[("1", "181"), "fitness"] == pytest.approx(0.8, abs=1e-1)
    assert result.fitness.max() == pytest.approx(2.8, abs=1e-1)


def test_fitness_sbd(tdata):
    result = fitness(tdata, depth_key="time", method="sbd", copy=True, tree="1")
    assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
