import networkx as nx
import pandas as pd
import pytest
import treedata as td

from pycea.tl.fitness import fitness


@pytest.fixture
def tdata():
    tree = nx.DiGraph([("root", "A"), ("root", "B"), ("B", "C"), ("B", "D"), ("B", "E")])
    nx.set_node_attributes(tree, {"root": 0, "A": 3, "B": 1, "C": 2.5, "D": 2.5, "E": 2.5}, "depth")
    tdata = td.TreeData(obst={"tree": tree, "empty": nx.DiGraph()})
    yield tdata


def test_fitness_sbd(tdata):
    out = fitness(tdata, method="sbd", copy=False, random_state=42)
    assert out is None
    tree = tdata.obst["tree"]
    assert all("fitness" in tree.nodes[n] for n in tree.nodes)
    assert "fitness" in tdata.obs.columns
    assert tdata.obs.loc["A", "fitness"] == pytest.approx(0.1785, abs=1e-3)


def test_fitness_lbi_copy(tdata):
    out = fitness(tdata, method="lbi", copy=True, random_state=42)
    assert isinstance(out, pd.DataFrame)
    assert "fitness" in out.columns
    assert out.loc["B", "fitness"] == pytest.approx(1.120, abs=1e-3)
    tree = tdata.obst["tree"]
    assert all("fitness" in tree.nodes[n] for n in tree.nodes)
    assert set(out.droplevel(0).index) == set(tree.nodes)


def test_bad_input(tdata):
    with pytest.raises(ValueError):
        fitness(tdata, method="nope")  # type: ignore[arg-type]


def test_fitness_parameters(tdata):
    # Tau parameter
    fitness(tdata, method="lbi", key_added="lbi", method_kwargs={"tau": 2}, copy=False, random_state=42)
    tree = tdata.obst["tree"]
    assert all("lbi" in tree.nodes[n] for n in tree.nodes)
    assert tree.nodes["B"]["lbi"] == pytest.approx(4.8951, abs=1e-3)
    # Gamma parameter
    fitness(
        tdata, method="sbd", key_added="sbd", method_kwargs={"gamma": 0.5, "attach_posteriors": True}, random_state=42
    )
    assert all("sbd" in tree.nodes[n] for n in tree.nodes)
    assert tree.nodes["B"]["sbd"] == pytest.approx(3.6119, abs=1e-3)
    assert tree.nodes["B"]["sbd_var"] is not None


def test_random_state_gives_reproducible_output(tdata):
    df1 = fitness(tdata, method="sbd", random_state=123, copy=True)
    df2 = fitness(tdata, method="sbd", random_state=123, copy=True)
    # Same seed should give identical series for all nodes
    pd.testing.assert_frame_equal(df1.sort_index(), df2.sort_index())


if __name__ == "__main__":
    pytest.main(["-v", __file__])
