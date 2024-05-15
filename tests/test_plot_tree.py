from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import pycea

plot_path = Path(__file__).parent / "plots"


def test_polar_with_clades(tdata):
    fig, ax = plt.subplots(dpi=300)
    pycea.pl.branches(tdata, key="tree", polar=True, color="clade", palette="Set1", na_color="black")
    pycea.pl.annotation(tdata, keys="clade")
    plt.savefig(plot_path / "polar_with_clades.png")
    plt.close()


def test_angled_numeric_annotations(tdata):
    fig, ax = plt.subplots(dpi=300)
    pycea.pl.branches(
        tdata, key="tree", polar=False, color="length", cmap="hsv", linewidth="length", angled_branches=True
    )
    pycea.pl.annotation(tdata, keys=["x", "y"], cmap="magma", width=0.1, gap=0.05)
    print(tdata.var_names)
    pycea.pl.annotation(tdata, keys=["0", "1", "2", "3", "4", "5"], label="genes")
    plt.savefig(plot_path / "angled_numeric_annotation.png")
    plt.close()


def test_matrix_annotation(tdata):
    fig, ax = plt.subplots(dpi=300)
    pycea.pl.branches(tdata, key="tree", polar=False, ax=ax)
    pycea.pl.annotation(tdata, keys=["clade"], ax=ax)
    pycea.pl.annotation(tdata, keys=["spatial_distance"], ax=ax)
    plt.savefig(plot_path / "matrix_annotation.png")
    plt.close()


def test_branches_invalid_input(tdata):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, key="tree", color=["bad"] * 5)
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, key="tree", linewidth=["bad"] * 5)
    # Can't plot polar with non-polar axis
    with pytest.raises(ValueError):
        pycea.pl.branches(tdata, key="tree", polar=True, ax=ax)
    plt.close()


def test_annotation_invalid_input(tdata):
    # Need to plot branches first
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys="clade")
    pycea.pl.branches(tdata, key="tree", ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys=None, ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys=False, ax=ax)
    with pytest.raises(ValueError):
        pycea.pl.annotation(tdata, keys="clade", label={}, ax=ax)
    plt.close()
