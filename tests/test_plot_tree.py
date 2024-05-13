from pathlib import Path

import matplotlib.pyplot as plt

import pycea

plot_path = Path(__file__).parent / "plots"


def test_plot_branches(tdata):
    # Polar categorical with missing
    pycea.pl.branches(tdata, key="tree", polar=True, color="clade", palette="Set1")
    plt.savefig(plot_path / "polar_categorical_branches.png")
    plt.close()
    # Numeric with line width
    pycea.pl.branches(
        tdata, key="tree", polar=False, color="length", cmap="hsv", linewidth="length", angled_branches=True
    )
    plt.savefig(plot_path / "angled_numeric_branches.png")
    plt.close()
