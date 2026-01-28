from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import networkx as nx
import requests
import treedata as td

from pycea.utils import get_leaves

if TYPE_CHECKING:
    from os import PathLike

DATASET_DIR = "~/.treedata/datasets"
ZENODO_DOI = "15750529"  # Needs to be updated if the dataset is changed


def _load_dataset(
    name: str, cache_dir: PathLike | str, backup_url: str | None = None, force_download: bool = False
) -> td.TreeData:
    """Load a dataset from the cache or download it if not present."""
    cache_dir = Path(cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = cache_dir / name
    if not filename.exists():
        print(f"Downloading dataset {name} from {backup_url}")
        r = requests.get(str(backup_url), stream=True)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Using cached dataset {name} from {filename}")
    return td.read_h5td(filename)


def _prune_tree(tree: nx.DiGraph, nodes: set[str]) -> nx.DiGraph:
    """Prune a tree to keep only the specified nodes and their ancestors."""
    tree = nx.DiGraph(tree.copy())
    keep = set(nodes)
    for n in nodes:
        keep |= nx.ancestors(tree, n)
    tree.remove_nodes_from(set(tree.nodes) - keep)
    return tree


def packer19(cache_dir: PathLike | str = DATASET_DIR, tree: Literal["full", "observed"] = "full") -> td.TreeData:
    """C elegans lineage tree with cell state and position :cite:p:`Packer_2019`.

    In this study, single-cell RNA sequencing (scRNA-seq) was performed on C. elegans
    embryos at various developmental stages. Using computational methods, gene expression
    patterns from the literature, and fluorescent reporter lines, single cells were then mapped
    to their position in the known C. elegans lineage tree. This dataset contains the average
    expression of each cell across the lineage tree with corresponding spatial coordinates
    from :cite:p:`Richards_2013`.


    Parameters
    ----------
    cache_dir
        The directory where the datasets are cached. Default is `~/.treedata/datasets`.
    tree
        The tree to load. If "full", the full lineage tree is used. If "observed", the tree is pruned to only include
        lineages that are resolved by Packer et al. 2019 at the 400 minute time point.

    Returns
    -------
    TreeData object.

    """
    tdata = _load_dataset(
        "packer19.h5td",
        cache_dir=cache_dir,
        backup_url=f"https://zenodo.org/records/{ZENODO_DOI}/files/packer19.h5td?download=1",
    )
    if tree == "observed":
        tdata.obst["tree"] = _prune_tree(
            tdata.obst["tree"], set(tdata.obs.query("~dies").index) & set(get_leaves(tdata.obst["tree"]))
        )
    return tdata


def yang22(tumors: str | list[str] | None = "3435_NT_T1", cache_dir: PathLike | str = DATASET_DIR) -> td.TreeData:
    """Single-cell lineage tracing from KP mouse model :cite:p:`Yang_2022`.

    In this study, CRISPR Cas9-based single-cell lineage tracing was performed in the KP
    autochthonous mouse model of non-small-cell lung cancer. Tumors were initiated
    with Cre recombinase and allowed to grow for approximately 4-6 months at
    which point mice were sacrificed and tumors harvested. After purifying cancer
    cells by fluorescent markers, cells were profiled with the 10X chromium platform.

    Parameters
    ----------
    tumors
        The set of tumors to load. Default is "3435_NT_T1".
    cache_dir
        The directory where the datasets are cached. Default is `~/.treedata/datasets`.

    Returns
    -------
    TreeData object.

    """
    tdata = _load_dataset(
        "yang22.h5td",
        cache_dir=cache_dir,
        backup_url=f"https://zenodo.org/records/{ZENODO_DOI}/files/yang22.h5td?download=1",
    )
    if tumors is not None:
        if isinstance(tumors, str):
            tumors = [tumors]
        elif not isinstance(tumors, list):
            raise ValueError("tumors must be a string or a list of strings.")
        print(f"Subsetting to tumors: {', '.join(tumors)}")
        tdata = tdata[tdata.obs["tumor"].isin(tumors)].copy()
        keys_to_delete = [key for key, value in tdata.obst.items() if value.size() == 0]
        for key in keys_to_delete:
            del tdata.obst[key]
    return tdata


def koblan25(experiment: str = "tumor", cache_dir: PathLike | str = DATASET_DIR) -> td.TreeData:
    """Spatially resolved lineage tracing of 4T1 tumors :cite:p:`Koblan_2025`.

    This study presents PEtracer, a novel prime editing-based lineage tracer that can be
    read out using either scRNA-seq or spatial imaging. PEtracer is validated in vitro
    using sequential rounds of static barcoding and applied in vivo to study the
    4T1 transplantable mouse breast cancer model. The tumor dataset contains
    malignant cells with lineage information as well as stromal and immune cells.

    Parameters
    ----------
    experiment

        - "tumor": Loads in vivo data from mouse 3 tumor 1.
        - "barcoding": Loads in vitro barcoding data from clone 4.
    cache_dir
        The directory where the datasets are cached. Default is `~/.treedata/datasets`.

    Returns
    -------
    TreeData object.

    """
    if experiment not in {"tumor", "barcoding"}:
        raise ValueError('experiment must be either "tumor" or "barcoding".')
    tdata = _load_dataset(
        f"koblan25_{experiment}.h5td",
        cache_dir=cache_dir,
        backup_url=f"https://zenodo.org/records/{ZENODO_DOI}/files/koblan25_{experiment}.h5td?download=1",
    )
    return tdata
