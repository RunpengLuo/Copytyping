import os
import sys

import pandas as pd
import numpy as np
import scanpy as sc
from scanpy import AnnData

from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sx_data import SX_Data

##################################################
def cluster_per_group(
    adata: AnnData,
    cluster_chroms=None,
    groupby="cell_label"
):
    # cluster within groups
    if cluster_chroms is not None:
        chrom_mask = adata.var["#CHR"].isin(cluster_chroms)
    else:
        chrom_mask = np.ones(adata.n_vars, dtype=bool)
    order_indices = []
    groups = adata.obs[groupby]
    for cat in groups.unique():
        cell_mask = groups == cat
        if np.sum(cell_mask) == 0:
            continue
        X_group = adata.X[cell_mask][:, chrom_mask]
        X_group = np.nan_to_num(X_group, nan=0.5)
        if X_group.shape[0] > 2:
            Z = linkage(X_group, method="ward", metric="euclidean")
            leaf_order = leaves_list(Z)
            order_indices.extend(np.where(cell_mask)[0][leaf_order])
        else:
            order_indices.extend(np.where(cell_mask)[0])
    return adata[order_indices, :].copy()

##################################################
# plot 1D cell/spot by genome bin heatmap for one modality
def plot_baf_1d(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    sample: str,
    data_type: str,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 10),
    filename=None,
    **kwargs
):
    bin_info = sx_data.bin_info
    if mask_cnp:
        bin_info = bin_info.loc[sx_data.ALL_MASK[mask_id], :]

    # BAF data
    Y = sx_data.Y
    D = sx_data.D
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=np.nan, dtype=np.float32), where=D > 0
    )
    if mask_cnp:
        baf_matrix = baf_matrix[sx_data.ALL_MASK[mask_id]]

    cell_labels = anns[lab_type].tolist()
    assert (len(bin_info), len(cell_labels)) == baf_matrix.shape
    baf_matrix = baf_matrix.T

    # build anndata
    adata = AnnData(X=baf_matrix)
    adata.obs[lab_type] = cell_labels
    adata.var[['#CHR','START','END']] = bin_info[['#CHR','START','END']].values
    adata_sorted = cluster_per_group(adata, cluster_chroms=None, groupby=lab_type)
    
    chroms = adata_sorted.var["#CHR"].to_numpy()
    chr_change_idx = np.where(chroms[1:] != chroms[:-1])[0] + 1
    chr_pos = [0] + chr_change_idx.tolist()
    var_group_labels = list(chroms[chr_pos])
    var_group_positions = [
        (chr_pos[i], chr_pos[i + 1] if i + 1 < len(chr_pos) else len(bin_info))
        for i in range(len(chr_pos))
    ]

    
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "baf_map",
        [(0.0, "blue"), (0.5, "green"), (1.0, "red")]
    )
    cmap.set_bad(color="white")  # NaNs pure white
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)

    ax_dict = sc.pl.heatmap(
        adata,
        var_names=adata.var_names,
        groupby=lab_type,
        figsize=figsize,
        cmap=cmap,
        norm=norm,
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        dendrogram=False,
        show=False,
        **kwargs,
    )

    ax_dict["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=0, ymax=adata.shape[0], color="black")
    ax_dict["heatmap_ax"].set_title(f"{sample} {data_type} BAF Heatmap", y=1.10)
    if not filename is None:
        sc.pl._utils.savefig(filename, dpi=300)
        plt.close()
        return
    plt.show()

def plot_rdr_1d(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    sample: str,
    data_type: str,
    base_props: np.ndarray,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 10),
    filename=None,
    verbose=1,
    **kwargs
):
    bin_info = sx_data.bin_info
    rdr_matrix = (sx_data.T / (base_props[:, None] @ sx_data.Tn[None, :])).T
    if mask_cnp:
        cnp_mask = sx_data.ALL_MASK[mask_id]
        bin_info = bin_info.loc[cnp_mask, :]
        rdr_matrix = rdr_matrix[:, cnp_mask]
    
    if verbose:
        print(f"before log2 transform median={np.median(rdr_matrix)} max={np.max(rdr_matrix)}")
    rdr_matrix = np.log2(np.clip(rdr_matrix, a_min=1e-6, a_max=np.inf))
    if verbose:
        print(f"after log2 transform median={np.median(rdr_matrix)} max={np.max(rdr_matrix)}")

    cell_labels = anns[lab_type].tolist()
    assert (len(cell_labels), len(bin_info)) == rdr_matrix.shape

    # build anndata
    adata = AnnData(X=rdr_matrix)
    adata.obs[lab_type] = cell_labels
    adata.var[['#CHR','START','END']] = bin_info[['#CHR','START','END']].values
    adata_sorted = cluster_per_group(adata, cluster_chroms=None, groupby=lab_type)
    
    chroms = adata_sorted.var["#CHR"].to_numpy()
    chr_change_idx = np.where(chroms[1:] != chroms[:-1])[0] + 1
    chr_pos = [0] + chr_change_idx.tolist()
    var_group_labels = list(chroms[chr_pos])
    var_group_positions = [
        (chr_pos[i], chr_pos[i + 1] if i + 1 < len(chr_pos) else len(bin_info))
        for i in range(len(chr_pos))
    ]

    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    ax_dict = sc.pl.heatmap(
        adata,
        var_names=adata.var_names,
        groupby=lab_type,
        figsize=figsize,
        cmap="seismic",
        norm=norm,
        show_gene_labels=False,
        var_group_positions=var_group_positions,
        var_group_labels=var_group_labels,
        dendrogram=False,
        show=False,
        **kwargs,
    )

    ax_dict["heatmap_ax"].vlines(chr_pos[1:], lw=0.6, ymin=0, ymax=adata.shape[0], color="black")
    ax_dict["heatmap_ax"].set_title(f"{sample} {data_type} log2-RDR Heatmap", y=1.10)
    if not filename is None:
        sc.pl._utils.savefig(filename, dpi=300)
        plt.close()
        return
    plt.show()

##################################################
# plot parameters
def plot_baseline_proportions(params: dict, out_file: str, data_type: str):
    base_props = params[f"{data_type}-lambda"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax.hist(x=base_props, bins=50)
    title = f"{data_type} baseline proportions mean={base_props.mean():.3f} std={base_props.std():.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    return

def plot_dispersions(params: dict, out_file: str, data_type: str, name="tau"):
    dispersions = params[f"{data_type}-{name}"]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
    ax.hist(x=dispersions, bins=50)
    title = f"{data_type} dispersion-{name} mean={dispersions.mean():.3f} std={dispersions.std():.3f}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=300)
    return


##################################################
# plot UMAP
def plot_umap_standard(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    sample: str,
    data_type: str,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(20, 10),
    filename=None,
    **kwargs
):
    
    pass

def plot_umap_baf(
    sx_data: SX_Data,
    anns: pd.DataFrame,
    sample: str,
    data_type: str,
    mask_cnp=True,
    mask_id="CNP",
    lab_type="cell_label",
    figsize=(12, 6),
    filename=None,
    **kwargs
):
    bin_info = sx_data.bin_info
    if mask_cnp:
        bin_info = bin_info.loc[sx_data.ALL_MASK[mask_id], :]
    
    # BAF data
    Y = sx_data.Y
    D = sx_data.D
    baf_matrix = np.divide(
        Y, D, out=np.full_like(D, fill_value=0.5, dtype=np.float32), where=D > 0
    )
    if mask_cnp:
        baf_matrix = baf_matrix[sx_data.ALL_MASK[mask_id]]
    
    cell_labels = anns[lab_type].tolist()
    assert (len(bin_info), len(cell_labels)) == baf_matrix.shape
    baf_matrix = baf_matrix.T

    # build anndata
    adata = AnnData(X=baf_matrix)
    adata.obs[lab_type] = cell_labels
    adata.obs["max_posterior"] = anns["max_posterior"].tolist()

    sc.pp.pca(adata)
    sc.pp.neighbors(adata, metric="euclidean")
    sc.tl.umap(adata)

    with PdfPages(filename) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sc.pl.umap(
            adata,
            color=lab_type,
            title=f"{sample} {data_type} UMAP\n{lab_type} annotation",
            show=False,
            ax=axes[0]
        )
        sc.pl.umap(
            adata,
            color="max_posterior",
            title=f"{sample} {data_type} UMAP\nmaximum posterior annotation",
            show=False,
            ax=axes[1]
        )
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        pdf.close()
    return

##################################################
# plot Visium spatial data

