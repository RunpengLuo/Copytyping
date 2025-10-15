import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData
import squidpy as sq
import snapatac2 as snap

from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse

from utils import *
from external import *

##################################################
def annotate_snps(
    segs: pd.DataFrame,
    snp_info: pd.DataFrame,
    allele_counts: np.ndarray,
    ref_counts: np.ndarray,
    alt_counts: np.ndarray,
):
    assert len(ref_counts) == len(alt_counts)
    assert len(ref_counts) == len(snp_info)
    snp_info["POS0"] = snp_info["POS"] - 1

    snp_info = annotate_snps_seg_idx(segs, snp_info, "SEG_IDX")
    na_snps = snp_info["SEG_IDX"].isna().to_numpy()
    num_na_snps = np.sum(na_snps)
    print(
        f"#SNPs outside CNV segments: {num_na_snps}/{len(snp_info)}={num_na_snps / len(snp_info):.3%}"
    )
    allele_counts = allele_counts[~na_snps]
    ref_counts = ref_counts[~na_snps]
    alt_counts = alt_counts[~na_snps]
    snp_info.dropna(subset="SEG_IDX", inplace=True)
    snp_info["SEG_IDX"] = snp_info["SEG_IDX"].astype(segs.index.dtype)
    snp_info = snp_info.reset_index(drop=True)

    snp_info["PHASE"] = snp_info["GT"].astype(np.float32)
    snp_info["START"] = snp_info["POS0"]
    snp_info["END"] = snp_info["POS"]
    snp_info["DP"] = allele_counts
    return snp_info, allele_counts, ref_counts, alt_counts

def annotate_snps_post(
    snp_info: pd.DataFrame,
    allele_infos: dict,
):
    """
    any SNPs that are unphased are discarded.
    """
    for data_type in allele_infos.keys():
        cell_snps = allele_infos[data_type][0]
        cell_snps = cell_snps.reset_index(drop=False).merge(
            right=snp_info[["#CHR", "POS", "POS0", "PHASE", "HB"]],
            on=["#CHR", "POS"],
            how="left",
            sort=False
        ).set_index("index")

        # some SNPs may outside CNV segments or unphased
        cell_snps = cell_snps.loc[cell_snps["PHASE"].notna(), :]

        cell_snps["PHASE"] = cell_snps["PHASE"].astype(np.float32)
        cell_snps["HB"] = cell_snps["PHASE"].astype(np.int32)
        cell_snps["POS0"] = cell_snps["POS0"].astype(cell_snps["POS"].dtype)
        allele_infos[data_type][0] = cell_snps
    return allele_infos


##################################################
def consolidate_snp_feature(
    adata: AnnData,
    snp_df: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    tmp_dir: str,
    modality: str,
    feature_may_overlap=True,
):
    print(f"consolidate SNPs with features for {modality}")

    ##################################################
    # union all overlapping features, build var_super_df
    adata.var["unique_index"] = np.arange(len(adata.var))

    if feature_may_overlap:
        print("detect overlapping features")
        tmp_feature_in_file = os.path.join(tmp_dir, f"tmp_{modality}.in.bed")
        tmp_feature_out_file = os.path.join(tmp_dir, f"tmp_{modality}.out.bed")
        adata.var.to_csv(
            tmp_feature_in_file,
            sep="\t",
            header=False,
            index=False,
            columns=["#CHR", "START", "END", "unique_index"],
        )

        var_df_clustered = run_bedtools_cluster(
            tmp_feature_in_file,
            tmp_feature_out_file,
            tmp_dir,
            max_dist=0,
            load_df=True,
            usecols=list(range(5)),
            names=["#CHR", "START", "END", "unique_index", "SUPER_VAR_IDX"],
        )
        adata.var = (
            adata.var.reset_index(drop=False)
            .merge(
                right=var_df_clustered[["unique_index", "SUPER_VAR_IDX"]],
                on="unique_index",
                how="left",
            )
            .set_index("index")
        )
    else:
        adata.var["SUPER_VAR_IDX"] = adata.var["unique_index"]

    var_supers = adata.var.groupby(by="SUPER_VAR_IDX", sort=False, as_index=True)
    var_super_df = var_supers.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
        }
    ).reset_index(drop=False)
    print(f"#{modality}-super-feature={len(var_super_df)}")

    ##################################################
    # assign HB tag to var_super_df
    var_super_df = assign_largest_overlap(
        var_super_df, haplo_blocks, "SUPER_VAR_IDX", "HB"
    )
    isna_super_var = var_super_df["HB"].isna()
    # this is mostly due to centromere/satellite-repeat mask.
    print(
        f"#{modality}-super-feature outside any haplotype blocks={np.sum(isna_super_var) / len(var_super_df):.3%}"
    )

    var_super_df.dropna(subset="HB", inplace=True)
    var_super_df["HB"] = var_super_df["HB"].astype(np.int32)

    # map HB tag to adata.var, filter any features accordingly
    adata.var = (
        adata.var.reset_index(drop=False)
        .merge(
            right=var_super_df[["SUPER_VAR_IDX", "HB"]],
            on="SUPER_VAR_IDX",
            how="left",
        )
        .set_index("index")
    )

    adata = adata[:, adata.var["HB"].notna()].copy()
    adata.var["HB"] = adata.var["HB"].astype(np.int32)

    ##################################################
    # assign SNPs to super-features
    print(f"#{modality}-SNP={len(snp_df)}")
    snp_df = assign_pos_to_range(
        snp_df, var_super_df, ref_id="SUPER_VAR_IDX", pos_col="POS0"
    )
    isna_snp_df = snp_df["SUPER_VAR_IDX"].isna()

    print(
        f"#{modality}-SNPs outside any super-feature={np.sum(isna_snp_df) / len(snp_df):.3%}"
    )
    snp_df = snp_df.loc[snp_df["SUPER_VAR_IDX"].notna(), :].copy(deep=True)

    ##################################################
    print("collect pseudobulk super-feature allele counts and total counts")
    if not snp_df is None:
        pseudobulk_allele_counts = (
            snp_df.groupby("SUPER_VAR_IDX", sort=False)["DP"]
            .sum()
            .rename("ALLELE_DP")
            .reset_index()
        )
        var_super_df = pd.merge(
            left=var_super_df,
            right=pseudobulk_allele_counts,
            on="SUPER_VAR_IDX",
            how="left",
        )
        var_super_df["ALLELE_DP"] = var_super_df["ALLELE_DP"].fillna(0).astype(np.int32)

    if not adata is None:
        pseudobulk_super_counts = (
            adata.var[["SUPER_VAR_IDX", "pseudobulk_counts"]]
            .groupby("SUPER_VAR_IDX", sort=False)["pseudobulk_counts"]
            .sum()
            .rename("VAR_DP")
            .reset_index()
        )
        var_super_df = pd.merge(
            left=var_super_df,
            right=pseudobulk_super_counts,
            on="SUPER_VAR_IDX",
            how="left",
        )
        var_super_df["VAR_DP"] = var_super_df["VAR_DP"].fillna(0).astype(np.int32)
    return adata, snp_df, var_super_df


##################################################
# binning features and SNPs
def co_binning_allele_feature(
    adata: AnnData,
    cell_snps: pd.DataFrame,
    var_super_df: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    min_allele_counts: int,
    min_total_counts: int,
    min_total_threshold: int,
    min_allele_threshold: int,
):
    print(
        f"adaptive co-binning over features with min_allele_counts={min_allele_counts} and min_total_counts={min_total_counts}"
    )
    bin_colnames = ["ALLELE_DP", "VAR_DP"]
    bin_min_counts = np.array([min_allele_counts, min_total_counts])
    var_super_df["BIN_ID"] = 0
    bin_id = 0
    hb_ids = var_super_df["HB"].unique()
    var_super_hbs = var_super_df.groupby(by="HB", sort=False)
    for hb in hb_ids:
        var_super_hb = var_super_hbs.get_group(hb)
        bin_id = adaptive_co_binning(
            var_super_df,
            var_super_hb,
            "BIN_ID",
            bin_colnames,
            bin_min_counts,
            s_block_id=bin_id,
        )

    var_super_bins = var_super_df.groupby(by="BIN_ID", sort=False, as_index=True)
    var_bins = var_super_bins.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "HB": ("HB", "first"),
            "ALLELE_DP": ("ALLELE_DP", "sum"),
            "VAR_DP": ("VAR_DP", "sum"),
        }
    ).reset_index(drop=False)
    var_bins.loc[:, "#VAR"] = var_super_bins.size().reset_index(drop=True)
    var_bins.loc[:, "BLOCKSIZE"] = var_bins["END"] - var_bins["START"]

    ##################################################
    # filter bins that has total pseudobulk count below <min_total_threshold>
    num_bins_raw = len(var_bins)
    var_bins = var_bins.loc[
        (var_bins["VAR_DP"] >= min_total_threshold)
        & (var_bins["ALLELE_DP"] >= min_allele_threshold),
        :,
    ].reset_index(drop=True)
    var_super_df = var_super_df.loc[
        var_super_df["BIN_ID"].isin(var_bins["BIN_ID"]), :
    ].copy(deep=True)
    var_super_df["BIN_ID"] = pd.factorize(var_super_df["BIN_ID"])[0]  # reset bin index

    num_bins = len(var_bins)
    var_bins["BIN_ID"] = np.arange(num_bins)  # reset bin index
    print(f"#valid bins={num_bins}/{num_bins_raw}={num_bins / num_bins_raw:.3%}")

    adata.var = (
        adata.var.reset_index(drop=False)
        .merge(
            right=var_super_df[["SUPER_VAR_IDX", "BIN_ID"]],
            on="SUPER_VAR_IDX",
            how="left",
            sort=False,
        )
        .set_index("index")
    )
    adata = adata[:, adata.var["BIN_ID"].notna()].copy()
    adata.var["BIN_ID"] = adata.var["BIN_ID"].astype(int)

    cell_snps = (
        cell_snps.reset_index(drop=False)
        .merge(
            right=var_super_df[["SUPER_VAR_IDX", "BIN_ID"]],
            on="SUPER_VAR_IDX",
            how="left",
            sort=False,
        )
        .set_index("index")
    )
    cell_snps = cell_snps.loc[cell_snps["BIN_ID"].notna(), :].copy(deep=True)
    cell_snps["BIN_ID"] = cell_snps["BIN_ID"].astype(int)

    ##################################################
    # append copy-number profile to bins
    var_bins = (
        var_bins.reset_index(drop=False)
        .merge(right=haplo_blocks[["HB", "CNP"]], on=["HB"], how="left", sort=False)
        .set_index("index")
    )
    assert var_bins["CNP"].notna().all(), "corrupted data"

    return adata, cell_snps, var_bins


##################################################
def aggregate_allele_counts(
    var_bins: pd.DataFrame,
    cell_snps: pd.DataFrame,
    dp_mtx: csr_matrix,
    ad_mtx: csr_matrix,
    modality: str,
    out_dir: str,
    v=1,
):
    print(f"aggregate allele counts per bin for {modality}")
    num_bins = len(var_bins)
    num_barcodes = dp_mtx.shape[1]

    # outputs
    b_allele_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)
    t_allele_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)
    # num.snps with nonzero DP
    snp_count_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)

    bin_ids = cell_snps["BIN_ID"].unique()
    cell_snps_bins = cell_snps.groupby(by="BIN_ID", sort=False, as_index=True)
    for bin_id in bin_ids:
        snps_bin = cell_snps_bins.get_group(bin_id)
        # original index in raw allele count matrix
        snp_indices = snps_bin["RAW_SNP_IDX"].to_numpy()  # (n, )
        snp_phases = snps_bin["PHASE"].to_numpy()[:, np.newaxis]  # (n, 1)

        # access allele-count matrix
        rows_dp = dp_mtx[snp_indices].toarray()
        rows_alt = ad_mtx[snp_indices].toarray()
        rows_ref = rows_dp - rows_alt

        # aggregate phased counts
        rows_beta = rows_alt * (1 - snp_phases) + rows_ref * snp_phases
        b_allele_mat[bin_id, :] = np.round(np.sum(rows_beta, axis=0))
        t_allele_mat[bin_id, :] = np.sum(rows_dp, axis=0)
        snp_count_mat[bin_id, :] = np.sum(rows_dp > 0, axis=0)
    a_allele_mat = (t_allele_mat - b_allele_mat).astype(np.int32)

    # append pseudobulk b-allele counts
    var_bins["B_ALLELE_DP"] = np.sum(b_allele_mat, axis=1)

    bin_Aallele_file = os.path.join(out_dir, f"Aallele.npz")
    bin_Ballele_file = os.path.join(out_dir, f"Ballele.npz")
    bin_Tallele_file = os.path.join(out_dir, f"Tallele.npz")
    bin_nSNP_file = os.path.join(out_dir, f"n_snps.npz")

    sparse.save_npz(bin_Aallele_file, csr_matrix(a_allele_mat))
    sparse.save_npz(bin_Ballele_file, csr_matrix(b_allele_mat))
    sparse.save_npz(bin_Tallele_file, csr_matrix(t_allele_mat))
    sparse.save_npz(bin_nSNP_file, csr_matrix(snp_count_mat))
    return


def aggregate_var_counts(
    var_bins: pd.DataFrame,
    adata: AnnData,
    modality: str,
    out_dir: str,
):
    print(f"aggregate total counts per bin for {modality}")

    n_cells, n_feats = adata.n_obs, adata.n_vars
    n_bins = len(var_bins)
    feat_bin = adata.var["BIN_ID"].to_numpy()
    indicator_matrix = sparse.csr_matrix(
        (np.ones_like(feat_bin, dtype=np.int8), (np.arange(n_feats), feat_bin)),
        shape=(n_feats, n_bins),
    )
    X = adata.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)

    bin_count_mat = X @ indicator_matrix  # shape: (n_cells × n_bins)
    bin_count_mat = bin_count_mat.T.tocsr()  # to (n_bins × n_cells) for consistency

    bin_count_file = os.path.join(out_dir, f"count.npz")
    sparse.save_npz(bin_count_file, bin_count_mat)
    return
