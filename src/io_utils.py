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
# preprocess IOs
##################################################

# Load DNA bulk data
def load_snps_HATCHet_old(
    phased_snps: pd.DataFrame,
    hatchet_files: list,
):
    [tumor_1bed_file] = hatchet_files
    snp_info = read_baf_file(tumor_1bed_file)
    snp_info = pd.merge(
        left=snp_info, right=phased_snps, on=["#CHR", "POS"], how="left"
    )
    snp_info.dropna(subset=["GT"], inplace=True)
    # assume one tumor sample for now
    ref_counts = snp_info["REF"].to_numpy().astype(np.int32)
    alt_counts = snp_info["ALT"].to_numpy().astype(np.int32)
    allale_counts = ref_counts + alt_counts
    return snp_info, allale_counts, ref_counts, alt_counts

def load_snps_HATCHet_new(
    phased_snps: pd.DataFrame,
    hatchet_files: list,
):
    [allele_dir] = hatchet_files
    snp_ifile = os.path.join(allele_dir, "snp_info.tsv.gz")
    ref_mfile = os.path.join(allele_dir, "snp_matrix.ref.npz")
    alt_mfile = os.path.join(allele_dir, "snp_matrix.alt.npz")
    snp_info = pd.read_table(snp_ifile, sep="\t")
    snp_info = pd.merge(
        left=snp_info, right=phased_snps, on=["#CHR", "POS"], how="left"
    )
    assert np.all(~pd.isna(snp_info["GT"])), (
        "invalid input, only phased SNPs should present here"
    )
    ref_mat = np.load(ref_mfile)["mat"].astype(np.int32)
    alt_mat = np.load(alt_mfile)["mat"].astype(np.int32)
    ref_counts = ref_mat[:, 1]
    alt_counts = alt_mat[:, 1]
    allale_counts = ref_counts + alt_counts
    return snp_info, allale_counts, ref_counts, alt_counts

def load_snps_pseudobulk(
    allele_infos: dict,
    modality: str
):
    assert modality == "visium", "todo"
    snp_info, allele_counts_pb, ref_counts_pb, alt_counts_pb = allele_infos["visium"]
    snp_info = snp_info.copy(deep=True)
    allele_counts = allele_counts_pb.sum(axis=1).toarray().astype(dtype=np.int32)
    ref_counts = ref_counts_pb.sum(axis=1).toarray().astype(dtype=np.int32)
    alt_counts = alt_counts_pb.sum(axis=1).toarray().astype(dtype=np.int32)
    return snp_info, allele_counts, ref_counts, alt_counts

##################################################
# Load Allele count data
def load_cellsnp_files(
    cellsnp_dir: str,
    snp_info: pd.DataFrame,
    barcodes: list,
):
    print(f"load cell-snp files from {cellsnp_dir}")
    barcode_file = os.path.join(cellsnp_dir, "cellSNP.samples.tsv")
    vcf_file = os.path.join(cellsnp_dir, "cellSNP.base.vcf.gz")
    dp_file = os.path.join(cellsnp_dir, "cellSNP.tag.DP.mtx")
    ad_file = os.path.join(cellsnp_dir, "cellSNP.tag.AD.mtx")

    raw_barcodes = read_barcodes(barcode_file)  # assume no header
    barcode_indices = np.array([raw_barcodes.index(x) for x in barcodes])
    dp_mat: csr_matrix = mmread(dp_file).tocsr()
    alt_mat: csr_matrix = mmread(ad_file).tocsr()
    ref_mat = dp_mat - alt_mat

    dp_mat = dp_mat[:, barcode_indices]
    alt_mat = alt_mat[:, barcode_indices]
    ref_mat = ref_mat[:, barcode_indices]

    cell_snps = read_VCF_cellsnp_err_header(vcf_file)
    cell_snps["RAW_SNP_IDX"] = np.arange(len(cell_snps))  # use to index matrix
    if not snp_info is None:
        cell_snps = pd.merge(
            left=cell_snps,
            right=snp_info[["#CHR", "POS", "POS0", "PHASE", "HB"]],
            on=["#CHR", "POS"],
            how="left",
        )
        # some cell-snp SNPs may outside CNV segments, due to post-filtering in HATCHet2
        cell_snps = cell_snps.loc[cell_snps["PHASE"].notna(), :]

        cell_snps["PHASE"] = cell_snps["PHASE"].astype(np.float32)
        cell_snps["HB"] = cell_snps["PHASE"].astype(np.int32)
        cell_snps["POS0"] = cell_snps["POS0"].astype(cell_snps["POS"].dtype)

    return [cell_snps, dp_mat, ref_mat, alt_mat]

def load_calicost_prep_data(calicost_prep_dir: str, barcodes: list):
    print(f"load allele count data from CalicoST preprocessed files")
    barcode_file = os.path.join(calicost_prep_dir, "barcodes.txt")
    a_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Aallele.npz")
    b_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Ballele.npz")
    usnp_id_file = os.path.join(calicost_prep_dir, "unique_snp_ids.npy")

    raw_barcodes = read_barcodes(barcode_file)
    barcode_indices = np.array([raw_barcodes.index(x) for x in barcodes])
    alt_mat = sparse.load_npz(a_mtx_file)
    ref_mat = sparse.load_npz(b_mtx_file)
    dp_mat = alt_mat + ref_mat

    dp_mat = dp_mat[:, barcode_indices]
    alt_mat = alt_mat[:, barcode_indices]
    ref_mat = ref_mat[:, barcode_indices]

    unique_snp_ids = np.load(usnp_id_file, allow_pickle=True)
    cell_snps = pd.DataFrame([x.split("_") for x in unique_snp_ids],
                  columns=["#CHR", "POS", "REF", "ALT"])
    if not cell_snps["#CHR"].str.startswith("chr").any():
        cell_snps["#CHR"] = "chr" + cell_snps["#CHR"].astype(str)
    cell_snps["POS"] = cell_snps["POS"].astype(int)
    cell_snps["RAW_SNP_IDX"] = np.arange(len(cell_snps))  # use to index matrix
    print(f"#SNPs={len(cell_snps)}")

    # placeholders
    cell_snps["GT"] = 1
    cell_snps["PS"] = 1
    assert (len(barcodes), len(cell_snps)) == ref_mat.shape
    return [cell_snps, dp_mat, ref_mat, alt_mat]

##################################################
# copytyping IOs
##################################################
def load_bin_information(mod_dir: str, modality: str):
    bin_info_file = os.path.join(mod_dir, "bin_information.tsv")
    df = pd.read_table(bin_info_file, sep="\t")
    return df

def load_allele_input(mod_dir: str, modality: str):
    bin_Aallele_file = os.path.join(mod_dir, "Aallele.npz")
    bin_Ballele_file = os.path.join(mod_dir, "Ballele.npz")
    bin_Tallele_file = os.path.join(mod_dir, "Tallele.npz")
    bin_nSNP_file = os.path.join(mod_dir, "n_snps.npz")

    a_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Aallele_file).toarray().astype(dtype=np.int32)
    )
    b_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Ballele_file).toarray().astype(dtype=np.int32)
    )
    t_allele_mat: np.ndarray = (
        sparse.load_npz(bin_Tallele_file).toarray().astype(dtype=np.int32)
    )
    snp_count_mat: np.ndarray = (
        sparse.load_npz(bin_nSNP_file).toarray().astype(dtype=np.int32)
    )
    return a_allele_mat, b_allele_mat, t_allele_mat, snp_count_mat

def load_count_input(mod_dir: str, modality: str):
    bin_count_file = os.path.join(mod_dir, f"count.npz")
    bin_count_mat: np.ndarray = (
        sparse.load_npz(bin_count_file).toarray().astype(dtype=np.int32)
    )
    return bin_count_mat

def load_spatial_coordinates(coord_file: str):
    pass
