import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc
from scanpy import AnnData
import squidpy as sq

from scipy.io import mmread
from scipy.sparse import csr_matrix
from scipy import sparse

from utils import *
from external import *


def load_bin_information(mod_dir: str, modality: str):
    bin_info_file = os.path.join(mod_dir, "bin_information.tsv")
    df = pd.read_table(bin_info_file, sep="\t")
    return df


def parse_cnp(bin_info: pd.DataFrame, laplace=0.01):
    num_clones = len(str(bin_info["CNP"].iloc[0]).split(";"))
    clones = ["normal"] + [f"clone{i}" for i in range(1, num_clones)]
    A = np.zeros((len(bin_info), num_clones), dtype=np.int32)
    B = np.zeros((len(bin_info), num_clones), dtype=np.int32)
    for i in range(num_clones):
        A[:, i] = bin_info.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[0]), axis=1
        ).to_numpy()
        B[:, i] = bin_info.apply(
            func=lambda r: int(r["CNP"].split(";")[i].split("|")[1]), axis=1
        ).to_numpy()
    C = A + B
    BAF = np.divide(
        B + laplace,
        C + laplace * 2,
        out=np.zeros_like(C, dtype=np.float32),
        where=(C > 0),
    )

    # assign the CNP group id
    bin_info["CNP_ID"] = bin_info.groupby("CNP", sort=False).ngroup()
    cnp_id2state = bin_info.groupby("CNP_ID", sort=False)["CNP"].first().to_dict()
    return clones, cnp_id2state, A, B, C, BAF


def get_cnp_mask(A, B, C, BAF, and_mask=None):
    """return 1d mask, False if the bin should be discarded during modelling"""
    tumor_mask = np.any(A != 1, axis=1) | np.any(
        B != 1, axis=1
    )  # not purely normal cell
    ai_mask = np.any(BAF != 0.5, axis=1)  # at least one clone is allelic imbalanced
    clonal_loh_mask = np.all(B[:, 1:] == 0, axis=1) & np.all(A[:, 1:] > 0, axis=1)

    if not and_mask is None:
        tumor_mask &= and_mask
        clonal_loh_mask &= and_mask
        ai_mask &= and_mask
    return {"CNP": tumor_mask, 
            "IMBALANCED": ai_mask,
            "CLONAL_LOH": clonal_loh_mask}


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
