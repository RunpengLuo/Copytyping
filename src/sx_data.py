import os
import sys

import numpy as np
import pandas as pd

from utils import *

from io_utils import *
from copytyping_utils import *

class SX_Data:
    def __init__(self, nbarcodes: int, prep_dir: str, data_type: str) -> None:
        assert data_type in ["GEX", "ATAC", "VISIUM"], data_type
        mod_dir = os.path.join(prep_dir, data_type)
        assert os.path.isdir(mod_dir)

        self.bin_info = load_bin_information(mod_dir, data_type)
        self.ncols = self.num_barcodes = nbarcodes
        self.nrows = self.num_bins = len(self.bin_info)
        print(f"{data_type} matrix shape={self.nrows, self.ncols}")

        clones, cnp_id2state, A, B, C, BAF = parse_cnp(self.bin_info, laplace=0.01)
        self.clones = clones
        self.cnp_id2state = cnp_id2state
        self.K = self.num_clones = len(clones)
        self.A = A
        self.B = B
        self.C = C
        self.BAF = BAF

        self.ALL_MASK = get_cnp_mask(A, B, C, BAF, and_mask=None)
        self.nrows_eff = np.sum(self.ALL_MASK["CNP"]) # ignore masked bins.
        print(f"#effective CNA bins={self.nrows_eff}")

        self.cnp_groups = self.get_cnp_shared_ids(apply_cnp_mask=True, mask_id="CNP")
        print(f"#effective unique CNA states={len(self.cnp_groups)}")
        print(cnp_id2state.items())

        a_allele_mat, b_allele_mat, t_allele_mat, snp_count_mat = load_allele_input(
            mod_dir, data_type
        )
        assert a_allele_mat.shape == (self.nrows, self.ncols), a_allele_mat.shape
        assert b_allele_mat.shape == (self.nrows, self.ncols), b_allele_mat.shape
        assert t_allele_mat.shape == (self.nrows, self.ncols), t_allele_mat.shape
        assert snp_count_mat.shape == (self.nrows, self.ncols), snp_count_mat.shape
        self.X = a_allele_mat
        self.Y = b_allele_mat
        self.D = t_allele_mat
        # self.snp_count_mat = snp_count_mat

        count_mat = load_count_input(mod_dir, data_type)
        assert count_mat.shape == (self.nrows, self.ncols), count_mat.shape
        self.T = count_mat
        self.Tn = np.sum(count_mat, axis=0)
        print(f"{data_type} data is loaded")
        return

    def apply_cnp_mask_shallow(self, mask_id="CNP"):
        cnp_mask = self.ALL_MASK[mask_id]
        M = {
            "A": self.A[cnp_mask, :],
            "B": self.B[cnp_mask, :],
            "C": self.C[cnp_mask, :],
            "BAF": self.BAF[cnp_mask, :],
            "X": self.X[cnp_mask, :],
            "Y": self.Y[cnp_mask, :],
            "D": self.D[cnp_mask, :],
            "T": self.T[cnp_mask, :],
        }
        return M
    
    def subset_matrix(self, cnp_ids: np.ndarray):
        M = {
            "A": self.A[cnp_ids, :],
            "B": self.B[cnp_ids, :],
            "C": self.C[cnp_ids, :],
            "BAF": self.BAF[cnp_ids, :],
            "X": self.X[cnp_ids, :],
            "Y": self.Y[cnp_ids, :],
            "D": self.D[cnp_ids, :],
            "T": self.T[cnp_ids, :],
        }
        return M


    def get_cnp_shared_ids(self, apply_cnp_mask=True, mask_id="CNP"):
        """per CNP group, get bins index"""
        assert "CNP_ID" in self.bin_info
        if apply_cnp_mask:
            cnp_mask = self.ALL_MASK[mask_id]
            cnp_groups = self.bin_info.loc[cnp_mask, :].groupby("CNP_ID", sort=False).groups
        else:
            cnp_groups = self.bin_info.groupby("CNP_ID", sort=False).groups
        
        cnp_groups = {k: np.array(v) for k, v in cnp_groups.items()}
        print(cnp_groups.keys())
        return cnp_groups
