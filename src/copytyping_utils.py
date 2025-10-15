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
    subclonal_mask = np.copy(tumor_mask)
    neutral_mask = np.all(A == 1, axis=1) & np.all(B == 1, axis=1)
    if A.shape[1] > 2:
        subclonal_mask = np.any(A[:, 2:] != A[:, 1][:, None], axis=1) | np.any(B[:, 2:] != B[:, 1][:, None], axis=1)
    if not and_mask is None:
        tumor_mask &= and_mask
        clonal_loh_mask &= and_mask
        ai_mask &= and_mask
        subclonal_mask &= and_mask
    return {"CNP": tumor_mask, 
            "IMBALANCED": ai_mask,
            "SUBCLONAL": subclonal_mask,
            "CLONAL_LOH": clonal_loh_mask,
            "NEUTRAL": neutral_mask}
