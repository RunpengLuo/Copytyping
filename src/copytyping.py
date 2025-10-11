import os
import sys

import numpy as np
import pandas as pd

import scanpy as sc

from utils import *
from raw_io_utils import *

from parsing import parse_arguments_copytyping
from sc_model import SC_Model
from plot_utils import *

if __name__ == "__main__":
    args = parse_arguments_copytyping()

    mode = args["mode"]
    sample = args["sample"]
    work_dir = args["work_dir"]
    out_prefix = args["out_prefix"]

    prep_dir = os.path.join(work_dir, "preprocess")
    out_dir = os.path.join(work_dir, f"{out_prefix}_assignment")
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    plot_dir = os.path.join(work_dir, f"{out_prefix}_plot")
    os.makedirs(plot_dir, exist_ok=True)
    sc.settings.figdir = plot_dir

    ##################################################
    # load data
    barcode_file = os.path.join(prep_dir, "Barcodes.tsv")
    barcodes = pd.read_table(barcode_file, sep="\t")
    print(f"#barcodes={len(barcodes)}")

    if mode in ["multiome", "GEX", "ATAC"]:
        sc_model = SC_Model(barcodes, prep_dir, out_dir, mode)
        params = sc_model.inference()
        anns = sc_model.map_decode(params, label="cell_label", thres=args["map_thres"])
        anns.to_csv(os.path.join(out_dir, "annotations.tsv"), sep="\t", header=True, index=False)

        for modality in sc_model.data_types:
            sx_data: SX_Data = sc_model.data_sources[modality]
            plot_baf_1d(sc_model.data_sources["GEX"], anns, sample, 
                        filename=f"{sample}.BAF_heatmap.{modality}.cell_label", lab_type="cell_label")

            plot_rdr_1d(sc_model.data_sources["GEX"], anns, sample, 
                        filename=f"{sample}.RDR_heatmap.{modality}.cell_label", lab_type="cell_label")
            
            if "cell_type" in anns:
                plot_baf_1d(sc_model.data_sources["GEX"], anns, sample, 
                        filename=f"{sample}.BAF_heatmap.{modality}.cell_type", lab_type="cell_type")

                plot_rdr_1d(sc_model.data_sources["GEX"], anns, sample, 
                        filename=f"{sample}.RDR_heatmap.{modality}.cell_type", lab_type="cell_type")
