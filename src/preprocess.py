import os
import sys
import shutil

import numpy as np
import pandas as pd

import scanpy as sc

from utils import *
from raw_io_utils import *

from parsing import parse_arguments_preprocess

if __name__ == "__main__":
    args = parse_arguments_preprocess()

    min_allele_counts_ps = 5000
    trust_PS = True

    min_allele_counts_rna = 1500
    min_total_counts_rna = 3000

    min_allele_counts_atac = 2500
    min_total_counts_atac = 5000

    min_allele_counts_visium = 5000
    min_total_counts_visium = 5000

    modality = args["modality"]
    sample = args["sample"]
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    prep_dir = os.path.join(out_dir, "preprocess")
    os.makedirs(prep_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ##################################################
    # preprocess HATCHet inputs
    seg_ucn = args["seg_ucn"]
    vcf_file = args["vcf_file"]
    if args["allele_dir"]:
        hatchet_files = [args["allele_dir"]]
        hatchet_version = "new"
    else:
        hatchet_files = [args["tumor_1bed"]]
        hatchet_version = "old"

    haplo_block_file = os.path.join(prep_dir, "haplotype_blocks.tsv")
    snp_info_file = os.path.join(prep_dir, "snp_information.tsv.gz")
    if os.path.exists(haplo_block_file) and os.path.exists(snp_info_file):
        print("load existing prep-ed haplotype block and snp files")
        haplo_blocks = pd.read_table(haplo_block_file, sep="\t", index_col=None)
        snp_info = pd.read_table(snp_info_file, sep="\t", index_col=None)
    else:
        haplo_blocks, snp_info = load_input_HATCHet(
            seg_ucn,
            vcf_file,
            hatchet_version,
            hatchet_files,
            trust_PS,
            min_allele_counts_ps,
            prep_dir,
            verbose=True,
        )

    ##################################################
    # preprocess 10x inputs
    barcode_file = args["barcodes"]
    ranger_dir = args["ranger_dir"]
    annotation_file = args["annotation_file"]
    celltype_file = args["celltype_file"]

    barcodes = read_barcodes(barcode_file)
    print(f"#barcodes={len(barcodes)}")

    barcodes_df = pd.DataFrame(data={"BARCODE": barcodes})
    if not celltype_file is None:
        print("append cell-type information")
        celltypes = read_celltypes(celltype_file)
        barcodes_df = pd.merge(
            left=barcodes_df, right=celltypes[["BARCODE", "cell_type"]], how="left"
        )
        barcodes_df["cell_type"] = (
            barcodes_df["cell_type"].fillna(value="Unknown").astype(str)
        )

    barcodes_df.to_csv(
        os.path.join(prep_dir, "Barcodes.tsv"), sep="\t", index=False, header=True
    )

    if modality == "multiome":
        adata = load_input_10x(ranger_dir, barcodes, annotation_file, modality=modality)

        ##################################################
        # handle scRNA-seq data
        # gex_dir = os.path.join(prep_dir, "GEX")
        # os.makedirs(gex_dir, exist_ok=True)
        # rna_adata = adata[:, adata.var["feature_types"] == "Gene Expression"].copy()
        # print(f"#genes={rna_adata.shape[1]}")
        # rna_bc_idxs, rna_cell_snps, rna_dp_mtx, rna_ad_mtx = load_cellsnp_files(
        #     args["cellsnp_dir_1"], snp_info, barcodes
        # )
        # rna_adata, rna_cell_snps, rna_super_features = consolidate_snp_feature_RNA(
        #     rna_adata, rna_cell_snps, haplo_blocks, tmp_dir, "GEX"
        # )

        # rna_super_features = stat_pseudobulk(
        #     rna_adata, rna_cell_snps, rna_super_features
        # )

        # rna_adata, rna_cell_snps, rna_bins = co_binning_allele_feature(
        #     rna_adata,
        #     rna_cell_snps,
        #     rna_super_features,
        #     haplo_blocks,
        #     min_allele_counts_rna,
        #     min_total_counts_rna,
        # )

        # rna_cell_snps.to_csv("cell_snps.tsv",
        #     sep="\t",
        #     header=True,
        #     index=False,
        # )

        # aggregate_allele_counts(
        #     rna_bins,
        #     rna_bc_idxs,
        #     rna_cell_snps,
        #     rna_dp_mtx,
        #     rna_ad_mtx,
        #     "GEX",
        #     gex_dir,
        # )

        # aggregate_var_counts(rna_bins, rna_adata, "GEX", gex_dir)

        # rna_bins.to_csv(
        #     os.path.join(gex_dir, "bin_information.tsv"),
        #     columns=["BIN_ID", "#CHR", "START", "END", "HB", "CNP", "#VAR", "VAR_DP", "ALLELE_DP", "B_ALLELE_DP"],
        #     sep="\t",
        #     header=True,
        #     index=False,
        # )

        ##################################################
        # handle scATAC-seq data
        atac_dir = os.path.join(prep_dir, "ATAC")
        os.makedirs(atac_dir, exist_ok=True)

        atac_bc_idxs, atac_cell_snps, atac_dp_mtx, atac_ad_mtx = load_cellsnp_files(
            args["cellsnp_dir_2"], snp_info, barcodes
        )

        atac_fragment_file = os.path.join(ranger_dir, "atac_fragments.tsv.gz")
        atac_adata_file = os.path.join(out_dir, "atac_adata.h5ad")
        if not os.path.exists(atac_adata_file):
            atac_adata = build_atac_tile_matrix(
                atac_fragment_file,
                args["genome_file"],
                barcodes,
                atac_adata_file,
                tmp_dir,
                window_size=5e3,
            )
        else:
            atac_adata = sc.read_h5ad(atac_adata_file)

        print(atac_adata)
        # consolidate_snp_feature_ATAC(atac_fragment_file, atac_cell_snps,
        #                              haplo_blocks, tmp_dir, args["genome_file"], window_size=1e5)

        # TODO extract fragment counts directly.
        # atac_adata = adata[:, adata.var["feature_types"] == "Peaks"].copy()
        # print(f"#peaks={atac_adata.shape[1]}")
        # atac_bc_idxs, atac_cell_snps, atac_dp_mtx, atac_ad_mtx = load_cellsnp_files(
        #     args["cellsnp_dir_2"], snp_info, barcodes
        # )
        # TODO think about better way to do the consolidation
        # atac_adata, atac_cell_snps, atac_super_features = consolidate_snp_feature(atac_adata, atac_cell_snps, haplo_blocks, tmp_dir, "scATAC-seq")
        # print(atac_adata.var.head())
        # print(atac_cell_snps.head())

        pass
    elif modality == "visium":
        ##################################################
        adata = load_input_10x(ranger_dir, barcodes, annotation_file, modality=modality)

        # spatial coordinates TODO
        pass

    else:
        # TODO
        pass

    # shutil.rmtree(tmp_dir)
