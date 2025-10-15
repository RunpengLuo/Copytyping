import os
import sys
import shutil

import numpy as np
import pandas as pd

import scanpy as sc

from parsing import parse_arguments_preprocess
from utils import *
from io_utils import *
from preprocess_utils import *
from snp_phaser import construct_phase_blocks


def wrapper_single_data_type(
    h5ad_file: str,
    haplo_blocks: pd.DataFrame,
    allele_infos: str,
    data_type: str,
    min_allele_counts: int,
    min_total_counts: int,
    min_total_thres: int,
    min_allele_thres: int,
    out_dir: str,
    tmp_dir: str,
):
    adata: AnnData = sc.read_h5ad(h5ad_file)
    print(adata)
    print(adata.var.head(3))

    cell_snps, dp_mat, ref_mat, alt_mat = allele_infos[data_type]
    adata, cell_snps, super_features = consolidate_snp_feature(
        adata,
        cell_snps,
        haplo_blocks,
        tmp_dir,
        data_type,
        feature_may_overlap=True,
    )

    adata, cell_snps, bins = co_binning_allele_feature(
        adata,
        cell_snps,
        super_features,
        haplo_blocks,
        min_allele_counts,
        min_total_counts,
        min_total_thres,
        min_allele_thres,
    )

    aggregate_allele_counts(
        bins,
        cell_snps,
        dp_mat,
        alt_mat,
        data_type,
        out_dir,
    )

    aggregate_var_counts(bins, adata, data_type, out_dir)

    bins.to_csv(
        os.path.join(out_dir, "bin_information.tsv"),
        columns=[
            "BIN_ID",
            "#CHR",
            "START",
            "END",
            "HB",
            "CNP",
            "#VAR",
            "VAR_DP",
            "ALLELE_DP",
            "B_ALLELE_DP",
        ],
        sep="\t",
        header=True,
        index=False,
    )
    return


if __name__ == "__main__":
    args = parse_arguments_preprocess()

    phase_method = "iid-map"
    phase_mode = "bulk"

    min_allele_counts_ps = 5000
    trust_PS = True

    min_allele_counts_rna = 1500
    min_total_counts_rna = 3000
    min_total_thres_rna = 200
    min_allele_thres_rna = 10

    min_allele_counts_atac = 2500
    min_total_counts_atac = 5000
    min_total_thres_atac = 200
    min_allele_thres_atac = 20

    min_allele_counts_visium = 5000
    min_total_counts_visium = 5000
    min_total_thres_visium = 500
    min_allele_thres_visium = 500

    modality = args["modality"]
    sample = args["sample"]
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    prep_dir = os.path.join(out_dir, "preprocess")
    os.makedirs(prep_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    ##################################################
    # load HATCHet copy-number profile
    segs, cnv_Aallele, cnv_Ballele, cnv_mixBAF, clone_props = format_cnv_profile(
        args["seg_ucn"]
    )

    ##################################################
    # load barcode file
    barcode_file = args["barcodes"]
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
    if modality == "visium":
        barcodes_df["BARCODE_RAW"] = barcodes_df["BARCODE"]
        if "_" in barcodes[0]:
            barcodes_df[["BARCODE", "SLICE"]] = barcodes_df["BARCODE_RAW"].str.split(
                "_", n=1, expand=True
            )
        else:
            barcodes_df["SLICE"] = "U1"
        num_slices = len(barcodes_df["SLICE"].unique())
        print(f"#slices={num_slices}")

    barcodes_df.to_csv(
        os.path.join(prep_dir, "Barcodes.tsv"), sep="\t", index=False, header=True
    )

    ##################################################
    # load allele informations per data type
    allele_infos = {}
    data_types = []
    if modality == "multiome":
        data_types = ["GEX", "ATAC"]
        # scRNA-seq, load from cellsnp-lite
        # rna_cell_snps, rna_dp_mat, rna_ref_mat, rna_alt_mat
        allele_infos["GEX"] = load_cellsnp_files(args["cellsnp_dir_1"], None, barcodes)
        # scATAC-seq, load from cellsnp-lite
        # atac_cell_snps, atac_dp_mat, atac_ref_mat, atac_alt_mat
        allele_infos["ATAC"] = load_cellsnp_files(args["cellsnp_dir_2"], None, barcodes)
    elif modality == "visium":
        allele_infos["visium"] = load_calicost_prep_data(
            args["calicoST_prep_dir"], barcodes
        )
    else:
        raise ValueError()

    ##################################################
    # phase SNPs and build haplotype blocks. shared across modalities
    # SNPs can be thrown in this step
    haplo_blocks = None
    snp_info = None
    haplo_block_file = os.path.join(prep_dir, "haplotype_blocks.tsv")
    snp_info_file = os.path.join(prep_dir, "snp_information.tsv.gz")
    if os.path.exists(haplo_block_file) and os.path.exists(snp_info_file):
        print("load existing prep-ed haplotype block and snp files")
        haplo_blocks = pd.read_table(haplo_block_file, sep="\t", index_col=None)
        snp_info = pd.read_table(snp_info_file, sep="\t", index_col=None)
    else:
        if phase_mode == "bulk":
            print("phase using bulk information")
            phased_snps = read_VCF(args["vcf_file"], phased=True)
            if args["allele_dir"]:
                snp_info, allele_counts, ref_counts, alt_counts = load_snps_HATCHet_new(
                    phased_snps, [args["allele_dir"]]
                )
            else:
                assert not args["tumor_1bed"] is None
                snp_info, allele_counts, ref_counts, alt_counts = load_snps_HATCHet_old(
                    phased_snps, [args["tumor_1bed"]]
                )
        else:
            print("phase using pseudobulk information")
            snp_info, allele_counts, ref_counts, alt_counts = load_snps_pseudobulk(
                allele_infos, modality
            )

        snp_info, allele_counts, ref_counts, alt_counts = annotate_snps(
            segs, snp_info, allele_counts, ref_counts, alt_counts
        )
        haplo_blocks, snp_info = construct_phase_blocks(
            segs,
            cnv_mixBAF,
            snp_info,
            ref_counts,
            alt_counts,
            phase_method=phase_method,
            trust_PS=trust_PS,
            min_allele_counts=min_allele_counts_ps,
            verbose=True,
        )
        print(f"save phased blocks")
        haplo_blocks.to_csv(
            os.path.join(prep_dir, "haplotype_blocks.tsv"),
            header=True,
            sep="\t",
            index=False,
        )
        snp_info.to_csv(
            os.path.join(prep_dir, "snp_information.tsv.gz"),
            header=True,
            sep="\t",
            index=False,
        )

    allele_infos = annotate_snps_post(snp_info, allele_infos)

    # TODO we can access phasing accuracy here?

    ##################################################
    # preprocess 10x inputs, binning independently per modality
    if modality == "multiome":
        print("process gene expression data")
        gex_dir = os.path.join(prep_dir, "GEX")
        os.makedirs(gex_dir, exist_ok=True)
        wrapper_single_data_type(
            args["rna_h5ad"],
            haplo_blocks,
            allele_infos,
            "GEX",
            min_allele_counts_rna,
            min_total_counts_rna,
            min_total_thres_rna,
            min_allele_thres_rna,
            gex_dir,
            tmp_dir,
        )

        print("process chromatin accessibility data")
        atac_dir = os.path.join(prep_dir, "ATAC")
        os.makedirs(atac_dir, exist_ok=True)
        wrapper_single_data_type(
            args["atac_h5ad"],
            haplo_blocks,
            allele_infos,
            "ATAC",
            min_allele_counts_atac,
            min_total_counts_atac,
            min_total_thres_atac,
            min_allele_thres_atac,
            atac_dir,
            tmp_dir,
        )
    elif modality == "visium":
        print("process spatial spot-level gene expression data")
        gex_dir = os.path.join(prep_dir, "visium")
        os.makedirs(gex_dir, exist_ok=True)
        wrapper_single_data_type(
            args["rna_h5ad"],
            haplo_blocks,
            allele_infos,
            "visium",
            min_allele_counts_visium,
            min_total_counts_visium,
            min_total_thres_visium,
            min_allele_thres_visium,
            gex_dir,
            tmp_dir,
        )
    else:
        # TODO
        pass

    # shutil.rmtree(tmp_dir)
