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
# Load DNA bulk data
def load_input_HATCHet(
    seg_ucn: str,
    phased_vcf_file: str,
    hatchet_version: str,
    hatchet_files=[],
    trust_PS=False,
    min_allele_counts=500,
    out_dir=None,
    verbose=False,
):
    print("==================================================")
    print(f"load inputs from HATCHet files, {hatchet_version} version")
    print(f"trust_PS={trust_PS} min_allele_counts={min_allele_counts}")

    phased_snps = read_VCF(phased_vcf_file, phased=True)
    if hatchet_version == "old":
        [tumor_1bed_file] = hatchet_files
        snp_info = read_baf_file(tumor_1bed_file)
        snp_info = pd.merge(
            left=snp_info, right=phased_snps, on=["#CHR", "POS"], how="left"
        )
        snp_info.dropna(subset=["GT"], inplace=True)
        # assume one tumor sample for now
        ref_counts = snp_info["REF"].to_numpy().astype(np.int32)
        alt_counts = snp_info["ALT"].to_numpy().astype(np.int32)
    else:  # new
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

    assert len(ref_counts) == len(alt_counts)
    assert len(ref_counts) == len(snp_info)
    snp_info["POS0"] = snp_info["POS"] - 1

    segs, cnv_Aallele, cnv_Ballele, cnv_mixBAF, clone_props = format_cnv_profile(
        seg_ucn
    )

    snp_info = annotate_snps_seg_idx(segs, snp_info, "SEG_IDX")
    na_snps = snp_info["SEG_IDX"].isna().to_numpy()
    num_na_snps = np.sum(na_snps)
    print(
        f"#SNPs outside CNV segments: {num_na_snps}/{len(snp_info)}={num_na_snps / len(snp_info):.3%}"
    )
    ref_counts = ref_counts[~na_snps]
    alt_counts = alt_counts[~na_snps]
    snp_info.dropna(subset="SEG_IDX", inplace=True)
    snp_info["SEG_IDX"] = snp_info["SEG_IDX"].astype(segs.index.dtype)
    snp_info = snp_info.reset_index(drop=True)

    snp_info["PHASE"] = snp_info["GT"].astype(np.float32)
    snp_info["START"] = snp_info["POS0"]
    snp_info["END"] = snp_info["POS"]
    snp_info["DP"] = ref_counts + alt_counts

    ##################################################
    # divide CNV segments into haplotype blocks (HB), SNPs are fully phased in HB.
    hb_id = 0
    snp_info["HB"] = 0
    snp_info["BAF"] = 0.0

    # sub-PS index relative to current SEG only
    snp_info["PS1"] = snp_info["PS"]
    snp_info_segs = snp_info.groupby(by="SEG_IDX", sort=False)
    print(f"#cnv-segment={len(snp_info_segs)}")
    for seg_id, seg_row in segs.iterrows():  # per CNV segment
        seg_ch, seg_s, seg_t = seg_row["#CHR"], seg_row["START"], seg_row["END"]
        seg_cnp = seg_row["CNP"]
        imbalanced = seg_row["imbalanced"]
        seg_baf = cnv_mixBAF[seg_id]
        seg_snps = snp_info_segs.get_group(seg_id)
        seg_snps_index = seg_snps.index
        ps1_ids = seg_snps["PS1"].to_numpy()
        # form haplotype-blocks and save to PS1
        if not trust_PS:
            ps1_id = 0
            # subdivide SNPs into blocks
            for ps in seg_snps["PS"].unique():
                ps_snps = seg_snps.loc[seg_snps["PS"] == ps, :]
                ps1_id = adaptive_binning(
                    snp_info, ps_snps, "PS1", "DP", min_allele_counts, s_block_id=ps1_id
                )
            ps1_ids = snp_info.loc[seg_snps_index, "PS1"]

        uniq_ps1s = np.unique(ps1_ids)
        num_ps = len(seg_snps["PS"].unique())
        num_ps1 = len(uniq_ps1s)
        if verbose:
            print(
                f"{seg_cnp}\t{seg_ch}:{seg_s}-{seg_t}\t#SNPS={len(seg_snps)}\t#PS={num_ps}\t#PS1={num_ps1}\tBAF={seg_baf:.3f}"
            )
        if not imbalanced:
            # do not phase balanced segment, BAF won't help here
            for ps1_id in uniq_ps1s:
                ps_snps = seg_snps.loc[ps1_ids == ps1_id]
                snp_ps_index = ps_snps.index
                ps_refs = ref_counts[snp_ps_index]
                ps_alts = alt_counts[snp_ps_index]
                ps_phases = ps_snps["PHASE"].to_numpy()
                ps_balleles = ps_refs * ps_phases + ps_alts * (1 - ps_phases)
                ps_baf = np.sum(ps_balleles) / np.sum(ps_refs + ps_alts)
                snp_info.loc[snp_ps_index, "BAF"] = ps_baf
                snp_info.loc[snp_ps_index, "HB"] = hb_id
                hb_id += 1
        else:
            # phase all SNPs within AI CNA segment here
            for ps1_id in uniq_ps1s:
                ps_snps = seg_snps.loc[ps1_ids == ps1_id]
                snp_ps_index = ps_snps.index
                ps_refs = ref_counts[snp_ps_index]
                ps_alts = alt_counts[snp_ps_index]
                ps_phases = ps_snps["PHASE"].to_numpy()
                ps_balleles = ps_refs * ps_phases + ps_alts * (1 - ps_phases)
                ps_bcount = np.sum(ps_balleles)
                ps_tcount = np.sum(ps_refs + ps_alts)
                ps_baf = ps_bcount / ps_tcount
                # compare log-likelihood
                delta = (2 * ps_bcount - ps_tcount) * np.log(seg_baf / (1 - seg_baf))
                if delta < 0:
                    snp_info.loc[snp_ps_index, "PHASE"] = 1 - ps_phases
                    ps_baf = 1 - ps_baf
                snp_info.loc[snp_ps_index, "BAF"] = ps_baf
            snp_info.loc[seg_snps_index, "HB"] = hb_id
            hb_id += 1

    # save b-allele SNP counts for sanity check on phasing
    phases = snp_info["PHASE"].to_numpy()
    snp_bafs = (ref_counts * phases + alt_counts * (1 - phases)) / (
        ref_counts + alt_counts
    )
    snp_info["BULK_BAF"] = snp_bafs.round(3)

    # construct haplotype block file
    haplo_snps_blocks = snp_info.groupby(by="HB", sort=False, as_index=True)
    haplo_blocks = haplo_snps_blocks.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "COV": ("DP", "mean"),
            "BAF": ("BAF", "first"),
            "SEG_IDX": ("SEG_IDX", "first"),
        }
    )
    haplo_blocks.loc[:, "#SNPS"] = haplo_snps_blocks.size().reset_index(drop=True)
    haplo_blocks.loc[:, "BLOCKSIZE"] = haplo_blocks["END"] - haplo_blocks["START"]
    haplo_blocks["HB"] = haplo_blocks.index
    haplo_blocks["COV"] = haplo_blocks["COV"].round(3)
    haplo_blocks["BAF"] = haplo_blocks["BAF"].round(3)

    # append CNV information
    segs["SEG_IDX"] = segs.index
    haplo_blocks = pd.merge(
        left=haplo_blocks, right=segs[["CNP", "SEG_IDX"]], on=["SEG_IDX"], how="left"
    )

    haplo_blocks = haplo_blocks[
        ["HB", "#CHR", "START", "END", "BLOCKSIZE", "#SNPS", "COV", "BAF", "CNP"]
    ]
    snp_info = snp_info[["#CHR", "POS", "POS0", "PHASE", "HB", "BULK_BAF"]]

    if not out_dir is None:
        print(f"save result into {out_dir}")
        haplo_blocks.to_csv(
            os.path.join(out_dir, "haplotype_blocks.tsv"),
            header=True,
            sep="\t",
            index=False,
        )
        snp_info.to_csv(
            os.path.join(out_dir, "snp_information.tsv.gz"),
            header=True,
            sep="\t",
            index=False,
        )
    return haplo_blocks, snp_info


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
    dp_mtx: csr_matrix = mmread(dp_file).tocsr()
    ad_mtx: csr_matrix = mmread(ad_file).tocsr()

    cell_snps = read_VCF_cellsnp_err_header(vcf_file)
    cell_snps["RAW_SNP_IDX"] = np.arange(len(cell_snps))  # use to index matrix
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

    return barcode_indices, cell_snps, dp_mtx, ad_mtx


def load_calicost_prep_data(calicost_prep_dir: str):
    print(f"load allele count data from CalicoST preprocessed files")
    barcode_file = os.path.join(calicost_prep_dir, "barcodes.txt")
    a_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Aallele.npz")
    b_mtx_file = os.path.join(calicost_prep_dir, "cell_snp_Ballele.npz")
    usnp_id_file = os.path.join(calicost_prep_dir, "unique_snp_ids.npy")
    # TODO
    return


##################################################
# Load 10x data
def load_input_10x(
    data_path: str,
    barcodes: list,
    annotation_file: str,
    modality="",
):
    """
    read in 10x directory, filter cells, add cell type annotations (optional), and add feature coordinates.
    """
    if modality == "visium":
        adata = sq.read.visium(data_path, load_images=False)
    elif modality == "multiome":
        adata = sc.read_10x_mtx(
            os.path.join(data_path, "filtered_feature_bc_matrix"),
            var_names="gene_ids",
            gex_only=False,
        )

    # filter barcodes
    adata = adata[barcodes, :].copy()
    adata.var["gene_ids"] = adata.var.index

    # load gene annotation
    print("load feature annotation")
    ann = load_annotation_file_bed(annotation_file)
    ann.drop_duplicates(subset="gene_ids", inplace=True)
    ann["gene_ids"] = ann["gene_ids"].astype(object)
    ann = ann.set_index("gene_ids")
    adata.var = adata.var.merge(
        ann, left_on="gene_ids", right_index=True, how="left", sort=False
    )

    if modality == "multiome":
        # ATAC peaks are located directly in the file, parse it
        mask = adata.var["feature_types"] == "Peaks"
        peaks = adata.var.index[mask]
        adata.var.loc[mask, "#CHR"] = peaks.str.split(":").str[0]
        adata.var.loc[mask, "START"] = (
            peaks.str.split(":").str[1].str.split("-").str[0].astype(int)
        )
        adata.var.loc[mask, "END"] = (
            peaks.str.split(":").str[1].str.split("-").str[1].astype(int)
        )
    print(f"#features={adata.shape[1]}")

    located_features = adata.var.index[adata.var["START"].notna()]
    adata = adata[:, located_features].copy()
    print(f"#features (ignore na positions)={adata.shape[1]}")

    adata.var["#CHR"] = adata.var["#CHR"].astype(str)
    adata.var["START"] = adata.var["START"].astype(int)
    adata.var["END"] = adata.var["END"].astype(int)

    # adata.var["chromosome"] = adata.var["#CHR"]
    # adata.var["start"] = adata.var["START"]
    # adata.var["end"] = adata.var["END"]

    adata.var["pseudobulk_counts"] = np.asarray(adata.X.sum(axis=0)).flatten()
    return adata


##################################################
# Load scATAC-seq fragments data
def load_atac_fragments(atac_frag_file: str):
    atac_fragments = pd.read_table(atac_frag_file, sep="\t", comment="#")
    atac_fragments = atac_fragments.rename(
        columns={
            "chrom": "#CHR",
            "chromStart": "START",
            "chromEnd": "END",
            "barcode": "BARCODE",
            "readSupport": "#READS",
            "strand": "STRAND",
        }
    )
    print(atac_fragments.head())
    return atac_fragments


def build_atac_tile_matrix(
    fragment_file: str, chrom_file: str, barcodes: list, out_file: str, tmp_dir: str, window_size=500
):
    print("build ATAC fragment count matrix by SnapATAC2")
    chrom_sizes = get_chr_sizes(chrom_file)

    print("load fragments")
    adata: AnnData = snap.pp.import_fragments(
        fragment_file,
        chrom_sizes,
        whitelist=barcodes,
        sorted_by_barcode=False,
        tempdir=tmp_dir
    )

    print("build tile matrix")
    snap.pp.add_tile_matrix(
        adata,
        bin_size=window_size,
        exclude_chroms=["chrM", "chrX", "chrY", "M", "X", "Y"],
    )
    adata.write_h5ad(out_file, compression="gzip")
    return adata


##################################################
def consolidate_snp_feature_ATAC(
    atac_frag_file: str,
    snp_df: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    tmp_dir: str,
    genome_file: str,
    window_size=1e5,
):
    print(f"consolidate SNPs with features for ATACseq")

    ##################################################
    # compute per-window ATAC coverage
    tmp_cov_file = os.path.join(tmp_dir, "tmp_ATAC_cov.out.bed")
    if not os.path.exists(tmp_cov_file):
        # 1. compute CNA windows over haplotype blocks
        tmp_block_in_file = os.path.join(tmp_dir, "tmp_block.in.bed")
        tmp_window_file = os.path.join(tmp_dir, "tmp_windows.out.bed")
        haplo_blocks.to_csv(
            tmp_block_in_file,
            sep="\t",
            header=False,
            index=False,
            columns=["#CHR", "START", "END", "HB"],
        )

        run_bedtools_makewindows(tmp_block_in_file, tmp_window_file, window_size)

        # add sex chroms to avoid bedtools error, no HB tag
        with open(tmp_window_file, "a") as fd:
            fd.write("chrX\t0\t1\tNA\n")
            fd.write("chrY\t0\t1\tNA\n")
            fd.close()

        run_bedtools_coverage(
            atac_frag_file,
            tmp_window_file,
            genome_file,
            tmp_cov_file,
            tmp_dir,
            load_df=False,
        )
    atac_cov_df = pd.read_table(
        tmp_cov_file,
        header=None,
        names=["#CHR", "START", "END", "HB_WIN", "count"],
        dtype={"HB_WIN": "string", "count": int},
    )

    var_super_df = atac_cov_df.loc[atac_cov_df["HB_WIN"] != "NA", :].reset_index(
        drop=True
    )
    var_super_df["HB"] = (
        var_super_df["HB_WIN"].apply(func=lambda v: v.split("_")[0]).astype(int)
    )
    var_super_df["SUPER_VAR_IDX"] = var_super_df.index
    var_super_df = var_super_df.drop(columns=["HB_WIN"])

    ##################################################
    snp_df = assign_snp_to_feature(var_super_df, snp_df, "ATAC")
    return snp_df, var_super_df


def assign_snp_to_feature(
    var_super_df: pd.DataFrame,
    snp_df: pd.DataFrame,
    feature_name: str,
):
    # assign SUPER_VAR_IDX tag to snp_df
    print(f"#{feature_name}-SNP={len(snp_df)}")
    snp_df = assign_pos_to_range(
        snp_df, var_super_df, ref_id="SUPER_VAR_IDX", pos_col="POS0"
    )
    isna_snp_df = snp_df["SUPER_VAR_IDX"].isna()
    # this is mostly due to XXX
    print(
        f"#{feature_name}-SNPs outside any super-feature={np.sum(isna_snp_df) / len(snp_df):.3%}"
    )
    snp_df = snp_df.loc[snp_df["SUPER_VAR_IDX"].notna(), :].copy(deep=True)
    return snp_df


def consolidate_snp_feature_RNA(
    adata: AnnData,
    snp_df: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    tmp_dir: str,
    feature_name: str,
):
    print(f"consolidate SNPs with features for {feature_name}")

    ##################################################
    # union all overlapping features, build var_super_df
    print("detect overlapping features")
    tmp_feature_in_file = os.path.join(tmp_dir, f"tmp_{feature_name}.in.bed")
    tmp_feature_out_file = os.path.join(tmp_dir, f"tmp_{feature_name}.out.bed")
    adata.var.to_csv(
        tmp_feature_in_file,
        sep="\t",
        header=False,
        index=False,
        columns=["#CHR", "START", "END", "gene_ids"],
    )

    var_df_clustered = run_bedtools_cluster(
        tmp_feature_in_file,
        tmp_feature_out_file,
        tmp_dir,
        max_dist=0,
        load_df=True,
        usecols=list(range(5)),
        names=["#CHR", "START", "END", "gene_ids", "SUPER_VAR_IDX"],
    )

    new_var = pd.merge(
        left=adata.var,
        right=var_df_clustered,
        on=["#CHR", "START", "END", "gene_ids"],
        how="left",
    )
    new_var.index = adata.var.index.astype(str)
    adata.var = new_var

    var_supers = adata.var.groupby(by="SUPER_VAR_IDX", sort=False, as_index=True)
    var_super_df = var_supers.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
        }
    ).reset_index(drop=False)
    print(f"#{feature_name}-super-feature={len(var_super_df)}")

    ##################################################
    # assign HB tag to var_super_df
    var_super_df = assign_largest_overlap(
        var_super_df, haplo_blocks, "SUPER_VAR_IDX", "HB"
    )
    isna_super_var = var_super_df["HB"].isna()
    # this is mostly due to centromere/satellite-repeat mask.
    print(
        f"#{feature_name}-super-feature outside any haplotype blocks={np.sum(isna_super_var) / len(var_super_df):.3%}"
    )

    var_super_df.dropna(subset="HB", inplace=True)
    var_super_df["HB"] = var_super_df["HB"].astype(np.int32)

    # map HB tag to adata.var, filter any features accordingly
    new_var = pd.merge(
        left=adata.var,
        right=var_super_df[["SUPER_VAR_IDX", "HB"]],
        on="SUPER_VAR_IDX",
        how="left",
    )
    new_var.index = adata.var.index.astype(str)
    adata.var = new_var

    adata = adata[:, adata.var["HB"].notna()].copy()
    adata.var["HB"] = adata.var["HB"].astype(np.int32)

    ##################################################
    snp_df = assign_snp_to_feature(var_super_df, snp_df, feature_name)
    return adata, snp_df, var_super_df


##################################################
# binning features and SNPs
def stat_pseudobulk(
    adata: AnnData,
    cell_snps: pd.DataFrame,
    var_super_df: pd.DataFrame,
):
    print("collect pseudobulk super-feature allele counts and total counts")
    if not cell_snps is None:
        pseudobulk_allele_counts = (
            cell_snps.groupby("SUPER_VAR_IDX", sort=False)["DP"]
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
    return var_super_df


def co_binning_allele_feature(
    adata: AnnData,
    cell_snps: pd.DataFrame,
    var_super_df: pd.DataFrame,
    haplo_blocks: pd.DataFrame,
    min_allele_counts: int,
    min_total_counts: int,
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

    # append copy-number profile to bins
    var_bins = pd.merge(
        left=var_bins, right=haplo_blocks[["HB", "CNP"]], on=["HB"], how="left"
    )
    assert var_bins["CNP"].notna().all(), "corrupted data"

    adata.var = pd.merge(
        left=adata.var,
        right=var_super_df[["SUPER_VAR_IDX", "BIN_ID"]],
        on="SUPER_VAR_IDX",
        how="left",
    )
    cell_snps = pd.merge(
        left=cell_snps,
        right=var_super_df[["SUPER_VAR_IDX", "BIN_ID"]],
        on="SUPER_VAR_IDX",
        how="left",
    )
    return adata, cell_snps, var_bins


##################################################
def aggregate_allele_counts(
    var_bins: pd.DataFrame,
    barcode_indicies: np.ndarray,
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

    # subset barcodes
    a_allele_mat = a_allele_mat[:, barcode_indicies]
    b_allele_mat = b_allele_mat[:, barcode_indicies]
    t_allele_mat = t_allele_mat[:, barcode_indicies]
    snp_count_mat = snp_count_mat[:, barcode_indicies]

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
    num_bins = len(var_bins)
    num_barcodes = adata.n_obs

    adata.var["VAR_IDX"] = np.arange(adata.X.shape[1])

    bin_count_mat = np.zeros((num_bins, num_barcodes), dtype=np.int32)
    bin_ids = adata.var["BIN_ID"].unique()
    cell_var_bins = adata.var.groupby(by="BIN_ID", sort=False, as_index=True)
    for bin_id in bin_ids:
        var_bin = cell_var_bins.get_group(bin_id)
        var_indices = var_bin["VAR_IDX"].to_numpy()
        bin_count_mat[bin_id, :] = np.sum(adata.X[:, var_indices], axis=1).ravel()

    bin_count_file = os.path.join(out_dir, f"count.npz")
    sparse.save_npz(bin_count_file, csr_matrix(bin_count_mat))
    return
