import numpy as np
import pandas as pd
from utils import *

def construct_phase_blocks(
    segs: pd.DataFrame,
    cnv_mixBAF: np.ndarray,
    snp_info: pd.DataFrame,
    ref_counts: np.ndarray,
    alt_counts: np.ndarray,
    phase_method="iid-map",
    trust_PS=False,
    min_allele_counts=500,
    verbose=1
):
    """divide CNV segments into haplotype blocks (HB), SNPs are fully phased in HB.
    Assume all SNPs are pre-phased.

    Results: 1. consistently phased SNPs have same HB id and updated PHASE
    Args:
        segs (pd.DataFrame): copy-number profile
        snp_info (pd.DataFrame): snp information, positions, phase, phaseset, SEG_IDX
        ref_counts (np.ndarray): reference counts
        alt_counts (np.ndarray): non-reference counts
        guide_bafs (_type_, optional): guiding B-allele frequency from copy-number profile. Defaults to None.
        trust_PS (bool, optional): trust phaseset. Defaults to False.
        min_allele_counts (int, optional): pre-binning if trust_PS=False. Defaults to 500.
    """
    # haplotype block id
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
        ##################################################
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
        
        ##################################################
        if not imbalanced:
            # do not phase balanced segment, BAF won't help here
            # this block won't be used for copytyping
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

    ##################################################
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
    return haplo_blocks, snp_info
