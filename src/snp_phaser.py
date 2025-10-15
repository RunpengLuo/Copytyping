import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln, logsumexp
from scipy.optimize import minimize_scalar
from utils import *
from io_utils import load_genetic_map

def construct_phase_blocks(
    segs: pd.DataFrame,
    cnv_mixBAF: np.ndarray,
    snp_info: pd.DataFrame,
    ref_counts: np.ndarray,
    alt_counts: np.ndarray,
    genetic_map_file=None,
    trust_PS=False,
    model_overdispersion=True,
    soft_phasing=False,
    min_allele_counts=500,
    verbose=1
):
    """
    1. locate SNPs in CNV segments
    2. learn dispersion parameter from netural bins (if use beta-binomial)
    2. per CNV segment
        1. if PS information is unreliable, further break into PS1
        2. SNPs within PS1 are phased
        3. fix BAF from bulk CNP, which defines the b-allele.
        4. use fixed BAF and transitions to do either EM E-step or HMM forward-backward -> soft phasing
        5. if hard-phasing, use MAP estimate.
    Results: consistently phased SNPs having same HB id and updated PHASE
    Args:
        segs (pd.DataFrame): copy-number profile
        snp_info (pd.DataFrame): snp information, positions, phase, phaseset, SEG_IDX
        ref_counts (np.ndarray): reference counts
        alt_counts (np.ndarray): non-reference counts
        guide_bafs (_type_, optional): guiding B-allele frequency from copy-number profile. Defaults to None.
        trust_PS (bool, optional): trust phaseset. Defaults to False.
        min_allele_counts (int, optional): PS1 pre-binning if trust_PS=False. Defaults to 500.
    """
    
    # haplotype block id
    hb_id = 0
    snp_info["HB"] = 0
    snp_info["BAF"] = 0.0

    ##################################################
    # form relative local haplotype blocks PS1
    # avoid degenerated PS
    snp_info["PS1"] = snp_info["PS"]
    snp_info_segs = snp_info.groupby(by="SEG_IDX", sort=False)
    print(f"#cnv-segment={len(snp_info_segs)}")
    snp_info["PS1"] = snp_info["PS"]
    if not trust_PS:
        ps1_id = 0
        for seg_id, seg_row in segs.iterrows():  # per CNV segment
            seg_snps = snp_info_segs.get_group(seg_id)            
            # form minimum haplotype-blocks and save to PS1
            for ps in seg_snps["PS"].unique():
                ps_snps = seg_snps.loc[seg_snps["PS"] == ps, :]
                ps1_id = adaptive_binning(
                    snp_info, ps_snps, "PS1", "DP", min_allele_counts, s_block_id=ps1_id
                )
    snp_ps1s = snp_info.groupby(by="PS1", sort=False, as_index=True)
    ps1_info = snp_ps1s.agg(
        **{
            "SEG_IDX": ("SEG_IDX", "first"),
            "imbalanced": ("imbalanced", "first"),
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "B_ALLELE_RAW": ("B_ALLELE_RAW", "sum"),
            "DP": ("DP", "sum"),
        }
    )
    ps1_info["PS1"] = ps1_info.index.to_numpy()
    ps1_info["POS"] = (ps1_info["START"] + ps1_info["END"]) // 2
    ps1_info["PHASE_RAW"] = 1.0
    ps1_info["A_ALLELE_RAW"] = ps1_info["DP"] - ps1_info["B_ALLELE_RAW"]
    ##################################################
    # estimate over-dispersion parameters from netural bins
    tau = None
    if model_overdispersion:
        netural_a_counts = ps1_info.loc[~ps1_info["imbalanced"], "A_ALLELE_RAW"].to_numpy()
        netural_b_counts = ps1_info.loc[~ps1_info["imbalanced"], "B_ALLELE_RAW"].to_numpy()
        if len(netural_a_counts) > 50:
            tau = estimate_overdispersion(
                netural_a_counts,
                netural_b_counts,
                init_taus=None
            )
    ##################################################
    # load switch probs
    if not genetic_map_file is None:
        ps1_info = estimate_switchprob_genetic_map(ps1_info, genetic_map_file, nu=1)
        phase_method = "hmm"
    else:
        phase_method = "em"

    ##################################################
    print(f"phase PS1-blocked SNPs with {phase_method}, tau={tau}, soft-phasing={soft_phasing}")
    # close to 1 if reference allele is consistent with CNP
    snp_info["PHASE"] = 1.0
    snp_info_segs = snp_info.groupby(by="SEG_IDX", sort=False)

    # close to 1 if b-allale is consistent with CNP
    ps1_info["PHASE"] = 1.0
    ps1_info_segs = ps1_info.groupby(by="SEG_IDX", sort=False)
    for seg_id, seg_row in segs.iterrows():  # per CNV segment
        seg_ch, seg_s, seg_t = seg_row["#CHR"], seg_row["START"], seg_row["END"]
        seg_cnp = seg_row["CNP"]
        imbalanced = seg_row["imbalanced"]
        seg_baf = cnv_mixBAF[seg_id]

        if seg_id not in ps1_info_segs.groups:
            print(
                f"{seg_cnp}\t{seg_ch}:{seg_s}-{seg_t}\tno PS1\tBAF={seg_baf:.3f}"
            )
            continue

        seg_ps1s = ps1_info_segs.get_group(seg_id)
        seg_ps1s_index = seg_ps1s.index
        if verbose:
            print(
                f"{seg_cnp}\t{seg_ch}:{seg_s}-{seg_t}\t#PS1={len(seg_ps1s)}\tBAF={seg_baf:.3f}"
            )

        seg_snps = snp_info_segs.get_group(seg_id)
        seg_snps_index = seg_snps.index
        if not imbalanced:
            # do not phase balanced segment, BAF won't help here
            # this block won't be used for copytyping
            snp_info.loc[seg_snps_index, "PHASE"] = seg_snps["PHASE_RAW"]
            for ps1_id in seg_ps1s["PS1"].to_numpy():
                ps_snps = seg_snps.loc[seg_snps["PS1"] == ps1_id]
                snp_ps_index = ps_snps.index
                snp_info.loc[snp_ps_index, "HB"] = hb_id
                hb_id += 1
            continue

        # phase all SNPs within imbalanced CNA segment here
        a_counts = seg_ps1s["A_ALLELE_RAW"].to_numpy()
        b_counts = seg_ps1s["B_ALLELE_RAW"].to_numpy()
        c_counts = seg_ps1s["DP"].to_numpy()
        if phase_method == "em":
            phase_posts = em_estep(a_counts, b_counts, c_counts, seg_baf, tau)
        else:
            switchprobs = seg_ps1s["switchprobs"].to_numpy()
            phase_posts = hmm_forward_backward(a_counts, b_counts, c_counts, switchprobs, seg_baf, tau)

        if not soft_phasing: # assign MAP-estimate
            phase_posts = np.round(phase_posts).astype(np.int8)
        ps1_info.loc[seg_ps1s_index, "PHASE"] = phase_posts
        ps1_info.loc[seg_ps1s_index, "HB"] = hb_id

        # map back to SNP-level phase
        raw_phase = seg_snps["PHASE_RAW"].to_numpy()
        correction = ps1_info.loc[seg_snps["PS1"].to_numpy(), "PHASE"].to_numpy()
        corr_phase = raw_phase * correction + (1 - raw_phase) * (1 - correction)
        snp_info.loc[seg_snps_index, "PHASE"] = corr_phase
        snp_info.loc[seg_snps_index, "HB"] = hb_id
        hb_id += 1

    ##################################################
    # save b-allele SNP counts for sanity check on phasing
    phases = snp_info["PHASE"].to_numpy()
    snp_info["B_ALLELE"] = (ref_counts * phases + alt_counts * (1 - phases)).round(3)
    snp_info["BAF_CORR"] = (snp_info["B_ALLELE"] / snp_info["DP"]).round(3)

    # construct haplotype block file
    haplo_snps_blocks = snp_info.groupby(by="HB", sort=False, as_index=True)
    haplo_blocks = haplo_snps_blocks.agg(
        **{
            "#CHR": ("#CHR", "first"),
            "START": ("START", "min"),
            "END": ("END", "max"),
            "COV": ("DP", "mean"),
            "DP": ("DP", "sum"),
            "B_ALLELE": ("B_ALLELE", "sum"),
            "SEG_IDX": ("SEG_IDX", "first"),
        }
    )
    haplo_blocks.loc[:, "#SNPS"] = haplo_snps_blocks.size().reset_index(drop=True)
    haplo_blocks.loc[:, "BLOCKSIZE"] = haplo_blocks["END"] - haplo_blocks["START"]
    haplo_blocks["HB"] = haplo_blocks.index
    haplo_blocks["COV"] = haplo_blocks["COV"].round(3)
    haplo_blocks["B_ALLELE"] = haplo_blocks["B_ALLELE"].round(3)
    haplo_blocks["BAF"] = (haplo_blocks["B_ALLELE"] / haplo_blocks["DP"]).round(3)

    # append CNV information
    segs["SEG_IDX"] = segs.index
    haplo_blocks = pd.merge(
        left=haplo_blocks, right=segs[["CNP", "SEG_IDX"]], on=["SEG_IDX"], how="left"
    )

    haplo_blocks = haplo_blocks[
        ["HB", "#CHR", "START", "END", "BLOCKSIZE", "#SNPS", "COV", "BAF", "CNP"]
    ]
    snp_info = snp_info[["#CHR", "POS", "POS0", "PHASE", "HB", "BAF_RAW", "BAF_CORR", "B_ALLELE", "DP"]]
    return haplo_blocks, snp_info

##################################################
def estimate_overdispersion(
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    init_taus=None,
    p=0.5
):
    """learn over-dispersion parameter from netrual bins
    """
    def neg_loglik_logw(logw):
        w = np.exp(logw[0])
        a0 = w*p
        b0 = w*(1-p)
        a1 = a_counts + a0
        b1 = b_counts + b0
        ll = np.sum(betaln(a1, b1) - betaln(a0, b0))
        return -ll
    
    if init_taus is None:
        init_taus = np.arange(10, 100, 10)
    max_tau = int(init_taus[-1] * 2)

    best = (0.0, np.inf)
    for tau0 in init_taus:
        res = minimize_scalar(
            neg_loglik_logw,
            x0=np.log([tau0]),
            method="L-BFGS-B",
            bounds=[(np.log(1), np.log(max_tau))],
            # options={"ftol":1e-6}
        )
        if res.fun < best[1]:
            best = (np.exp(res.x[0]), res.fun)
    if np.abs(best[0] - max_tau) <= 1e-6:
        return None # fall-back to binomial instead
    return best[0]

##################################################
def estimate_switchprob_genetic_map(
    snp_info: pd.DataFrame,
    genetic_map_file: str,
    nu=1,
    min_switchprob=1e-6
):
    """
    compute switchprobs for adjacent SNPs or bins.
    For bins, the bin boundary is used.
    """
    print("compute prior phase-switch probability from genetic map")
    genetic_map = load_genetic_map(genetic_map_file, mode="eagle2")    
    snp_info["d_morgans"] = 0.0
    genetic_map_chrs = genetic_map.groupby(by="#CHR", sort=False)
    for ch, ch_snps in snp_info.groupby(by="#CHR", sort=False):
        ch_maps = genetic_map_chrs.get_group(ch)
        start_cms = np.interp(
            ch_snps["START"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )
        pos_cms = np.interp(
            ch_snps["POS"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )
        end_cms = np.interp(
            ch_snps["END"].to_numpy(),
            ch_maps["POS"].to_numpy(),
            ch_maps["pos_cm"].to_numpy()
        )

        d_morgans_midpoint = np.zeros(len(pos_cms), dtype=np.float32)
        d_morgans_midpoint[:-1] = pos_cms[1:] - pos_cms[:-1]

        d_morgans = np.zeros(len(pos_cms), dtype=np.float32)
        d_morgans = start_cms[1:] - end_cms[:-1]

        # avoid direct overlapping bins
        d_morgans[d_morgans <= 0] = d_morgans_midpoint[d_morgans <= 0]
        d_morgans = np.maximum(d_morgans, 0.0) # avoid non-monotonic noise if any
        snp_info.loc[ch_snps.index, "d_morgans"] = d_morgans / 100

    
    snp_info["switchprobs"] = 0.0 # P(0->1 or 1->0), lower value favors phase switch
    for ch, ch_snps in snp_info.groupby(by="#CHR", sort=False):
        d_morgans = ch_snps["d_morgans"].to_numpy()
        switchprobs = (1 - np.exp(-2 * nu * d_morgans)) / 2.0
        switchprobs = np.clip(switchprobs, a_min=min_switchprob, a_max=0.5)
        snp_info.loc[ch_snps.index[1:], "switchprobs"] = switchprobs
    return snp_info

##################################################
# inference, return conditional marginal posterior for phase
def hmm_forward_backward(
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    c_counts: np.ndarray,
    switchprobs: np.ndarray,
    p: float,
    tau=None,
):
    nobs = len(a_counts)
    log_emissions = compute_lls(a_counts, b_counts, c_counts, p, tau)
    log_switch = np.log(switchprobs)
    log_stay = np.log(1.0 - switchprobs)

    log_alpha = np.zeros((nobs, 2), dtype=np.float64)  # alpha(z_n)
    log_alpha[0] = log_emissions[0] - np.log(2.0)
    for obs in range(1, nobs):
        log_alpha[obs, 0] = log_emissions[obs, 0] + logsumexp(
            [log_alpha[obs - 1, 0] + log_stay, 
             log_alpha[obs - 1, 1] + log_switch]
        )
        log_alpha[obs, 1] = log_emissions[obs, 1] + logsumexp(
            [log_alpha[obs - 1, 1] + log_stay, 
             log_alpha[obs - 1, 0] + log_switch]
        )

    log_beta = np.zeros((nobs, 2), dtype=np.float64)  # beta(z_n)
    log_beta[-1] = 0.0
    for obs in reversed(range(nobs - 1)):
        log_beta[obs, 0] = logsumexp([
            log_beta[obs + 1, 0] + log_emissions[obs + 1, 0] + log_stay[obs + 1],
            log_beta[obs + 1, 1] + log_emissions[obs + 1, 1] + log_switch[obs + 1],
        ])
        log_beta[obs, 1] = logsumexp([
            log_beta[obs + 1, 1] + log_emissions[obs + 1, 1] + log_stay[obs + 1],
            log_beta[obs + 1, 0] + log_emissions[obs + 1, 0] + log_switch[obs + 1],
        ])

    log_gamma = log_alpha + log_beta
    phase_posts = np.exp(log_gamma - logsumexp(log_gamma, axis=1, keepdims=True) , dtype=np.float32)
    return phase_posts[:, 0]

def em_estep(
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    c_counts: np.ndarray,
    p: float,
    tau=None,
):
    lls = compute_lls(a_counts, b_counts, c_counts, p, tau)
    phase_posts = np.exp(lls - logsumexp(lls, axis=1, keepdims=True), dtype=np.float32)
    return phase_posts[:, 0]

def compute_lls(
    a_counts: np.ndarray,
    b_counts: np.ndarray,
    c_counts: np.ndarray,
    p: float,
    tau=None,
):
    p = max(p, 1e-6) # avoid log(0) 
    # lls[:, 0] = logp(X,Y|D,Z=1;baf,tau)
    lls = np.zeros((len(a_counts), 2), dtype=np.float64)
    if not tau is None:
        alpha = tau * p
        beta = tau * (1.0 - p)
        log_binom = gammaln(c_counts + 1) - gammaln(b_counts + 1) - gammaln(a_counts + 1)
        lls[:, 0] = log_binom + betaln(b_counts + alpha, a_counts + beta) - betaln(alpha, beta)
        lls[:, 1] = log_binom + betaln(a_counts + alpha, b_counts + beta) - betaln(alpha, beta)
    else:
        # binom
        logp = np.log(p)
        logp_ = np.log(1 - p)
        log_binom = gammaln(c_counts + 1) - gammaln(b_counts + 1) - gammaln(a_counts + 1)
        lls[:, 0] = log_binom + b_counts * logp + a_counts * logp_
        lls[:, 1] = log_binom + a_counts * logp + b_counts * logp_
    return lls # (n, 2)
