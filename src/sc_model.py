import os
import sys

import numpy as np
import pandas as pd

from utils import *
from sx_data import *
from model_utils import *

import matplotlib.pyplot as plt
import seaborn as sns


global_barcodes = None
##################################################
class SC_Model:
    """Single-cell EM model, no spatial information, tumor purity=1 for each cell."""

    def __init__(
        self, barcodes: pd.DataFrame, prep_dir: str, out_dir: str, modality: str, verbose=1
    ) -> None:
        self.barcodes = barcodes
        self.N = self.num_barcodes = len(barcodes)
        self.out_dir = out_dir
        if modality == "multiome":
            self.data_types = ["GEX", "ATAC"]
        else:
            self.data_types = [modality]
        self.data_sources = {}
        for data_type in self.data_types:
            sx_data = SX_Data(len(barcodes), prep_dir, data_type)
            num_clones = sx_data.num_clones
            clones = sx_data.clones
            self.data_sources[data_type] = sx_data
        # this must be same for all modality.
        self.K = self.num_clones = num_clones
        self.clones = clones

        self.verbose = verbose

        # some hack here TODO
        global global_barcodes 
        global_barcodes = barcodes
        return

    ##################################################
    def _init_params(
        self,
        init_fix_params=None,
        init_params=None,
    ):
        params = {}
        if not init_params is None:
            for key, param in init_params.items():
                params[key] = param

        if "GEX" in self.data_sources:
            if params.get("GEX-lambda", None) is None:
                full_props = self.initialize_baseline_proportions("GEX")
                # only retain CNA-related bins
                params["GEX-lambda"] = full_props[self.data_sources["GEX"].ALL_MASK["CNP"]]

        if params.get("pi", None) is None:
            params["pi"] = np.ones(self.K) / self.K

        default_tau = 30
        default_inv_phi = 0.1  # inverse of phi
        if "GEX" in self.data_sources:
            if params.get("GEX-tau", None) is None:
                params["GEX-tau"] = np.full(
                    self.data_sources["GEX"].nrows_eff,
                    fill_value=default_tau,
                    dtype=np.float32,
                )
                params["GEX-inv_phi"] = np.full(
                    self.data_sources["GEX"].nrows_eff,
                    fill_value=default_inv_phi,
                    dtype=np.float32,
                )

        fix_params = {key: False for key in params.keys()}
        if not init_fix_params is None:
            for key in init_fix_params.keys():
                fix_params[key] = init_fix_params[key]

        return params, fix_params

    def initialize_baseline_proportions(self, modality: str):
        """this returns baseline proportions for all bins"""
        print(f"initialize baseline proportion for {modality}")
        if "cell_type" in self.barcodes:
            normal_labels = (
                ~self.barcodes["cell_type"].str.contains("tumor", case=False)
            ).to_numpy()
            num_normal_cells = np.sum(normal_labels)
            assert np.any(normal_labels)
            print(f"#annotated normal cells={num_normal_cells}/{self.num_barcodes}")
        else:
            print("infer normal cells using allele model")
            # TODO allele model
            pass
        base_props = compute_baseline_proportions(
            self.data_sources[modality].T, normal_labels
        )
        return base_props

    ##################################################
    def compute_log_likelihood(self, params: dict):
        global_lls = np.zeros((self.N, self.K), dtype=np.float32)
        # sum over all modalities
        for modality in self.data_types:
            sx_data: SX_Data = self.data_sources[modality]
            M = sx_data.apply_cnp_mask_shallow()

            # allele log-probs
            allele_ll_mat = _cond_betabin_logpmf(
                M["X"], M["Y"], M["D"], params[f"{modality}-tau"], M["BAF"]
            )
            allele_lls = allele_ll_mat.sum(axis=0) # (N,K)
            global_lls += allele_lls

            # total log-probs
            total_ll_mat = _cond_negbin_logpmf(
                M["T"], sx_data.Tn, M["C"], params[f"{modality}-lambda"], params[f"{modality}-inv_phi"]
            )
            total_lls = total_ll_mat.sum(axis=0) # (N,K)
            global_lls += total_lls

        global_lls += np.log(params["pi"])[None, :] # (N,K)
        log_marg = logsumexp(global_lls, axis=1) # (N,1)
        ll = np.sum(log_marg)
        return ll, log_marg, global_lls

    def _e_step(self, params: dict, t=0) -> np.ndarray:
        """compute allele and total log-probs for each modality

        Args:
            params (dict): parameters

        Returns:
            np.ndarray: gammas (N,K)
        """
        ll, log_marg, global_lls = self.compute_log_likelihood(params)

        # normalize
        gamma = np.exp(global_lls - logsumexp(global_lls, axis=1, keepdims=True))  # softmax

        # debug
        # self.barcodes.loc[:, self.clones] = gamma.round(5)
        # self.barcodes.to_csv(os.path.join(self.out_dir, f"e_step.iter{t}.tsv"), sep="\t", index=False, header=True)
        return gamma

    def _m_step(self, gamma: np.ndarray, params: dict, fix_params: dict, t=0):
        """m-step

        Args:
            gammas (np.ndarray): (N,K)
            params (dict): parameter values from previous iteration.
            fix_params (dict): which parameter is fixed.
        """

        if not fix_params["pi"]:
            # update mixing density for clone assignments
            params["pi"] = np.sum(gamma, axis=0) / self.N
        
        for modality in self.data_types:
            sx_data: SX_Data = self.data_sources[modality]
            if not fix_params.get(f"{modality}-lambda", True):
                # update baseline proportion
                # since normal cells assignment might change, 
                # baseline expression also update accordingly
                # but we only do it for first few iters?
                pass

            if not fix_params.get(f"{modality}-tau", True):
                # update over-dispersion parameter for betabin
                # bins with same CNP are shared
                for cnp_id, cnp_idx in sx_data.cnp_groups.items():
                    cnp_state = sx_data.cnp_id2state[cnp_id]


            if not fix_params.get(f"{modality}-inv_phi", True):
                # update over-dispersion parameter for negbin
                # bins with same CNP are shared
                for cnp_id, cnp_idx in sx_data.cnp_groups.items():
                    cnp_state = sx_data.cnp_id2state[cnp_id]

                
        return

    def inference(
        self,
        fix_params=None,
        init_params=None,
        max_iter=100,
        tol=1e-4,
        eps=1e-10
    ):
        print("Start EM")
        # Parameters
        params, fix_params = self._init_params(fix_params, init_params)
        if self.verbose:
            for key, param in params.items():
                print(f"{key}\t{param.shape}\tfixed=" + str(fix_params[key]))

        ll_trace = []
        prev_ll = -np.inf
        for t in range(max_iter):
            gamma = self._e_step(params, t)
            self._m_step(gamma, params, fix_params, t)

            ll, _, _ = self.compute_log_likelihood(params)
            ll_trace.append(ll)
            if self.verbose:
                print(f"[{t:03d}] log-likelihood = {ll:.6f}")
                print(params["pi"])

            if t > 0:
                rel_change = np.abs(ll - prev_ll) / (np.abs(prev_ll) + eps)
                if rel_change < tol:
                    print(f"Converged at iteration {t} (delta = {rel_change:.2e})")
                    break
            prev_ll = ll

        # potentially plot the LL here TODO ll_trace
        return params
    
    def map_decode(self, params: dict, label="cell_label", thres=0.65):
        print("Decode labels with MAP")
        posteriors = self._e_step(params)
        posts = self.barcodes.copy(deep=True)
        posts.loc[:, self.clones] = posteriors
        posts["max_posterior"] = posts[self.clones].max(axis=1)
        posts[label] = posts[self.clones].idxmax(axis=1)
        posts.loc[posts["max_posterior"] < thres, label] = "NA"
        return posts

    # def initialize_spot_proportions(self, modality: str):
    #     return


##################################################
# Likelihood functions
def _cond_betabin_logpmf(
    X: np.ndarray,
    Y: np.ndarray,
    D: np.ndarray,
    tau: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """
        compute loglik conditioned on labels per bin per cell per clone
        bb_ll_{g,n,k} = logP(Y_{g,n}|l_n=k;param)

    Args:
        X (np.ndarray): a-allele counts (G, N)
        Y (np.ndarray): b-allele counts (G, N)
        D (np.ndarray): total-allele counts (G, N)
        tau (np.ndarray): dispersion (G,)
        p (np.ndarray): BAF (G,K)
    
    Returns:
        np.ndarray: (G,N,K)
    """
    (G,N) = X.shape
    K = p.shape[1]

    # FIXME fixed, remove later
    # fig, axes = plt.subplots(1, 2)
    # dt = (Y - X)[:,global_barcodes.loc[global_barcodes["cell_type"]=="Tumor_cell"].index].flatten()
    # sns.histplot(x=dt, hue=dt==0, ax=axes[0], bins=50)
    
    # baf = np.divide(Y, D, where=D>0, out=np.full_like(D, fill_value=-1, dtype=np.float32))
    # sns.histplot(x=baf[:,global_barcodes.loc[global_barcodes["cell_type"]=="Tumor_cell"].index].flatten(), 
    #              ax=axes[1], bins=50, binrange=[0,1])
    # plt.savefig("tumor.png")

    # fig, axes = plt.subplots(1, 2)
    # dt = (Y - X)[:,global_barcodes.loc[global_barcodes["cell_type"]!="Tumor_cell"].index].flatten()
    # sns.histplot(x=dt, hue=dt==0, ax=axes[0], bins=50)
    
    # baf = np.divide(Y, D, where=D>0, out=np.full_like(D, fill_value=-1, dtype=np.float32))
    # sns.histplot(x=baf[:,global_barcodes.loc[global_barcodes["cell_type"]!="Tumor_cell"].index].flatten(), 
    #              ax=axes[1], bins=50, binrange=[0,1])
    # plt.savefig("normal.png")

    # (G, N, K)
    _X = X[:, :, None]
    _Y = Y[:, :, None]
    _D = D[:, :, None]
    _tau = np.broadcast_to(np.atleast_1d(tau)[:, None, None], (G, 1, 1))
    _p = p[:, None, :]

    a = _tau * _p
    b = _tau * (1.0 - _p)

    log_binom = gammaln(_D + 1) - gammaln(_Y + 1) - gammaln(_X + 1)
    ll = log_binom + betaln(_Y + a, _X + b) - betaln(a, b)
    return ll

def _cond_negbin_logpmf(
    T: np.ndarray,
    Tn: np.ndarray,
    C: np.ndarray,
    lambda_g: np.ndarray,
    inv_phi: np.ndarray,
) -> np.ndarray:
    """compute loglik conditioned on labels per bin per cell per clone
        bb_ll_{g,n,k} = logP(T_{g,n}|l_n=k;param)

    Args:
        T (np.ndarray): (G,N)
        Tn (np.ndarray): (N',)
        C (np.ndarray): (G,K)
        lambda_g (np.ndarray): (G,)
        inv_phi (np.ndarray): (G,)

    Returns:
        np.ndarray: (G,N,K)
    """
    G = T.shape[0]

    mu_gk = compute_rdr(lambda_g, C)
    mu_counts = mu_gk[:, None, :] * Tn[None, :, None] # (G,N,K)
    _T = T[:, :, None] # (G, N, K)
    
    _inv_phi = np.broadcast_to(np.atleast_1d(inv_phi)[:, None, None], (G, 1, 1))

    log_binom = gammaln(_T + _inv_phi) - gammaln(_inv_phi) - gammaln(_T + 1)
    ll = log_binom + _inv_phi * np.log(_inv_phi / (_inv_phi + mu_counts))
    ll = ll + _T * np.log(mu_counts / (_inv_phi + mu_counts))
    return ll
