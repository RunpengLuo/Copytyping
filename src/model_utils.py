import numpy as np
import pandas as pd

from scipy.special import softmax, expit, betaln, digamma, gammaln, logsumexp
from scipy.stats import binom, beta, norm
from scipy.stats import binomtest, chi2, norm, combine_pvalues, goodness_of_fit
from statsmodels.stats.multitest import multipletests
from scipy.optimize import minimize, minimize_scalar
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF


##################################################
def compute_baseline_proportions(T: np.ndarray, normal_labels: np.ndarray):
    # note that T here is full size, includes all bins.
    T_normal = T[:, normal_labels]
    base_raw = np.sum(T_normal, axis=1)
    base_props = base_raw / np.sum(base_raw)
    return base_props


def compute_rdr(lambda_g: np.ndarray, C: np.ndarray):
    """compute mu_{g,k}=C[g,k] / sum_{g}{lam_g * C[g,k]}

    Args:
        lambda_g (np.ndarray): (G,)
        C (np.ndarray): (G,K)
    """
    denom = (lambda_g[:, None] * C).sum(axis=0) # (K, )
    mu_gk = C / denom # (G, K)
    return mu_gk
    