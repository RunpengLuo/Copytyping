import os
import sys

# import numpy as np
# import pandas as pd


import jax
import jax.numpy as jnp
from jax.nn import softmax
from jax.scipy.special import gammaln
from jaxopt import LBFGS

# from utils import *
# from sx_data import *
# from model_utils import *

# import matplotlib.pyplot as plt
# import seaborn as sns


# class SP_Model:
#     """Single-spot MAP model, no spatial information, tumor purity=1 for each cell."""
#     def __init__(self) -> None:
#         pass

# def sp_solver(
#     X: np.ndarray, # (G,N)
#     Y: np.ndarray, # (G,N)
#     D: np.ndarray, # (G,N)
#     T: np.ndarray, # (G,N)
#     L: np.ndarray, # (N,N) graph laplacian
#     A: np.ndarray, # (G,K)
#     B: np.ndarray, # (G,K)
#     C: np.ndarray, # (G,K)
#     BAF: np.ndarray, # (G,K)
#     params: dict,
# ):
#     """
#     Spatial data modelling
#     Logistic-Gaussian Markov random field
#     X,Y,D: allele counts (G,N)
#     T: total counts (G,N)
#     L: graph laplacian, L=D-W (N,N)
#     A,B,C: copy-number profile (N,K)
#     BAF: b-allele frequency for copy-number profile (N,K)
#     """


######################


def log_negbin(x, mu, phi):
    return (
        gammaln(x + phi)
        - gammaln(x + 1)
        - gammaln(phi)
        + phi * (jnp.log(phi) - jnp.log(phi + mu))
        + x * (jnp.log(mu) - jnp.log(phi + mu))
    )


def log_betabin(y, d, p, tau):
    a, b = tau * p, tau * (1 - p)
    return (
        gammaln(y + a)
        + gammaln(d - y + b)
        + gammaln(a + b)
        - gammaln(d + a + b)
        - gammaln(a)
        - gammaln(b)
    )


def gmrf_prior(Z, L, beta):
    return 0.5 * beta * jnp.sum(Z * (L @ Z))


def entropy(U):
    return -jnp.sum(U * jnp.log(U + 1e-8), axis=-1)  # per-spot entropy


def objective(Z, T, Tn, Y, D, L, mu_gk, p_gk, lambda_g, phi_g, tau_g, beta, lam):
    # clone mixture (N,K)
    U = softmax(Z, axis=-1)

    # mixture means per bin g, per spot n
    mu_tilde = (U @ mu_gk.T) * lambda_g * Tn[:, None]  # (N,G)
    p_tilde = (U @ (mu_gk * p_gk).T) / (U @ mu_gk.T)

    # likelihoods
    ll_nb = jnp.sum(log_negbin(T, mu_tilde, phi_g))
    ll_bb = jnp.sum(log_betabin(Y, D, p_tilde, tau_g))

    # spatial + entropy
    lp = -gmrf_prior(Z, L, beta)
    h = -lam * jnp.sum(entropy(U))

    return -(ll_nb + ll_bb + lp + h)  # negative for minimization


def objective_wrapped(Z, p):
    return objective(Z, **p)


def fit_map(Z0, params, maxiter=200):
    # f_val, f_grad = jax.value_and_grad(objective_wrapped)(Z0, params)
    # print("Objective value:", f_val)
    # print("Gradient shape:", f_grad.shape)

    solver = LBFGS(fun=objective_wrapped, maxiter=maxiter, tol=1e-5)
    opt_res = solver.run(init_params=Z0, p=params)
    Z_map = opt_res.params
    U_map = softmax(Z_map, axis=-1)
    return Z_map, U_map, opt_res.state.value


# Example small data
N, G, K = 30, 200, 3
key = jax.random.PRNGKey(0)

T = jax.random.poisson(key, 5.0, (N, G))
Tn = T.sum(axis=1)
Y = jax.random.binomial(key, 20, 0.3, (N, G))
D = jnp.full((N, G), 20.0)
L = jnp.eye(N)  # replace with Visium Laplacian

mu_gk = jnp.ones((G, K)) * 1.0
p_gk = jnp.ones((G, K)) * 0.4
lambda_g = jnp.ones(G)
phi_g = jnp.ones(G) * 10.0
tau_g = jnp.ones(G) * 50.0

params = dict(
    T=T,
    Tn=Tn,
    Y=Y,
    D=D,
    L=L,
    mu_gk=mu_gk,
    p_gk=p_gk,
    lambda_g=lambda_g,
    phi_g=phi_g,
    tau_g=tau_g,
    beta=1.0,
    lam=0.1,
)

Z0 = jnp.zeros((N, K))
Z_map, U_map, loss = fit_map(Z0, params)
print("Final loss:", float(loss))
print("Inferred U (first 3 rows):")
print(U_map)
