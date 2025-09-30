
from typing import Dict, List, Tuple

import numpy as np
from typing import Any

from lamap.ecdf import build_site_ecdfs, ecdf_interval_mass
from lamap.utils import Site, sample_sites_from_mask, topk_mask_from_dist, union_from_logQ
from lamap.weight import w_exponential


def lamap_from_sites(sites: List[Site],
                     vars_dict: Dict[str, np.ndarray],
                     eps: Dict[str, float],
                     K: int,
                     rate: float,
                     scale: float,
                     catchment_radius: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute a LAMAP surface U (H×W) for a given site set and parameters.
    Returns (U, debug) where debug holds intermediates if you want to inspect.
    """
    # grid & flatten
    H, Wcols = next(iter(vars_dict.values())).shape
    ygrid, xgrid = np.meshgrid(np.arange(H) + 0.5, np.arange(Wcols) + 0.5, indexing='ij')
    Xc = xgrid.ravel()
    Yc = ygrid.ravel()
    C = H * Wcols
    Vnames = list(vars_dict.keys())
    V = len(Vnames)
    # bounds per variable
    Xv = {v: vars_dict[v].ravel().astype(np.float64) for v in Vnames}
    A = {v: Xv[v] - eps[v] for v in Vnames}
    B = {v: Xv[v] + eps[v] for v in Vnames}

    # ECDF samples per site
    site_samples = build_site_ecdfs(sites, vars_dict, catchment_radius)

    # distances [S, C]
    S = len(sites)
    D = np.empty((S, C), dtype=np.float64)
    for j, s in enumerate(sites):
        D[j, :] = np.hypot(Xc - s.x, Yc - s.y)

    # K-nearest mask
    mask_topk = topk_mask_from_dist(D, min(K, S))

    # distance weights (log)
    Ww = w_exponential(D, rate=rate, scale=scale)  # (0,1]
    logW = np.log(Ww, where=(Ww > 0), out=np.full_like(Ww, -np.inf))

    # logP[v, j, c]
    tiny = 1e-300
    logP = np.full((V, S, C), -np.inf, dtype=np.float64)
    for vi, v in enumerate(Vnames):
        a, b = A[v], B[v]
        for j, s in enumerate(sites):
            ss = site_samples[s.id][v]
            if ss.size == 0:
                continue
            mass = ecdf_interval_mass(ss, a, b)       # [C] in [0,1]
            logP[vi, j, :] = np.log(np.maximum(mass, tiny))

    # sum across variables (independence within site)
    logPsite = np.sum(logP, axis=0, dtype=np.float64)      # [S, C]
    logPsite[~mask_topk] = -np.inf                          # K-nearest only

    # add log weights and union across sites
    logQ = logW + logPsite                                  # [S, C]
    U_flat = union_from_logQ(logQ)                          # [C]
    U = U_flat.reshape(H, Wcols)

    debug = dict(D=D, mask_topk=mask_topk, logW=logW, logP=logP,
                 logPsite=logPsite, logQ=logQ, site_samples=site_samples)
    return U, debug

def bootstrap_null_lamap(
    n_sites: int,
    R: int,
    vars_dict: dict,
    eps: dict,
    K: int,
    rate: float,
    scale: float,
    catchment_radius: float,
    mask: np.ndarray = None,
    seed=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo null: draw site sets under mask (CSR if mask=None), compute LAMAP R times.
    Returns (mean_U, sd_U), each H×W.
    """
    rng = np.random.default_rng(seed)
    H, Wcols = next(iter(vars_dict.values())).shape
    if mask is None:
        mask = np.ones((H, Wcols), dtype=bool)

    acc_mean = np.zeros((H, Wcols), dtype=np.float64)
    acc_m2   = np.zeros((H, Wcols), dtype=np.float64)  # for Welford online variance

    for r in range(1, R + 1):
        sites_r = sample_sites_from_mask(mask, n_sites, rng)
        U_r, _ = lamap_from_sites(sites_r, vars_dict, eps, K, rate, scale, catchment_radius)
        # online mean/variance (per cell)
        delta = U_r - acc_mean
        acc_mean += delta / r
        acc_m2   += delta * (U_r - acc_mean)
        if R <= 10 or (r % max(1, R // 10) == 0):
            print(f"null bootstrap {r}/{R}")

    mean_U = acc_mean
    sd_U = np.sqrt(acc_m2 / max(R - 1, 1))
    return mean_U, sd_U