from dataclasses import dataclass
from typing import Any, Tuple
import numpy as np

import numpy as np
from typing import List, Tuple

# ---------- Small utility ----------
@dataclass
class Site:
    id: Any
    x: float
    y: float

def circular_mask(cy: float, cx: float, radius: float, shape: Tuple[int,int]) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy)**2 + (xx - cx)**2 <= radius**2


def sample_sites_from_mask(mask: np.ndarray, n: int, rng=None) -> List[Site]:
    """
    Pick n distinct cells uniformly from a boolean mask (True = eligible).
    Returns a list of Site objects with cell centers as coordinates.
    """
    rng = np.random.default_rng() if rng is None else rng
    H, Wcols = mask.shape
    idx = np.flatnonzero(mask.ravel())
    if idx.size < n:
        raise ValueError(f"Mask has only {idx.size} eligible cells, need {n}.")
    picks = rng.choice(idx, size=n, replace=False)
    yy, xx = np.divmod(picks, Wcols)
    # +0.5 to match your existing grid convention
    return [Site(f"N{i+1}", float(xx[i] + 0.5), float(yy[i] + 0.5)) for i in range(n)]


def log1mexp_vector(x: np.ndarray) -> np.ndarray:
    """
    Stable elementwise log(1 - exp(x)) for x <= 0.
    Handles x = -inf (returns 0), x = 0 (returns -inf).
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.minimum(x, 0.0)  # enforce x <= 0 for safety
    cutoff = -np.log(2.0)   # ~ -0.6931
    with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
        return np.where(
            x <= cutoff,
            np.log1p(-np.exp(x)),     # safe when exp(x) is small
            np.log(-np.expm1(x))      # safe when exp(x) ~ 1
        )

def union_from_logQ(logQ: np.ndarray) -> np.ndarray:
    """
    Given logQ [S, C] with logQ_jc = log(q_jc) <= 0 or -inf,
    compute U[c] = 1 - prod_j (1 - q_jc) in a numerically stable way.
    """
    X = np.asarray(logQ, dtype=np.float64)
    # Treat any NaN as 'no contribution' (log 0):
    X = np.where(np.isfinite(X), X, -np.inf)
    X = np.minimum(X, 0.0)  # enforce logQ <= 0
    log1mQ = log1mexp_vector(X)                       # [S, C]
    log1mU = np.sum(log1mQ, axis=0, dtype=np.float64) # [C]
    with np.errstate(over='ignore', under='ignore'):
        U = 1.0 - np.exp(log1mU)                      # [C]
    return np.clip(U, 0.0, 1.0)

# ---------- Stable union (independent sites) ----------
def union_independent_from_q(q: np.ndarray) -> np.ndarray:
    """
    q shape [S, C]; returns U shape [C].
    U = 1 - Π_j (1 - q_j). Compute via log(1-U) = Σ_j log(1 - q_j).
    """
    q = np.clip(q, 0.0, 1.0 - 1e-12)  # avoid log(0) in log1p(-q)
    log1m_q = np.log1p(-q)            # log(1 - q)
    log1m_U = np.sum(log1m_q, axis=0)
    U = 1.0 - np.exp(log1m_U)
    return np.clip(U, 0.0, 1.0)

# ---------- K-nearest sites per cell ----------
def topk_mask_from_dist(dist_SC: np.ndarray, k: int) -> np.ndarray:
    """
    dist_SC shape [S, C]. Returns boolean mask [S, C] where True means
    "site is among the k nearest for that cell".
    """
    S, C = dist_SC.shape
    k_eff = min(k, S)
    idx = np.argpartition(dist_SC, kth=k_eff-1, axis=0)[:k_eff, :]  # (k_eff, C)
    mask = np.zeros((S, C), dtype=bool)
    # robust column-wise fill (no tricky broadcasting)
    for c in range(C):
        mask[idx[:, c], c] = True
    return mask


