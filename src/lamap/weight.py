import numpy as np

# ---------- Distance weighting ----------
def w_exponential(dist: np.ndarray, rate: float, scale: float) -> np.ndarray:
    """
    Exponential weighting used in your R code: exp( -(dist/scale) * rate ).
    dist shape [S, C] or [C] or [S], returns same shape.
    """
    return np.exp(-(dist/scale) * rate)

def w_uniform(dist: np.ndarray, cutoff: float) -> np.ndarray:
    return (dist <= cutoff).astype(np.float64)