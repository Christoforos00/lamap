from typing import Any, Dict, List
import numpy as np

from lamap.utils import Site, circular_mask

# ---------- ECDF interval mass ----------
def ecdf_interval_mass(sorted_sample: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Empirical mass P(a <= Z <= b) using binary searches over a sorted sample.
    a, b are broadcastable to the number of target cells (C).
    """
    left = np.searchsorted(sorted_sample, a, side='left')
    right = np.searchsorted(sorted_sample, b, side='right')
    n = max(sorted_sample.size, 1)
    mass = (right - left) / n
    # mass ∈ [0,1] naturally; clip guards rare numerical quirks
    return np.clip(mass, 0.0, 1.0)


def build_site_ecdfs(sites: List[Site],
                     vars_dict: Dict[str, np.ndarray],
                     catchment_radius: float) -> Dict[Any, Dict[str, np.ndarray]]:
    """Per-site, per-variable sorted samples from circular catchments."""
    H, Wcols = next(iter(vars_dict.values())).shape
    samples = {s.id: {} for s in sites}
    for s in sites:
        mask = circular_mask(s.y, s.x, catchment_radius, (H, Wcols))
        for v, arr in vars_dict.items():
            vals = arr[mask].ravel()
            vals = vals[np.isfinite(vals)]
            samples[s.id][v] = np.sort(vals)
    return samples