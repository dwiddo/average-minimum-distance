import numpy as np
from scipy.spatial.distance import cdist

# l-infinity distance (between two AMDs)
def linf(v, v_):
    return np.amax(np.abs(v-v_))

# Earth mover's distance between two PDDs
def emd(pdd, pdd_):
    from .core.Wasserstein import wasserstein
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric='chebyshev')
    return wasserstein(pdd[:, 0], pdd_[:, 0], dm)

# either: two 2D matrices (compare rows as AMDs pairwise)
#         a 1D reference and 2D (compare one to many)

# for batch comparisons of AMDs.
# either arg can be any of: an AMD, a list of AMDs, or a np.array with AMDs in rows.
# returns a float for 1 v 1 comparison, a vector for 1 v many, a distance matrix for many vs many.
def compare(reference, comparison, k=None):
    
    if isinstance(reference, list):
        reference = np.array(reference)
    if isinstance(comparison, list):
        comparison = np.array(comparison)
        
    if not reference.shape[-1] == comparison.shape[-1]:
        raise ValueError(f'Reference(s) and comparison(s) have incompatible shapes {reference.shape}, {comparison.shape} as {reference.shape[-1]} != {comparison.shape[-1]}')
    
    if k is not None:
        if k >= reference.shape[-1] or k <= 0:
            raise ValueError(f'Value of k={k} out of accepted range (1,{reference.shape[-1]})')
    
    if len(comparison.shape) == 1:
        comparison = np.array([comparison])
    if len(reference.shape) == 1:
        reference = np.array([reference])
        
    return cdist(reference, comparison, metric='chebyshev')