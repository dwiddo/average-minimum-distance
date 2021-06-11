import numpy as np
from scipy.spatial.distance import cdist

def linf(v, v_):
    '''l-infinity distance (between two AMDs)'''
    return np.amax(np.abs(v-v_))

def emd(pdd, pdd_):
    '''Earth mover's distance between two PDDs'''
    from .core.Wasserstein import wasserstein
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric='chebyshev')
    return wasserstein(pdd[:, 0], pdd_[:, 0], dm)


def compare(reference, comparison, k=None):
    
    '''for batch comparisons of AMDs.
       Either arg can be any of: an AMD, a list of AMDs, or a np.array with AMDs in rows.
       Returns a float for 1 v 1 comparison, a vector for 1 v many, a distance matrix for many vs many.
    '''
    
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

if __name__ == '__main__':
    pass