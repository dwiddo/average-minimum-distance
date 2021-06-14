import numpy as np
from scipy.spatial.distance import cdist

def linf(v, v_):
    """l-infinity distance between vectors (AMDs)."""
    return np.amax(np.abs(v-v_))

def emd(pdd, pdd_, metric='chebyshev', **kwargs):
    """
    Earth mover's distance between two PDDs.
    
    Parameters
    ----------
    pdd : ndarray
        A :math:`m_a` by k PDD matrix (with weights in the first column).
    pdd_ : ndarray
        A :math:`m_b` by k PDD matrix (with weights in the first column).
    metric : str or callable, optional
        Usually rows are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.
        
    Returns
    -------
    float
        Earth mover's distance between PDDs, where rows of the PDDs are
        compared with the metric.

    Raises
    ------
    ValueError
        Thrown if reference and comparison do not have 
        the same number of columns.
    
    """
    
    from .core.Wasserstein import wasserstein
    
    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    
    return wasserstein(pdd[:, 0], pdd_[:, 0], dm)


def compare(reference, comparison, k=None, metric='chebyshev', **kwargs):
    """
    For batch or single comparisons of AMDs for a single k.
    Computes distance between each pair of the two collections of inputs, 
    essentially wrapping scipy's cdist.
    
    Parameters
    ----------
    reference : ndarray or list of ndarrays
        An :math:`m_a` by n array (of :math:`m_a` vectors/AMDs). 
        Can also be a list of vectors or a single vector.
    comparison : ndarray or list of ndarrays
        An :math:`m_b` by n array (of :math:`m_b` vectors/AMDs). 
        Can also be a list of vectors or a single vector.
    k : int, optional
        Only compare vectors/AMDs up to the k-th element.
    metric : str or callable, optional
        Usually AMDs are compared with the Chebyshev/l-infinity distance.
        Can take any metric + kwargs accepted by scipy.spatial.distance.cdist.

    Returns
    -------
    ndarray
        A :math:`m_a` by :math:`m_b` distance matrix.
        The :math:`ij` th entry is the distance between reference[i]
        and comparison[j] given by the metric.

    Raises
    ------
    ValueError
        Thrown if reference and comparison do not have the same number of
        columns, or if k <= 0 or k > the number of columns. 
    
    # if list is not convertible to ndarray?
    """
    
    if isinstance(reference, list):
        reference = np.array(reference)
    if isinstance(comparison, list):
        comparison = np.array(comparison)
        
    if k is not None:
        if k >= reference.shape[-1] or k <= 0:
            raise ValueError(f'Value of k={k} out of accepted range (1,{reference.shape[-1]})')
    
    if len(comparison.shape) == 1:
        comparison = np.array([comparison])
    if len(reference.shape) == 1:
        reference = np.array([reference])
    
    return cdist(reference[:,:k], comparison[:,:k], metric=metric, **kwargs)

if __name__ == '__main__':
    pass