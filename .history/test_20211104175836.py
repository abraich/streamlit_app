"""Computes the values at x of the piecewise constant function
    on the given `bins` with values in `values`"""
import numpy as np
def piecewise(x, bins, values): 
    """Computes the values at x of the piecewise constant function
    on the given `bins` with values in `values`"""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(bins, np.ndarray):
        bins = np.array(bins)
    if not isinstance(values, np.ndarray):
        values = np.array(values)
    if len(x) != len(bins) - 1:
        raise ValueError("x and bins must have the same length")
    if len(values) != len(bins) - 1:
        raise ValueError("values and bins must have the same length")
    if not np.all(np.diff(bins) > 0):
        raise ValueError("bins must be in ascending order")
    if not np.all(x >= bins[0]) and np.all(x <= bins[-1]):
        raise ValueError("x must be in the range of bins")
    if not np.all(values >= bins[0]) and np.all(values <= bins[-1]):
        raise ValueError("values must be in the range of bins")
    if not np.all(np.diff(values) > 0):
        raise ValueError("values must be in ascending order")
    return np.piecewise(x, bins, values)
    


    