import numpy as np

def get_quantile(data, quantile):
    """
    Return the quantile value of the data.
    """
    return np.percentile(data, quantile)


