import numpy as np
# get quantile from survival probability for a given percentile
def get_quantile(S_list, p):
    '''
    :param S_list: list of survival probability
    :param p: percentile
    :return: quantile
    '''
    S_list = np.array(S_list)
    quantile = np.quantile(S_list, p)
    return quantile

# test

    

