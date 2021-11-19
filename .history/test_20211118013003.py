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
if __name__ == '__main__':
    S_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(get_quantile(S_list, 0.7))
    
    

