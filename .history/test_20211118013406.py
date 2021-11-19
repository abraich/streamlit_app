import numpy as np
# get quantile from survival probability for a given percentile
def get_quantile(S_list,times, p):
    S_list = np.array(S_list)
    times = np.array(times)
    # sort by survival time
    S_list_sorted = np.sort(S_list)
    times_sorted = np.sort(times)
    # get the index of the first value of the sorted list that is greater than p
    index = np.searchsorted(S_list_sorted, p)
    # if p is greater than all values in the list, return the last value
    if index == len(S_list_sorted):
        index = -1
    # get the corresponding time
    return times_sorted[index]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# test 
if __name__ == '__main__':
    n= 100
    S_list = sigmoid(np.random.uniform(0,1,n))
    times = np.random.uniform(0,1,n)
    p = 0.5
    print(get_quantile(S_list,times,p))
    