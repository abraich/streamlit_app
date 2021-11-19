import numpy as np
import matplotlib.pyplot as plt
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
    times = np.linspace(0,10,n) # time points
    
    S_list = 1- sigmoid(times) # survival probability
    
    plt.plot(S_list,times,'.')
    plt.show()
    for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print(p,get_quantile(S_list,times,p))
    