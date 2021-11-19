import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# get quantile  : input S_list, times, percentile : if percentile = min(x : S[i] > x) , return times[armindex]
def get_quantile(S_list, times, percentile):
    for i in range(len(S_list)):
        if S_list[i] > percentile:
            return times[i]
    return times[-1]
    
if __name__ == '__main__':
    n= 100
    times = np.linspace(0,10,n) # time points
    
    S_list = 1.5- sigmoid(times) # survival probability
    
    plt.plot(times,S_list)
    plt.show()
    
    # get quantile
    print(get_quantile(S_list, times, 0.5))
  
    