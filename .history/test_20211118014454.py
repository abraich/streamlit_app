import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# get quantile 
def get_quantile(perc, S_list,times,eps=1e-6):
    S_list = np.array(S_list)
    for i in range(len(S_list)):
        if S_list[i] > perc:
            return times[i], S_list[i]
    return times[-1], S_list[-1]
    
    
if __name__ == '__main__':
    n= 100
    times = np.linspace(0,10,n) # time points
    
    S_list = 1.5- sigmoid(times) # survival probability
    
    plt.plot(times,S_list)
    plt.show()
    
    # get quantile
    print(get_quantile(0.5,S_list,times))
  
    