import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# get quantile 
def get_quantile(p, S_list,times):
    
    
if __name__ == '__main__':
    n= 100
    times = np.linspace(0,10,n) # time points
    
    S_list = 1.5- sigmoid(times) # survival probability
    
    plt.plot(times,S_list)
    plt.show()
    for p in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        print(p,get_quantile(S_list,times,p))
    