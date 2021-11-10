import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# define a function calculate the Mean integrated squared error using trapez  betwen two survival curves 

def Mise(s_true,s_pred,interval):
    # s_true and s_pred are two survival curves with the same length
    # interval is the time interval between two data points
    # calculate the area under the curve
    a=np.trapz(s_true,dx=interval)
    b=np.trapz(s_pred,dx=interval)
    # calculate the mean integrated squared error
    mise=np.trapz(np.power(s_true-s_pred,2),dx=interval)/a
    
    # plot the survival curves 
    plt.figure(figsize=(10,5))
    plt.plot(s_true,label='True')
    plt.plot(s_pred,label='Pred')
    plt.legend()
    plt.show()
    return mise
# test the function

def sigmoid(x):
    return 1/(1+np.exp(-x))

s_true=1-sigmoid(np.arange(-10,10,0.1))
s_pred=1-sigmoid(np.arange(-10,10,0.1)+sigmoid(np.random.normal(-10,1,len(s_true))))
interval=0.1
mise=Mise(s_true,s_pred,interval)
print(mise)