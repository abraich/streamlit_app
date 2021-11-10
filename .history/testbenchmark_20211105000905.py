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
    return mise
# test the function
s_true=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
s_pred=np.array([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1])
interval=0.1
mise=Mise(s_true,s_pred,interval)
print(mise)