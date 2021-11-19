import numpy as np
import matplotlib.pyplot as plt

# boxplot function

def boxplot(data, labels, title, xlabel, ylabel, save_path):
    plt.boxplot(data, labels=labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.show()


  
    