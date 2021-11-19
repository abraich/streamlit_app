import numpy as np
import matplotlib.pyplot as plt

# boxplot function

def boxplot(models_list,
            d_exp,
            xlabel='Models',
            ylabel='Mise Surv 0',
            option=0):
    n = len(models_list)
    if option == 0:
        input = [d_exp[f'mise_0_{model_name}'] for model_name in models_list]
    if option == 1:
        input = [d_exp[f'mise_1_{model_name}'] for model_name in models_list]
    if option == 2:
        input = [d_exp[f'CATE_{model_name}'] for model_name in models_list]

    fig, ax = plt.subplots()
    bp = ax.boxplot(input, widths=0.2, sym='', patch_artist=True)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='black')
    plt.setp(bp['medians'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.setp(bp['means'], color='blue')
    
    ax.set_xticklabels(models_list)
    ax.set_xticks(np.arange(1, n+1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    
  
# test
if __name__ == "__main__":
    models_list = ['A', 'B', 'C']
    d_exp = {
        'mise_0_A': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mise_0_B': [0.2, 0.3, 0.4, 0.5, 0.6],
        'mise_0_C': [0.3, 0.4, 0.5, 0.6, 0.7],
        'mise_1_A': [0.1, 0.2, 0.3, 0.4, 0.5],
        'mise_1_B': [0.2, 0.3, 0.4, 0.5, 0.6],
        'mise_1_C': [0.3, 0.4, 0.5, 0.6, 0.7],
        'CATE_A': [0.1, 0.2, 0.3, 0.4, 0.5],
        'CATE_B': [0.2, 0.3, 0.4, 0.5, 0.6],
        'CATE_C': [0.3, 0.4, 0.5, 0.6, 0.7]
    }
    boxplot(models_list, d_exp, option=0)
    boxplot(models_list, d_exp, option=1)
    boxplot(models_list, d_exp, option=2)