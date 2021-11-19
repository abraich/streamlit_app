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
    # for each model different color
    plt.setp(bp['boxes'],
                color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])   
    plt.setp(bp['whiskers'],'color','black')
    plt.setp(bp['fliers'], color='black')
    plt.setp(bp['medians'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['means'], color='black')

    # set xlabel and ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # set title
    ax.set_title('Boxplot of Mise')
    # set xticks
    ax.set_xticks([i+1 for i in range(n)])
    # set xticklabels
    ax.set_xticklabels(models_list)
    # set ylim
    ax.set_ylim(0, 1)
    # set yticks
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    # set grid
    ax.grid(True)
    # show plot
    plt.show()
    
# test
if __name__ == '__main__':
    models_list = ['model_0', 'model_1', 'model_2']
    d_exp = {