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
    # for each boxplot different color
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['medians'], color='black')
    plt.setp(bp['fliers'], color='black')
    plt.setp(bp['caps'], color='black')

    ax.set_xticklabels(models_list)
    ax.set_xticks(np.arange(1, n + 1))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title('Boxplot of the Mise survenance')
    plt.show()

# test
if __name__ == '__main__':
    