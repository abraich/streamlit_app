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

    # add x-ticks
    plt.xticks([1, 2, 3, 4, 5, 6], models_list, rotation=90)

    # add y-ticks
    plt.yticks(np.arange(0, 1.1, 0.1))

    # add x-label
    plt.xlabel(xlabel)

    # add y-label
    plt.ylabel(ylabel)

    # add title
    plt.title('Boxplot Mise sur le test set')

    # show plot
    plt.show()

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
        'mise_0_model_0': np.random.rand(100),
        'mise_0_model_1': np.random.rand(100),
        'mise_0_model_2': np.random.rand(100),
        'mise_1_model_0': np.random.rand(100),
        'mise_1_model_1': np.random.rand(100),
        'mise_1_model_2': np.random.rand(100),
        'CATE_model_0': np.random.rand(100),
        'CATE_model_1': np.random.rand(100),
        'CATE_model_2': np.random.rand(100),
    }
    boxplot(models_list, d_exp, option=0)
    boxplot(models_list, d_exp, option=1)
    boxplot(models_list, d_exp, option=2)