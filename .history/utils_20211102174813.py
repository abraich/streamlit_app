


# from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torchtuples as tt 

import pandas as pd
import seaborn as sns
import torch  # For building the networks
import torch.nn as nn
import torch.nn.functional as F

from matplotlib.pyplot import figure
from pycox.models import PMF , utils
from scipy.linalg import toeplitz
from scipy.stats import bernoulli, multivariate_normal
from sklearn.preprocessing import StandardScaler


from torch import Tensor
from torchtuples import TupleTree
import scipy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.models.svm import KernelSVMModel
from pysurvival.models.survival_forest import ConditionalSurvivalForestModel
from pysurvival.models.multi_task import NeuralMultiTaskModel
from pysurvival.models.parametric import GompertzModel
from pysurvival.models.semi_parametric import NonLinearCoxPHModel

from pysurvival.models.parametric import ExponentialModel
from pysurvival.models.parametric import WeibullModel

from pysurvival.models.parametric import LogLogisticModel
from pysurvival.models.parametric import LogNormalModel


from numpy.random import binomial, multivariate_normal

from scipy.integrate import simps
from scipy.linalg.special_matrices import toeplitz
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm


from numba import jit

from lifelines import KaplanMeierFitter, CoxPHFitter

from pysurvival.models.survival_forest import ExtraSurvivalTreesModel
from pysurvival.models.multi_task import NeuralMultiTaskModel

seed = 31415
np.random.seed(seed)


scaler = preprocessing.MinMaxScaler()


sns.set(color_codes=True)


def sigmoid(x):
    idx = x > 0
    out = np.empty(x.size)
    out[idx] = 1 / (1. + np.exp(-x[idx]))
    exp_x = np.exp(x[~idx])
    out[~idx] = exp_x / (1. + exp_x)
    return out

#  Wasserstein distance


class SinkhornDistance(nn.Module):
    """[summary]

    """

    def __init__(self, eps, max_iter, reduction="none"):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = (torch.empty(batch_size,
                          x_points,
                          dtype=torch.float,
                          requires_grad=False).fill_(1.0 / x_points).squeeze())
        nu = (torch.empty(batch_size,
                          y_points,
                          dtype=torch.float,
                          requires_grad=False).fill_(1.0 / y_points).squeeze())

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = (self.eps * (torch.log(mu + 1e-8) -
                             torch.logsumexp(self.M(C, u, v), dim=-1)) + u)
            v = (self.eps *
                 (torch.log(nu + 1e-8) -
                  torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) +
                 v)
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == "mean":
            cost = cost.mean()
        elif self.reduction == "sum":
            cost = cost.sum()

        return cost  # , pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin))**p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1


scheme_subd = "equidistant"  # 'quantiles'


def load_data_sim_benchmark(path="./sim_surv"):

    df = pd.read_csv(path + ".csv")

    dim = df.shape[1]-8

    x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
    leave = x_z_list + ["event", "T_f_cens"]

    ##
    rs = ShuffleSplit(test_size=.4, random_state=0)
    df_ = df[leave].copy()
    for train_index, test_index in rs.split(df_):
        df_train = df_.drop(test_index)
        df_test = df_.drop(train_index)

    def get_separ_data(x):
        mask_1 = x["tt"] == 1
        mask_0 = x["tt"] == 0
        x_1 = x[mask_1].drop(columns="tt")
        x_0 = x[mask_0].drop(columns="tt")
        return x_0, x_1

    df_train_0,  df_train_1 = get_separ_data(df_train)
    df_test_0, df_test_1 = get_separ_data(df_test)

    x_0_train = df_train_0.iloc[:, :-2].values
    e_0_train = df_train_0.iloc[:, -2].values

    x_1_train = df_train_1.iloc[:, :-2].values
    e_1_train = df_train_1.iloc[:, -2].values

    T_f_0_train = df_train_0.iloc[:, -1].values
    T_f_1_train = df_train_1.iloc[:, -1].values

    x_0_test = df_train_0.iloc[:, :-2].values
    e_0_test = df_train_0.iloc[:, -2].values

    x_1_test = df_train_1.iloc[:, :-2].values
    e_1_test = df_train_1.iloc[:, -2].values

    T_f_0_test = df_train_0.iloc[:, -1].values
    T_f_1_test = df_train_1.iloc[:, -1].values

    return df_train, df_test, df_train_0, df_train_1, df_test_0, df_test_1, \
        x_0_train, e_0_train, x_1_train, e_1_train, T_f_0_train, T_f_1_train, x_0_test,\
        e_0_test, x_1_test, e_1_test, T_f_0_test, T_f_1_test


def load_data_sim_hd(path="./sim_surv", pmf=True, num_durations=12, dim=2):
    """[summary]

    Args:
        path (str, optional): [description]. Defaults to "./sim_surv".
        pmf (bool, optional): [description]. Defaults to True.
        num_durations (int, optional): [description]. Defaults to 12.
        dim (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(path + ".csv")

    x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
    leave = x_z_list + ["event", "T_f_cens"]
    rs = ShuffleSplit(n_splits=100, test_size=.4, random_state=0)
    df_ = df[leave].copy()
    for train_index, test_index in rs.split(df_):
        df_train = df_.drop(test_index)
        df_test = df_.drop(train_index)
        df_val = df_test.sample(frac=0.5)
        df_test = df_test.drop(df_val.index)
        if pmf:
            labtrans = PMF.label_transform(num_durations, scheme=scheme_subd)

        def get_target(df):
            return (df["T_f_cens"].values, df["event"].values)

        y_train_surv = labtrans.fit_transform(*get_target(df_train))
        y_val_surv = labtrans.transform(*get_target(df_val))
        # à modifier en fonction des arguments dans forward / loss
        # tt.tuplefy(x_train, (y_train_surv, x_train))
        train = (df_train[x_z_list].values.astype("float32"), y_train_surv)
        val = (
            df_val[x_z_list].values.astype("float32"),
            y_val_surv,
        )  # tt.tuplefy(x_val, (y_val_surv, x_val))
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
        x_test = df_test[x_z_list].values.astype("float32")
    return train[0], train[1], train, val, durations_test, events_test, labtrans, x_test


def load_data_sim_tcga(path="./sim_surv_fromdata_tcga", pmf=True, num_durations=12, dim=2):
    """[summary]

    Args:
        path (str, optional): [description]. Defaults to "./sim_surv".
        pmf (bool, optional): [description]. Defaults to True.
        num_durations (int, optional): [description]. Defaults to 12.
        dim (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    df = pd.read_csv(path + ".csv")

    x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
    leave = x_z_list + ["event", "T_f_cens"]
    rs = ShuffleSplit(n_splits=100, test_size=.4, random_state=0)
    df_ = df[leave].copy()
    for train_index, test_index in rs.split(df_):
        df_train = df_.drop(test_index)
        df_test = df_.drop(train_index)
        df_val = df_test.sample(frac=0.5)
        df_test = df_test.drop(df_val.index)
        if pmf:
            labtrans = PMF.label_transform(num_durations, scheme=scheme)

        def get_target(df):
            return (df["T_f_cens"].values, df["event"].values)

        y_train_surv = labtrans.fit_transform(*get_target(df_train))
        y_val_surv = labtrans.transform(*get_target(df_val))

        # à modifier en fonction des arguments dans forward / loss
        # tt.tuplefy(x_train, (y_train_surv, x_train))
        train = (df_train[x_z_list].values.astype("float32"), y_train_surv)
        val = (
            df_val[x_z_list].values.astype("float32"),
            y_val_surv,
        )  # tt.tuplefy(x_val, (y_val_surv, x_val))
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
        x_test = df_test[x_z_list].values.astype("float32")
    return train[0], train[1], train, val, durations_test, events_test, labtrans, x_test


#  Helpers
def get_data(input):
    x = input[:, :-1]
    t = input[:, -1]
    x = x.clone().detach().float()  # torch.tensor(x).float()
    t = t.clone().detach().float()  # torch.tensor(t).float()
    t = t.view(-1, 1)
    return x, t


def sepr_repr(x):
    mask_1, mask_0 = (x[:, -1] == 1), (x[:, -1] == 0)
    x_1 = x[mask_1]
    x_1 = x_1[:, :-1]
    x_0 = x[mask_0]
    x_0 = x_0[:, :-1]
    m = max(x_0.shape, x_1.shape)
    z0 = torch.zeros(m)
    m0 = x_0.shape[0]
    z0[:m0, ] = x_0
    z1 = torch.zeros(m)
    m1 = x_1.shape[0]
    z1[:m1, ] = x_1
    return z0, z1


def get_separ_data(x):

    mask_1 = x["tt"] == 1
    mask_0 = x["tt"] == 0
    x_1 = x[mask_1].drop(columns="tt")
    x_0 = x[mask_0].drop(columns="tt")
    return x_0, x_1


#### 


def quantile(t, S, p):
    index = np.argmin([np.abs(S[k]-p) for k in range(len(S))])
    return t[index]


def PlotResults(d, patient):
    I = d['I']
    S_0_pred = d['S_0_pred'][patient]
    S_1_pred = d['S_1_pred'][patient]
    S_0_true = d['S_0_true'][patient]
    S_1_true = d['S_1_true'][patient]

    CATE_true = S_1_true - S_0_true
    CATE_pred = S_1_pred - S_0_pred

    if len(CATE_true) != len(CATE_pred):
        print('Error : shape not equal')

    p_list = [0.1, 0.25, 0.4, 0.5, 0.75, 0.9]

    d_quantile = {}
    d_quantile['p'] = p_list
    d_quantile['t*_0 true'] = [quantile(I, S_0_true, p) for p in p_list]
    d_quantile['t*_0 pred'] = [quantile(I, S_0_pred, p) for p in p_list]
    d_quantile['dif_0'] = np.sqrt(np.mean(
        (np.array(d_quantile['t*_0 true'])-np.array(d_quantile['t*_0 pred']))**2))

    d_quantile['t*_1 true'] = [quantile(I, S_1_true, p) for p in p_list]
    d_quantile['t*_1 pred'] = [quantile(I, S_1_pred, p) for p in p_list]
    d_quantile['dif_1'] = np.sqrt(np.mean(
        (np.array(d_quantile['t*_1 true'])-np.array(d_quantile['t*_1 pred']))**2))

    fig1 = plt.figure(figsize=(18, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(I, S_0_true, label='S_true for T=0', marker='o',
             linestyle='dashed', markersize=1, color='b')
    ax1.plot(I, S_1_true, label='S_true for T=1', marker='o',
             linestyle='dashed', markersize=1, color='r')
    ax1.plot(I, S_0_pred, label='S_pred for T=0',
             color='b', drawstyle='steps-post')
    ax1.plot(I, S_1_pred, label='S_pred for T=1',
             color='r', drawstyle='steps-post')
    #ax1.fill_between(I, S_0_true, S_0_pred, color="grey")
    # ax1.fill_between(I, S_1_true, S_1_pred, color="#5bc0de")

    plt.legend()
    plt.close()

    fig2 = plt.figure(figsize=(18, 10))
    ax2 = fig2.add_subplot(111)
    ax2.plot(I, CATE_true, label='CATE_true', marker='o',
             linestyle='dashed', markersize=1, color='b')
    ax2.plot(I, CATE_pred, label='CATE_pred', color='r')
    plt.legend()
    plt.close()

    """pehe = np.exp(CATE_true-CATE_pred)
    s_pehe = simps(pehe, I).round(4) / max(I)
    ax2.plot(I, pehe , label=f'exp(CATE_true-CATE_pred) : S_PEHE = {s_pehe}', color='r')
    ax2.plot(I, [pehe.mean()]*len(I), label='Mean', marker='o',
               linestyle='dashed', markersize=1, color='b')
    plt.legend()
    plt.close()"""
    d_dtw0 = dtw_F(S_0_true, S_0_pred)
    d_dtw1 = dtw_F(S_1_true, S_1_pred)

    dtw = (d_dtw0, d_dtw1)

    #fig3 = plt.figure(figsize=(18, 10))
    #ax3 = fig3.add_subplot(111)

    #kdetrue = gaussian_kde(CATE_true)
    #kdepred = gaussian_kde(CATE_pred)

    """grid = np.linspace(-1.5, 1.5, 300)
    ax3.plot(grid, kdetrue(grid), label="CATE_true distribution")
    ax3.plot(grid, kdepred(grid), label="CATE_pred distribution")
    plt.legend()
    plt.close()"""

    return fig1, fig2, d_quantile, dtw  # fig3


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

    fig = plt.figure(figsize=(18, 10), dpi=100)
    ax = fig.add_subplot(111)
    bp = ax.boxplot(input, widths=0.2, sym='', patch_artist=True)
    plt.setp(bp['caps'], color='blue', alpha=1)
    plt.setp(bp['whiskers'], color='blue', alpha=1)
    plt.setp(bp['medians'], color='tomato', alpha=1, linewidth=4.0)
    plt.setp(bp['boxes'],
             facecolor='lightblue',
             alpha=1,
             color='blue',
             linewidth=0.5)
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    plt.xticks(np.arange(n) + 1, models_list, rotation=60)
    plt.close()
    return fig


def plots(patient, d_all, model_name):
    d = d_all[f'd_{model_name}']
    I = d['I']

    S_0_pred = d['S_0_pred'][patient]
    S_1_pred = d['S_1_pred'][patient]
    S_0_true = d['S_0_true'][patient]
    S_1_true = d['S_1_true'][patient]

    CATE_true = S_1_true - S_0_true
    CATE_pred = S_1_pred - S_0_pred

    d_ours = d_all['d_SurvCaus']

    S_0_pred_ours = d_ours['S_0_pred'][patient]
    S_1_pred_ours = d_ours['S_1_pred'][patient]

    CATE_pred_ours = S_1_pred_ours - S_0_pred_ours

    p_ours = np.argmin(np.array(d_ours['mise_0'])+np.array(d_ours['mise_1']))
    p_bench = np.argmin(np.array(d['mise_0'])+np.array(d['mise_1']))
    print("(p_ours,p_bench) =", (p_ours, p_bench))
    # Plot survie

    fig_surv = plt.figure(figsize=(18, 10))
    ax1 = fig_surv.add_subplot(111)
    ax1.plot(I, S_0_true, label='S_true for T=0',
             marker='o', markersize=1, color='b')
    ax1.plot(I, S_1_true, label='S_true for T=1',
             marker='o', markersize=1, color='r')
    ax1.plot(I, S_0_pred, label='S_pred for T=0',
             color='b', linestyle='dashed')
    ax1.plot(I, S_1_pred, label='S_pred for T=1',
             color='r', linestyle='dashed')
    ax1.plot(I, S_0_pred_ours, label='S_pred_ours for T=0',
             color='b', linestyle='-.', drawstyle='steps-post')
    ax1.plot(I, S_1_pred_ours, label='S_pred_ours for T=1',
             color='r', linestyle='-.', drawstyle='steps-post')

    plt.legend()
    plt.title(
        f"Mise (0,1) =  OURS : ({d_ours['mise_0'][patient].round(4)},{d_ours['mise_1'][patient].round(4)}) || {model_name} : ({d['mise_0'][patient].round(4)},{d['mise_1'][patient].round(4)}) ")

    # plot CATE
    fig_cate = plt.figure(figsize=(18, 10))
    ax2 = fig_cate.add_subplot(111)
    ax2.plot(I, CATE_true, label='CATE_true',
             marker='o', markersize=1, color='b')
    ax2.plot(I, CATE_pred, label='CATE_pred', color='r')
    ax2.plot(I, CATE_pred_ours, label='CATE_pred_ours', color='g')
    plt.legend()
    plt.title(
        f"Mise CATE =  OURS : {d_ours['mise_cate'][patient].round(4)} || {model_name} : {d['mise_cate'][patient].round(4)} ")

    return fig_surv, fig_cate
#####


def dtw_F(s1, s2):

    s1 = np.double(s1)
    s2 = np.double(s2)
    # np.double(dtw.distance_fast(s1, s2, use_pruning=True)).round(4)
    return np.sqrt(np.mean((s1 - s2)**2))


def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    a = a / np.mean(a)
    b = np.asarray(b, dtype=np.float)
    b = b / np.mean(b)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true)[1:], np.array(y_pred)[1:]
    return np.mean(np.where(y_true != 0.0, np.abs(1.0 - y_true / y_pred),
                            0.0)) * 100
    # return np.mean(np.abs(1 - (y_pred)/(y_true))) * 100


def H_Score(y_true, y_pred):
    h = np.log(y_pred[1:]) - np.log(y_true[1:])
    return np.mean(np.power(h, 2))


def mise(y_true, y_pred, I):
    #simps((y_true-y_pred)**2, I).round(6) / max(I)
    #surface_true = simps(np.abs(y_true), I).round(6) / max(I)
    #surface_pred = simps(np.abs(y_pred), I).round(6) / max(I)
    return np.sqrt(simps((y_true-y_pred)**2, I).round(6)) / max(I)


def interpolate(I, cuts, S_0_pred, S_1_pred):
    from scipy import interpolate
    tck_0 = interpolate.interp1d(cuts, S_0_pred, kind='linear')  # linear
    tck_1 = interpolate.interp1d(cuts, S_1_pred, kind='linear')
    S_0_pred_ = tck_0(I)
    S_1_pred_ = tck_1(I)
    return S_0_pred_, S_1_pred_


@jit(forceobj=True, nogil=True, boundscheck=False)
def piecewise(x, bins, values):
    bins = np.asarray(bins[:len(x)])
    values = np.asarray(values[:len(w)])

    """Computes the values at x of the piecewise constant function
    on the given `bins` with values in `values`

    Parameters
    ----------
    x : numpy.ndarray
        A numpy array of shape (n,) corresponding to the values at which we
        compute the piecewise function. Warning: the values in x must
        be SORTED in increasing order beforehand.

    bins : numpy.ndarray
        A numpy array of shape (n_bins,) corresponding to the bundaries of intervals
        on which the function is constant. Note that the first interval is
        (-inf, bins[0]] and the last is [bins[-1], inf). The value on (-inf, bins[0]] is 
        value[0], the value on [bins[-1], inf) is value[-1]

    values : numpy.ndarray
        A numpy array of shape (n_bins,) corresponding to the values of
        the function within each interval

    Returns
    -------
    output : numpy.ndarray
        A numpy array of shape (n,) containing the values of the piecewise function at x.
    """

    n_bins = bins.shape[0]
    n_values = values.shape[0]
    assert x.ndim == 1
    assert bins.ndim == 1
    assert values.ndim == 1
    assert n_values == n_bins

    values = np.append(values, values[-1])
    out = np.empty(x.shape, dtype=x.dtype)
    bin_idx = 0
    for i, x_i in enumerate(x):
        if bin_idx != n_bins - 1:
            while x_i > bins[bin_idx]:
                bin_idx += 1
        out[i] = values[bin_idx]

    return out



def search_beta(alpha_t,lamb,coeff_tt, y_min,t):
    log_u = np.log(np.random.uniform(0, 1, 25))
    g_t = np.log(-log_u) - alpha_t * np.log(lamb*y_min) - coeff_tt *t 
    return g_t

 
def BART():
    return ConditionalSurvivalForestModel()
   
    
    
    
