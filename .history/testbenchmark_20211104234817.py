from utils import *

from losses import *



from numba import jit

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
import pandas as pd
import seaborn as sns
import torch  # For building the networks
import torch.nn as nn
import torch.nn.functional as F

from numpy.random import binomial, multivariate_normal

from scipy.integrate import simps
from scipy.linalg.special_matrices import toeplitz
from scipy.stats import gaussian_kde
from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

# search hyperparameter
import optuna

scaler = preprocessing.MinMaxScaler()


args_cuda = torch.cuda.is_available()

"""
Simulation parameters
"""
p_sim = {}
p_sim['n_samples'] = 3000
p_sim['n_features'] = 25
p_sim['beta'] = [0.01 * (p_sim['n_features'] - i) / p_sim['n_features']
                 for i in range(0, p_sim['n_features'])]
p_sim['alpha'] = 3
p_sim['lamb'] = 1.
p_sim['coef_tt'] = 1.8
p_sim['rho'] = 0.
p_sim['kappa'] = 2.
p_sim['wd_param'] = 20.
p_sim['scheme'] = 'linear'
p_sim['path_data'] = "./sim_surv"

"""
Model parameters
"""

p_survcaus = {}
p_survcaus['num_durations'] = 25
p_survcaus['encoded_features'] = 20
p_survcaus['alpha_wass'] = 1
p_survcaus['batch_size'] = 256
p_survcaus['epochs'] = 100
p_survcaus['lr'] = 1e-2
p_survcaus['patience'] = 10


"""
Model architecture
"""


class NetCFRSurv(nn.Module):

    def __init__(self, in_features, encoded_features, out_features, alpha=1):
        super().__init__()
        self.psi = nn.Sequential(
            nn.Linear(in_features-1, 32),  nn.LeakyReLU(),
            nn.Linear(32, 32),  nn.ReLU(),
            nn.Linear(32, 28),  nn.LeakyReLU(),
            nn.Linear(28, encoded_features),
        )

        self.surv_net = nn.Sequential(
            nn.Linear(encoded_features + 1, 128), nn.LeakyReLU(),
            nn.Linear(128, 128),  nn.ReLU(),

            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, 128),
            nn.LeakyReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.1),

            nn.Linear(128, 50),  nn.ReLU(),
            nn.Linear(50, out_features),
        )
        self.alpha = alpha
        self.loss_surv = NewLoss()  # NLLPMFLoss()   ## NLLLogistiHazardLoss()
        self.loss_wass = WassLoss()  # IPM

    def forward(self, input):
        x, t = get_data(input)
        psi = self.psi(x)
        # psi_inv = self.psi_inv(psi)
        psi_t = torch.cat((psi, t), 1)
        phi = self.surv_net(psi_t)
        return phi, psi_t

    def get_repr(self, input):
        x, t = get_data(input)
        return torch.cat((self.psi(x), t), 1)

    def predict(self, input):

        psi_t = self.get_repr(input)
        return self.surv_net(psi_t)



class Evaluation():
    def __init__(self, params_simu, params_survcaus):
        self.params_simu = params_simu
        self.params_survcaus = params_survcaus
        self.SC = SurvCaus(self.params_simu, self.params_survcaus)
        self.path = params_simu['path_data']
        self.data = DataLoader(
            self.params_simu, self.params_survcaus).get_data()
        self.cuts = self.SC.cuts
        self.t_max = min(max(self.data.df_train['T_f_cens']), max(self.cuts))
        self.N = 1500
        self.I = np.linspace(0, self.t_max, self.N)

    def S(self, tt_p, xbeta_p, t):
        c_tt = self.params_simu['coef_tt'] * tt_p

        if self.params_simu['scheme'] == 'linear':
            return np.exp(-(self.params_simu['lamb'] * t) ** self.params_simu['alpha'] * np.exp(xbeta_p + c_tt))
        else:
            sh_z = xbeta_p + np.cos(xbeta_p + c_tt) + \
                np.abs(xbeta_p - c_tt) + c_tt
            return np.exp(-(self.params_simu['lamb'] * t) ** self.params_simu['alpha'] * np.exp(sh_z))

   

    def Results_SurvCaus(self, model, is_train):
        if is_train:
            x_ = self.data.df_train.iloc[:, :-2]
        else:
            x_ = self.data.df_test.iloc[:, :-2]

        cl_patient_list = x_.iloc[:, :-1].dot(self.params_simu['beta']).values
        S_0_pred_mat = model.S_pred(0, x_.values)
        S_1_pred_mat = model.S_pred(1, x_.values)

        PEHE_list = []
        mise_0_list = []
        mise_1_list = []
        mise_cate_list = []

        S_0_true_list = []
        S_1_true_list = []
        S_0_pred_list = []
        S_1_pred_list = []
        d = {}

        for patient in tqdm(range(x_.shape[0])):

            xbeta = cl_patient_list[patient]

            S_0_pred = S_0_pred_mat[patient].values.squeeze()
            S_1_pred = S_1_pred_mat[patient].values.squeeze()

            # S_0_pred_, S_1_pred_ = piecewise(I, S_0_pred, cuts), piecewise(I, S_1_pred, cuts)
            S_0_pred_, S_1_pred_ = interpolate(
                self.I, self.cuts, S_0_pred, S_1_pred)

            S_0_true_, S_1_true_ = self.S(
                0, xbeta, self.I), self.S(1, xbeta, self.I)

            S_0_true_list.append(S_0_true_)
            S_1_true_list.append(S_1_true_)
            S_0_pred_list.append(S_0_pred_)
            S_1_pred_list.append(S_1_pred_)

            CATE_true = (S_1_true_ - S_0_true_)
            CATE_pred = (S_1_pred_ - S_0_pred_)

            PEHE_list.append((CATE_true-CATE_pred)**2)

            mise_0 = mise(S_0_true_, S_0_pred_, self.I)
            mise_1 = mise(S_1_true_, S_1_pred_, self.I)

            mise_cate = mise(CATE_true, CATE_pred, self.I)

            mise_0_list.append(mise_0)
            mise_1_list.append(mise_1)
            mise_cate_list.append(mise_cate)

        d['I'] = self.I
        d['cuts'] = self.cuts.tolist()
        d['mise_0'] = mise_0_list
        d['mise_1'] = mise_1_list
        d['mise_cate'] = mise_cate_list
        d['sqrt PEHE'] = np.sqrt(np.mean(PEHE_list, axis=0))
        d['S_0_pred'] = S_0_pred_list
        d['S_1_pred'] = S_1_pred_list
        d['S_0_true'] = S_0_true_list
        d['S_1_true'] = S_1_true_list

        return d

    def Results_Benchmark(self, model0, model1, is_train):
        if is_train:
            x_ = self.data.df_train.iloc[:, :-3]
        else:
            x_ = self.data.df_test.iloc[:, :-3]

        d = {}
        PEHE_list = []
        mise_0_list = []
        mise_1_list = []
        mise_cate_list = []

        S_0_true_list = []
        S_1_true_list = []
        S_0_pred_list = []
        S_1_pred_list = []

        cl_patient_list = x_.dot(self.params_simu['beta']).values

        S_0_pred_mat = model0.predict_survival(x_)
        S_1_pred_mat = model1.predict_survival(x_)

        for patient in tqdm(range(x_.shape[0])):

            cl_patient = cl_patient_list[patient]
            S_0_pred = np.asarray(S_0_pred_mat[patient, :].flatten())
            S_1_pred = np.asarray(S_1_pred_mat[patient, :].flatten())

            #S_0_pred_ = self.piecewise_c(self.I, model0.times, S_0_pred)
            #S_1_pred_ = self.piecewise_c(self.I, model1.times, S_1_pred)
            S_0_pred_ = piecewise(self.I, model0.times, S_0_pred)
            S_1_pred_ = piecewise(self.I, model1.times, S_1_pred)
            S_0_true_, S_1_true_ = self.S(
                0,  cl_patient, self.I), self.S(1, cl_patient, self.I)

            S_0_true_list.append(S_0_true_)
            S_1_true_list.append(S_1_true_)
            S_0_pred_list.append(S_0_pred_)
            S_1_pred_list.append(S_1_pred_)

            CATE_true = S_1_true_ - S_0_true_
            CATE_pred = S_1_pred_ - S_0_pred_
            PEHE_list.append((CATE_true-CATE_pred)**2)

            mise_0 = mise(S_0_true_, S_0_pred_, self.I)
            mise_1 = mise(S_1_true_, S_1_pred_, self.I)

            mise_cate = mise(CATE_true, CATE_pred, self.I)

            mise_0_list.append(mise_0)
            mise_1_list.append(mise_1)
            mise_cate_list.append(mise_cate)

        d['I'] = self.I.tolist()
        d['cuts'] = self.cuts.tolist()
        d['mise_0'] = mise_0_list
        d['mise_1'] = mise_1_list
        d['mise_cate'] = mise_cate_list
        d['sqrt PEHE'] = np.sqrt(np.mean(PEHE_list, axis=0))
        d['S_0_pred'] = S_0_pred_list
        d['S_1_pred'] = S_1_pred_list
        d['S_0_true'] = S_0_true_list
        d['S_1_true'] = S_1_true_list

        return d

    def All_Results(self, list_models=["SurvCaus", "SurvCaus_0"], is_train=True):
        self.list_models = list_models
        d_list_models = {}
        cate_dict = {}
        surv0_dict = {}
        surv1_dict = {}
        pehe_dict = {}
        bilan = {}
        bilan['models'] = list_models
        bilan['Mise0'] = []
        bilan['Mise1'] = []
        bilan['CATE'] = []
        bilan['PEHE'] = []

        for model_name in list_models:
            print(model_name)
            if model_name == "SurvCaus":
                print(self.params_survcaus)
                SC = SurvCaus(self.params_simu, self.params_survcaus)
                SC.fit_model()
                d_list_models[f'd_{model_name}'] = self.Results_SurvCaus(
                    SC, is_train)

            if model_name == "SurvCaus_0":
                params_survcaus_0 = self.params_survcaus.copy()
                params_survcaus_0['alpha_wass'] = 0.
                print(params_survcaus_0)
                SC0 = SurvCaus(self.params_simu, params_survcaus_0)
                SC0.fit_model()
                d_list_models[f'd_{model_name}'] = self.Results_SurvCaus(
                    SC0, is_train)

            if model_name == 'BART':
                model0, model1 = BART(), BART()
                model0.fit(self.data.x_0_train,
                           self.data.T_f_0_train, self.data.e_0_train)
                model1.fit(self.data.x_1_train,
                           self.data.T_f_1_train, self.data.e_1_train)
                d_list_models[f'd_{model_name}'] = self.Results_Benchmark(
                    model0, model1, is_train)

            if model_name == 'CoxPH':
                model0, model1 = CoxPHModel(), CoxPHModel()
                model0.fit(self.data.x_0_train, self.data.T_f_0_train,
                           self.data.e_0_train, max_iter=2000, lr=0.1, tol=10e-3, verbose=True)
                model1.fit(self.data.x_1_train, self.data.T_f_1_train,
                           self.data.e_1_train, max_iter=2000, lr=0.1, tol=10e-3, verbose=True)
                d_list_models[f'd_{model_name}'] = self.Results_Benchmark(
                    model0, model1, is_train)

            cate_dict[f'{model_name}'] = d_list_models[f'd_{model_name}']['mise_cate']
            surv0_dict[f'{model_name}'] = d_list_models[f'd_{model_name}']['mise_0']
            surv1_dict[f'{model_name}'] = d_list_models[f'd_{model_name}']['mise_1']
            pehe_dict[f'{model_name}'] = d_list_models[f'd_{model_name}']['sqrt PEHE']

            bilan['CATE'].append((np.mean(cate_dict[f'{model_name}']).round(
                3), np.std(cate_dict[f'{model_name}']).round(3)))
            bilan['PEHE'].append((np.mean(pehe_dict[f'{model_name}']).round(
                3), np.std(pehe_dict[f'{model_name}']).round(3)))
            bilan['Mise0'].append((np.mean(surv0_dict[f'{model_name}']).round(
                3), np.std(surv0_dict[f'{model_name}']).round(3)))
            bilan['Mise1'].append((np.mean(surv1_dict[f'{model_name}']).round(
                3), np.std(surv1_dict[f'{model_name}']).round(3)))

        self.bilan_benchmark = pd.DataFrame(bilan)
        self.bilan_json = bilan
        self.d_list_models = d_list_models

        def box_plot(l, title):
            fig, ax = plt.subplots()
            ax.boxplot(l.values())
            ax.set_xticklabels(l.keys())
            plt.title(str(title))
            return fig

        self.box_plot_cate = box_plot(cate_dict, "MISE des CATE")
        self.box_plot_surv0 = box_plot(surv0_dict, "MISE de Surv0")
        self.box_plot_surv1 = box_plot(surv1_dict, "MISE de Surv1")
        self.box_plot_pehe = box_plot(pehe_dict, "sqrt PEHE")

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

        p_ours = np.argmin(
            np.array(d_ours['mise_0'])+np.array(d_ours['mise_1']))
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

    def represent_patient(self, patient):
        for model_name in self.list_models[1:]:
            fig_surv, fig_cate = plots(
                patient, self.d_list_models, model_name=model_name)

    def