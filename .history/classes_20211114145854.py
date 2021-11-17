from scipy.stats.stats import mode
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


""" DataLoader class """


class DataLoader():
    def __init__(self, params_sim, params_survcaus):
        super().__init__()
        self.path = params_sim['path_data']
        self.pmf = True
        self.scheme_subd = 'quantiles'  # "equidistant"
        self.num_durations = params_survcaus['num_durations']

    def load_data_sim_benchmark(self):

        df = pd.read_csv(self.path + ".csv")
        dim = df.shape[1]-8

        x_z_list = ["X" + str(i) for i in range(1, dim + 1)] + ["tt"]
        leave = x_z_list + ["event", "T_f_cens"]

        ##
        rs = ShuffleSplit(test_size=.4, random_state=0)
        df_ = df[leave].copy()


        for train_index, test_index in rs.split(df_):
            df_train = df_.drop(test_index)
            df_test = df_.drop(train_index)
            df_val = df_test.sample(frac=0.5)
            df_test = df_test.drop(df_val.index)

        if self.pmf:
            labtrans = PMF.label_transform(
                self.num_durations, scheme=self.scheme_subd)

        def get_target(df):
            return (df["T_f_cens"].values, df["event"].values)

        y_train_surv = labtrans.fit_transform(*get_target(df_train))
        y_val_surv = labtrans.transform(*get_target(df_val))

        train = (df_train[x_z_list].values.astype("float32"), y_train_surv)
        val = (
            df_val[x_z_list].values.astype("float32"),
            y_val_surv,
        )
        # We don't need to transform the test labels
        durations_test, events_test = get_target(df_test)
        x_test = df_test[x_z_list].values.astype("float32")

        # SPlit data for OURS
        self.x_train, self.y_train, self.train, self.val,\
            self.durations_test, self.events_test, self.labtrans, self.x_test = train[
                0], train[1], train, val, durations_test, events_test, labtrans, x_test

        #  SPlit data for benchmarking

        def get_separ_data(x):
            mask_1 = x["tt"] == 1
            mask_0 = x["tt"] == 0
            x_1 = x[mask_1].drop(columns="tt")
            x_0 = x[mask_0].drop(columns="tt")
            return x_0, x_1

        df_train_0,  df_train_1 = get_separ_data(df_train)
        df_test_0, df_test_1 = get_separ_data(df_test)

        self.x_0_train = df_train_0.iloc[:, :-2].values
        self.e_0_train = df_train_0.iloc[:, -2].values

        self.x_1_train = df_train_1.iloc[:, :-2].values
        self.e_1_train = df_train_1.iloc[:, -2].values

        self.T_f_0_train = df_train_0.iloc[:, -1].values
        self.T_f_1_train = df_train_1.iloc[:, -1].values

        self.x_0_test = df_train_0.iloc[:, :-2].values
        self.e_0_test = df_train_0.iloc[:, -2].values

        self.x_1_test = df_test_1.iloc[:, :-2].values
        self.e_1_test = df_test_1.iloc[:, -2].values

        self.T_f_0_test = df_test_0.iloc[:, -1].values
        self.T_f_1_test = df_test_1.iloc[:, -1].values

        self.df_train = df_train
        self.df_test = df_test
        self.df_val = df_val

    def get_data(self):
        self.load_data_sim_benchmark()
        return self


""" Sensitivity class """
# sensitivity_analysis of X : selection of features with the highest sensitivity

class Sensitivity():
    def __init__(self, params_sim, params_survcaus):
        super().__init__()
        self.params_sim = params_sim
        self.params_survcaus = params_survcaus
        self.data = DataLoader(params_sim, params_survcaus).get_data()

    def sensitivity_analysis(self):
        # load data
        x_train, y_train, x_val, y_val, x_test, durations_test, events_test, labtrans, x_test = self.data.x_train, self.data.y_train, self.data.x_val, self.data.y_val, self.data.x_test, self.data.durations_test, self.data.events_test, self.data.labtrans, self.data.x_test

        # define the model
        model = SurvivalModel(self.params_sim, self.params_survcaus).get_model()
        model.fit(x_train, y_train, x_val, y_val)
        S_pred_train = model.predict(x_train)

        
        
        delta = 0.01
        # modifie x : x_i <- x_i - delta
        x_train_ = x_train.copy()
        x_val_ = x_val.copy()
        x_test_ = x_test.copy()
        sens_train = []
        for i in range(x_train_.shape[1]):
            x_train_[:, i] = x_train_[:, i] - delta
            x_val_[:, i] = x_val_[:, i] - delta
            x_test_[:, i] = x_test_[:, i] - delta
            model.fit(x_train_, y_train, x_val_, y_val)
            S_pred_train_ = model.predict(x_train_)
           
            # compute the sensitivity
            sens_train.append(mise(S_pred_train, S_pred_train_))
        
        return sens_train
    



""" SurvCaus class """


class SurvCaus(nn.Module):

    def __init__(self, params_sim, params_survcaus):
        super().__init__()

        num_durations = params_survcaus['num_durations']
        encoded_features = params_survcaus['encoded_features']
        alpha_surv = 1.
        alpha_wass = params_survcaus['alpha_wass']
        batch_size = params_survcaus['batch_size']
        epochs = params_survcaus['epochs']
        lr = params_survcaus['lr']
        is_tcga = False
        path_data = params_sim['path_data']
        patience = params_survcaus['patience']

        self.num_durations = num_durations
        if is_tcga:
            self.data.x_train, self.data.y_train, self.train, self.val, self.durations_test, self.events_test, self.labtrans, self.data.x_test = load_data_sim_hd(path=path_data,
                                                                                                                                                                  num_durations=self.num_durations, pmf=True, dim=self.dim)
        else:
            self.data = DataLoader(
                params_sim, params_survcaus).get_data()

        self.in_features = self.data.x_train.shape[1]
        self.encoded_features = encoded_features
        self.out_features = self.data.labtrans.out_features
        self.cuts = self.data.labtrans.cuts
        self.net = NetCFRSurv(
            self.in_features, self.encoded_features, self.out_features)

        self.repr_weights = self.net.psi[6].weight
        self.surv_weights = self.net.surv_net[12].weight

        self.alpha_wass = alpha_wass
        self.alpha_surv = alpha_surv
        self.loss = Loss(self.alpha_wass, self.alpha_surv)
        self.lr = lr
        self.model = PMF(self.net, tt.optim.Adam(self.lr),
                         duration_index=self.data.labtrans.cuts, loss=self.loss)
        self.batch_size = batch_size
        self.epochs = epochs

        if args_cuda:
            self.net.cuda()
            self.loss.cuda()
            self.model.cuda()
        self.metrics = dict(loss_surv=Loss(0, 1), loss_wass=Loss(1, 0))
        self.callbacks = [tt.cb.EarlyStopping(patience=patience)]

    def fit_model(self, is_train=True):
        """Fits the model on the device .

        Args:
            is_train (bool, optional): [description]. Defaults to True.
        """
        """if args_cuda:
            self.data.x_train, self.data.y_train, self.data.val = self.data.x_train.cuda(
            ), self.data.y_train.cuda(), self.data.val.cuda()"""
        log = self.model.fit(self.data.x_train, self.data.y_train, self.batch_size,
                             self.epochs, callbacks=self.callbacks, metrics=self.metrics, val_data=self.data.val)
        self.log = log
        # print('Saving model ...')
        # pickle.dump(self.net, open(filename, 'wb'))
    # sensitivity analysis of X
    def sensitivity_analysis(self, X, y, n_samples=100, n_features=10):
        """Calculates the sensitivity of the model.

        Args:
            X (np.array): [description].
            y (np.array): [description].
            n_samples (int, optional): [description]. Defaults to 100.
            n_features (int, optional): [description]. Defaults to 10.

        Returns:
            [type]: [description]
        """
        # initialize the sensitivity object
        sensitivity = Sensitivity(X, y, n_samples, n_features)

        # get the indices of the features to be removed
        indices = sensitivity.get_indices()

        # get the features to be removed
        features = sensitivity.get_features(indices)

        # calculate the sensitivity of the model
        sensitivity_score = sensitivity.calculate_sensitivity(
            self.net, self.data.x_train, self.data.y_train, self.data.val, self.batch_size, self.epochs, self.callbacks, self.metrics, self.loss)

        # return the results
        return sensitivity_score, features
    def res_model(self):

        res = self.model.log.to_pandas()
        self.res = res
        p1 = res[['train_loss', 'val_loss']].plot(figsize=(16, 10))
        p2 = res[['train_loss_surv', 'val_loss_surv']].plot(figsize=(16, 10))
        p3 = res[['train_loss_wass', 'val_loss_wass']].plot(figsize=(16, 10))
        return res  # ,p1,p2,p3

    def wass_init(self):

        x1, x0 = sepr_repr(torch.tensor(self.data.x_train).float())
        return SinkhornDistance(eps=0.001, max_iter=100, reduction=None)(x1, x0)

    def S_pred(self, tt, x):
        x_c = x.copy()
        ones = np.ones_like(x[:, -1])
        x_c[:, -1] = tt * ones + (1-tt)*(1-ones)
        surv = self.model.predict_surv_df(x_c)
        return surv


""" Evaluation class """


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
        self.N = 500
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
        fsm_dict = {}
        bilan = {}
        bilan['models'] = list_models
        bilan['Mise0'] = []
        bilan['Mise1'] = []
        bilan['CATE'] = []
        bilan['PEHE'] = []
        bilan['FSM'] = []
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
            fsm_dict[f'{model_name}'] = (np.array(surv0_dict[f'{model_name}'].copy())+np.array(surv1_dict[f'{model_name}'].copy()))**2
            
            bilan['CATE'].append((np.mean(cate_dict[f'{model_name}']).round(
                3), np.std(cate_dict[f'{model_name}']).round(3)))
            bilan['PEHE'].append((np.mean(pehe_dict[f'{model_name}']).round(
                3), np.std(pehe_dict[f'{model_name}']).round(3)))
            bilan['Mise0'].append((np.mean(surv0_dict[f'{model_name}']).round(
                3), np.std(surv0_dict[f'{model_name}']).round(3)))
            bilan['Mise1'].append((np.mean(surv1_dict[f'{model_name}']).round(
                3), np.std(surv1_dict[f'{model_name}']).round(3)))
            bilan['FSM'].append((np.mean(fsm_dict[f'{model_name}']).round( 
                3), np.std(fsm_dict[f'{model_name}']).round(3)))

      
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
        self.box_plot_FSM = box_plot(fsm_dict, "FSM")

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


""" Tunning hyperparameter class """


class Tunning():
    def __init__(self, params_simu):
        super().__init__()
        self.params_simu = params_simu

    def objective_survcaus(self, trial):

        self.params_search = {'num_durations':  trial.suggest_int('num_durations', 20, 30),
                              'encoded_features': trial.suggest_int('encoded_features', 10, 100),
                              'alpha_wass': trial.suggest_uniform('alpha_wass', 0, 10),
                              #'batch_size': trial.suggest_int('batch_size', 64, 256),
                              # 'epochs': 20,  #trial.suggest_int('epochs', 50, 200),
                              'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
                              # 'patience': 4  #trial.suggest_int('patience', 2, 10)
                              }
        self.params_search['epochs'] = 20
        self.params_search['patience'] = 4
        self.params_search['batch_size'] = 256
        Eval = Evaluation(self.params_simu, self.params_search)
        SC = SurvCaus(self.params_simu, self.params_search)
        SC.fit_model()
        #d_train = Eval.Results_SurvCaus(SC, is_train=True)
        d_val = Eval.Results_SurvCaus(SC, is_train=False)
        #MMise_train = (np.mean(d_train["mise_0"])+np.mean(d_train["mise_1"]))/2
        MMise_val = (np.mean(d_val["mise_0"])+np.mean(d_val["mise_1"]))/2

        # MMise_cate_train = np.mean(d_train["mise_cate"]).round(2)
        # MMise_cate_val = np.mean(d_val["mise_cate"]).round(2)
        return MMise_val  # + MMise_train

    def objective_bart(self, trial):
        
        self.params_search = {'num_trees':  trial.suggest_int('num_durations', 10, 30),
                              'encoded_features': trial.suggest_int('encoded_features', 10, 100),
                              'alpha_wass': trial.suggest_uniform('alpha_wass', 0, 10),
                              #'batch_size': trial.suggest_int('batch_size', 64, 256),
                              # 'epochs': 20,  #trial.suggest_int('epochs', 50, 200),
                              'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
                              # 'patience': 4  #trial.suggest_int('patience', 2, 10)
                              }
        self.params_search['epochs'] = 20
        self.params_search['patience'] = 4
        self.params_search['batch_size'] = 256
        Eval = Evaluation(self.params_simu, self.params_search)
        SC = SurvCaus(self.params_simu, self.params_search)
        SC.fit_model()
        #d_train = Eval.Results_SurvCaus(SC, is_train=True)
        d_val = Eval.Results_SurvCaus(SC, is_train=False)
        #MMise_train = (np.mean(d_train["mise_0"])+np.mean(d_train["mise_1"]))/2self.params_search = {'num_durations':  trial.suggest_int('num_durations', 20, 30),
                            'encoded_features': trial.suggest_int('encoded_features', 10, 100),
                            'alpha_wass': trial.suggest_uniform('alpha_wass', 0, 10),
                            #'batch_size': trial.suggest_int('batch_size', 64, 256),
                            # 'epochs': 20,  #trial.suggest_int('epochs', 50, 200),
                            'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
                            # 'patience': 4  #trial.suggest_int('patience', 2, 10)
                            }
        
    def get_best_hyperparameter_survcaus(self, n_trials=10):
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective_survcaus, n_trials)
        self.best_trial_values = self.study.best_trial.values
        self.best_trial_params = self.study.best_trial.params
        return self.best_trial_params

    

class Simulation:

    def __init__(self, params_simu):

        n_samples = params_simu['n_samples']
        n_features = params_simu['n_features']
        beta = params_simu['beta']
        alpha = params_simu['alpha']
        lamb = params_simu['lamb']
        kappa_cens = params_simu['kappa']
        coef_tt = params_simu['coef_tt']
        rho = params_simu['rho']
        scheme = params_simu['scheme']
        wd_para = params_simu['wd_param']

        """[summary]
        Simulation des données de survie causales. Nous controlons la distance de wasserstein entre les groupes traité/non traité
        et le taux de censure.

        Args:
            n_samples ([integer]): [data size]
            n_features ([integer]): [number of columns/features]
            beta ([float]): [c'est le paramètre d'influence linéraire des colonnes]
            alpha ([float]): [description]
            lamb ([float]): [description]
            kappa_cens ([float]): [description]
            rho (float, optional): [description]. Defaults to 0.0.
            scheme (str, optional): [description]. Defaults to 'linear'.
            wd_para (int, optional): [description]. Defaults to 5.
        """

        self.n_features = n_features
        """if len(beta) != n_features:
            print("Error: n_features != len(beta) ")"""

        # Simulation of baseline covariates
        cov = toeplitz(rho ** np.arange(0, self.n_features))
        
        # multivariate normal
        self.X = multivariate_normal(np.zeros(self.n_features), cov, size=n_samples)
        # uniformal distribution
        #self.X = np.random.uniform(0, 1, size=(n_samples, n_features))
        
        self.n_samples = n_samples
        self.beta = beta
        self.alpha = alpha
        self.lamb = lamb
        self.kappa_cens = kappa_cens
        self.coef_tt = coef_tt  # coef of treatement
        self.scheme = scheme
        self.wd_para = wd_para
        self.Xbeta = self.X.dot(self.beta)
        self.path_data = params_simu["path_data"]

    # True Survival function

    def S(self, xbeta_p, tt_p, t):  # Agathe : pourquoi X apparait et pas Xbeta ?
        c_tt = self.coef_tt * tt_p  # coef_tt : coef de traitement

        if self.scheme == 'linear':
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(xbeta_p + c_tt))
        else:
            sh_z = xbeta_p + np.cos(xbeta_p + c_tt) + \
                np.abs(xbeta_p - c_tt) + c_tt
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(sh_z))

    def S_1(self, xbeta_p, t):
        return self.S(xbeta_p, 1, t)

    def S_0(self, xbeta_p, t):
        return self.S(xbeta_p, 0, t)

    # Simulation of times to event according to a Weibull-Cox model
    def simulation_T(self, tt):
        c_tt = self.coef_tt * tt
        log_u = np.log(np.random.uniform(0, 1, self.n_samples))

        if self.scheme == 'linear':  # schema linéraire ens X
            log_T = 1.0 / self.alpha * \
                (np.log(-log_u) - (self.Xbeta + c_tt))
        else:  # schema non linéraire en X
            log_T = 1.0 / self.alpha * (np.log(-log_u) - (self.Xbeta + np.cos(self.Xbeta + c_tt)
                                                          - np.abs(self.Xbeta - c_tt)
                                                          - c_tt))

        return np.exp(log_T) / self.lamb

    # Simulation of times to event according an exponential distribution
    def simulation_C(self, lamb_c):
        log_u = np.log(np.random.uniform(0, 1, self.n_samples))
        return -log_u/ lamb_c

    def simulation_surv(self):
        # treatment simulation
        idx = np.arange(self.n_features)
        params_tt = (-1) ** idx * np.exp(-idx / 10.)

        p_tt = sigmoid(self.X.dot(params_tt))
        tt = binomial(1, p_tt)  # treatment

        for j in range(self.n_features):
            self.X[:, j] -= self.wd_para/2 * tt
            self.X[:, j] += self.wd_para/2 * (1-tt)

        # simulation of factual and counterfactual times to event
        T_f = self.simulation_T(tt)
        T_cf = self.simulation_T(1-tt)
        mean_T_f = np.mean(T_f)

        # simulation of censoring
        C = self.simulation_C(1/(self.kappa_cens * mean_T_f))

        # definition of observations
        T_f_cens = np.minimum(T_f, C)
        event = (T_f <= C) * 1

        # definition of T1 and T0 (it matches the factual and counterfactual times to event)
        T_1 = T_f*tt + T_cf * (1-tt)
        T_0 = T_f*(1-tt) + T_cf * tt

        # transform treatment specific covariates in tensor and compute the Wasserstein distance
        mask_1, mask_0 = (tt == 1), (tt == 0)
        X_tesnor = torch.tensor(self.X).float()
        x_1 = X_tesnor[mask_1]
        x_0 = X_tesnor[mask_0]
        m = max(x_0.shape, x_1.shape)
        z0 = torch.zeros(m)
        m0 = x_0.shape[0]
        z0[:m0, ] = x_0
        z1 = torch.zeros(m)
        m1 = x_1.shape[0]
        z1[:m1, ] = x_1
        wd = SinkhornDistance(eps=0.001, max_iter=100,
                              reduction=None)(z0, z1).item()

        # data_frame construction
        colmns = ["X" + str(j) for j in range(1, self.n_features + 1)]
        data_sim = pd.DataFrame(data=self.X, columns=colmns)
        
        # scaling 
        #data_sim = pd.DataFrame(scaler.fit_transform(data_sim),columns=colmns)
        
        data_sim["tt"] = tt
        data_sim["T_f_cens"] = T_f_cens
        data_sim["event"] = event
        data_sim["T_1"] = T_1
        data_sim["T_0"] = T_0

        data_sim["T_f"] = T_f
        data_sim["T_cf"] = T_cf

        data_sim["Xbeta"] = self.Xbeta
        self.data_sim = data_sim

        # observed censoring, treatment proportions
        self.perc_treatement = int((data_sim["tt"].mean() * 100))
        self.perc_event = int(data_sim["event"].mean() * 100)
        # Wasserstein distances
        print("WD = ", wd)
        print(f"tt = 1 : {self.perc_treatement} % ")
        print(f"event = 1 : {self.perc_event} %")
        self.wd = wd
        data_sim.to_csv(self.path_data + ".csv", index=False, header=True)
        return data_sim

    # Distribution plots for tt=0/1

    def plot_dist(self):

        colmns = ["X" + str(j) for j in range(1, self.n_features + 1)] + ["tt"]
        return sns.pairplot(self.data_sim[colmns], hue="tt", diag_kind="hist", height=6)

    # True S_O and S_1 for a patient
    def plot_surv_true(self, patient=1):

        t_max = max(self.data_sim["T_f"])
        times = np.linspace(0, t_max, 100)
        """
        indx_1 = self.S_1(0, times) >= 0.01
        indx_0 = self.S_0(0, times) >= 0.01
        idx_min = indx_1+indx_0
        times = times[idx_min]"""

        xbeta_p = self.Xbeta[patient]

        T_0_p = self.data_sim["T_0"].values[patient].round(2)
        T_1_p = self.data_sim["T_1"].values[patient].round(2)
        tt_p = self.data_sim["tt"].values[patient]
        self.S_0_true = self.S_0(xbeta_p, times)
        self.S_1_true = self.S_1(xbeta_p, times)

        fig = plt.figure(figsize=(18, 10))
        ax = fig.add_subplot(111)
        ax.plot(
            times,
            self.S_0_true,
            label="S_true for tt=0",
            marker="o",
            markersize=1,
            color="b",
        )
        ax.plot(
            times,
            self.S_1_true,
            label="S_true for tt=1",
            marker="o",
            markersize=1,
            color="r",
        )
        ax.vlines(
            x=T_0_p,
            ymin=0,
            ymax=1,
            colors="b",
            linestyle="dashed",
            label="T_0={} ".format(T_0_p),
        )
        ax.vlines(
            x=T_1_p,
            ymin=0,
            ymax=1,
            colors="r",
            linestyle="dashed",
            label="T_1={} ".format(T_1_p),
        )
        plt.title(
            "For patient={} and treatement tt={}. Event (=1) = {} % & Treatement (=1) ={} %".format(
                patient, tt_p, self.perc_event, self.perc_treatement
            )
        )
        plt.legend()
        plt.close()
        return fig


"""
tunning = Tunning(p_sim)
tunning.get_best_hyperparameter()
"""

"""# Simulation of data
simu = Simulation(p_sim)
simu.simulation_surv()

# Evaluation - benchmark
Ev = Evaluation(p_sim, p_survcaus)
Ev.All_Results(list_models=["SurvCaus", "SurvCaus_0"],
               is_train=True)      # , 'CoxPH', 'BART'


# Bilan of benchmark
print(Ev.bilan_benchmark)

Ev.box_plot_cate.show()"""

"""simu = Simulation(p_sim)
simu.simulation_surv()"""

"""
tunning = Tunning(p_sim)

p_survcaus_best = tunning.get_best_hyperparameter(n_trials=1)
p_survcaus_best

# Evaluation - benchmark

p_survcaus = p_survcaus_best.copy()

p_survcaus['patience'] = 4
p_survcaus['epochs']=20



Ev = Evaluation(p_sim, p_survcaus)
Ev.All_Results(list_models=["SurvCaus", "SurvCaus_0", 'CoxPH','BART'],
               is_train=True)      # , 'CoxPH', 'BART'


# Bilan of benchmark
print(Ev.bilan_benchmark)

# 600/600 [00:03<00:00, 179.35it/s]
"""
