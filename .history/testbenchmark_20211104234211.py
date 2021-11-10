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
    