from utils import *

from datetime import datetime
import pickle
from typing import Tuple
import streamlit as st

# from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr  

import pandas as pd
import seaborn as sns
import torch  # For building the networks
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt  # Some useful functions
from matplotlib.pyplot import figure
from pycox.models import PMF , utils
from scipy.linalg import toeplitz
from scipy.stats import bernoulli, multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
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


sns.set(color_codes=True)

# For preprocessing

import matplotlib as mpl

import os

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
"""from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.tree import SurvivalTree
from sksurv.util import Surv

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis"""
from pysurvival.models.survival_forest import ExtraSurvivalTreesModel
from pysurvival.models.multi_task import NeuralMultiTaskModel

seed = 31415
np.random.seed(seed)

fontsize = 18
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
def BART():
    return ConditionalSurvivalForestModel()
plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)

font = {"family": "normal", "weight": "bold", "size": 24}

plt.rc("font", **font)
params = {
    "legend.fontsize": "x-large",
    # 'figure.figsize': (15, 5),
    "axes.labelsize": "x-large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
}
plt.rcParams.update(params)

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.

sns.set_style("white")
sns.set_context("paper")
sns.set()
title_fontsize = 12
label_fontsize = 10


np.random.seed(1234)
_ = torch.manual_seed(123)
