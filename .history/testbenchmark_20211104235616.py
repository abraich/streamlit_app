from utils import *
from classes import *
from losses import *

# defin a class for model evaluation 
# input : params_simu, params_survcaus
class Evaluation(object):
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
