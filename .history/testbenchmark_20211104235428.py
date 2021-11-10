from utils import *
from classes import *
from losses import *

# defin a class for model evaluation and testing
# input : params_simu, params_survcaus
class TestBenchmark(object):
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

    