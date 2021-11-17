
from classes import *
import sympy
class SimulationNew:
    
    def __init__(self, params_simu):

        n_samples = params_simu['n_samples']
        n_features = params_simu['n_features']
        beta = params_simu['beta']
        alpha = params_simu['alpha']
        lamb = params_simu['lamb']
        kappa_cens = params_simu['kappa']
        coef_tt = params_simu['coef_tt']
        rho = params_simu['rho']
        

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
        self.scheme = params_simu['scheme']
        self.scheme_function = params_simu['scheme'].get_scheme_function()
        self.sheme_type = params_simu['scheme'].get_scheme_type()
        self.wd_para = wd_para
        self.f_x = self.scheme_function(self.X)
        self.path_data = params_simu["path_data"]
        if self.sheme_type == 'nonlinear':
            self.Xbeta = self.f_x
        else : 
            self.Xbeta =self.X.dot(self.beta)

    # True Survival function
    def S_p(self, x, tt,t,patient):
        tt_p = tt #[patient]
        c_tt = self.coef_tt * tt_p

        if self.scheme == 'linear':
            x_beta_p =x.dot(self.beta)[patient]
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(x_beta_p + c_tt))
        else:
            sh_z = self.scheme_function(x)[patient] + c_tt
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(sh_z))
        

    """def S(self, xbeta_p, tt_p, t):  # Agathe : pourquoi X apparait et pas Xbeta ?
        c_tt = self.coef_tt * tt_p  # coef_tt : coef de traitement

        if self.scheme == 'linear':
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(xbeta_p + c_tt))
        else:
            sh_z = self.scheme_function(self.X) + c_tt
            return np.exp(-((self.lamb * t) ** self.alpha) * np.exp(sh_z))

    def S_1(self, xbeta_p, t):
        return self.S(xbeta_p, 1, t)

    def S_0(self, xbeta_p, t):
        return self.S(xbeta_p, 0, t)"""

    # Simulation of times to event according to a Weibull-Cox model
    def simulation_T(self, tt):
        c_tt = self.coef_tt * tt
        log_u = np.log(np.random.uniform(0, 1, self.n_samples))

        if self.scheme == 'linear':  # schema linéraire ens X
            log_T = 1.0 / self.alpha * \
                (np.log(-log_u) - (self.Xbeta + c_tt))
        else:  # schema non linéraire en X
            log_T = 1.0 / self.alpha * (np.log(-log_u) - self.scheme_function(self.X) - c_tt)
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
        #data_sim["f_X"] = self.f_x
        self.data_sim = data_sim

        # observed censoring, treatment proportions
        self.perc_treatement = int((data_sim["tt"].mean() * 100))
        self.perc_event = int(data_sim["event"].mean() * 100)
        # Wasserstein distances
        print("WD = ", wd)
        print(f"tt = 1 : {self.perc_treatement} % ")
        print(f"event = 1 : {self.perc_event} %")
        print('Scheme : ', self.sheme_type)
    
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

        #xbeta_p = self.Xbeta[patient]

        T_0_p = self.data_sim["T_0"].values[patient].round(2)
        T_1_p = self.data_sim["T_1"].values[patient].round(2)
        tt_p = self.data_sim["tt"].values[patient]
        self.S_0_true = self.S_p(self.X,0,times,patient)

        self.S_1_true = self.S_p(self.X,1,times,patient)

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

class Scheme:
    """
    input:
        type_s: 'linear' or 'nonlinear'
        function: function to be used
    """
    def __init__(self, type_s, function):
        self.type = type_s
        self.function = function  # fonction de x , args*
    
    def get_scheme_type(self):
        return self.type
    def get_scheme_function(self):
        return self.function
  
        
  




"""
Simulation parameters
"""
"""p_sim = {}
p_sim['n_samples'] = 1000
p_sim['n_features'] = 25
p_sim['beta'] = [0.01 * (p_sim['n_features'] - i) / p_sim['n_features']
                 for i in range(0, p_sim['n_features'])]



p_sim['alpha'] = 3
p_sim['lamb'] = 1
p_sim['coef_tt'] = 1.8
p_sim['rho'] = 0.0
p_sim['kappa'] = 3.
p_sim['wd_param'] = 3. #4.

p_sim['path_data'] = "./sim_surv"


# x : matrixe de features
# sin(πXi1Xi2) + 2(Xi3 − 0.5)2 + Xi4 + 0.5Xi5,
f1 = lambda x : np.sin(np.pi*x[:,0]*x[:,1] + 2*(x[:,2]-0.5)**2 + x[:,3] + 0.5*x[:,4])
# 

# use sympy to plot the function f1 
# 
                     
p_sim['scheme'] =  Scheme('nonlinear', f1)


# Simulation of data
simu = SimulationNew(p_sim)
data = simu.simulation_surv()
print(data)
print(simu)

S = simu.S_p (x=simu.X , tt=0, t=np.linspace(0, max(simu.data_sim["T_f"]), 100),patient=1)
"""

"""# plot S
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(111)
ax.plot(
    np.linspace(0, max(simu.data_sim["T_f"]), 100),
    S,
    # add sheme
    label="S_true for tt=0 and sheme = {}".format(simu.scheme.get_scheme_type()),
    marker="o",
    markersize=1,
    color="b",
)
plt.title("S_true for tt=0 and sheme = {}".format(simu.scheme.get_scheme_type()))
plt.legend()
  
plt.show() """



## class Experiment
#class Experiment:
#    def __init__(self, p_sim, p_exp):
#        self.p_sim = p_sim
#        self.p_exp = p_exp
#        self.data_sim = None
#        self.data_exp = None
#        self.wd = None
#
#    def simulation_data(self):
#        simu = SimulationNew(self.p_sim)
#        self.data_sim = simu.simulation_surv()
#        return self.data_sim
#
#    def experiment_data(self):
#        self.data_exp = pd.read_csv(self.p_exp, index_col=0)
#        return self.data_exp
#
#    def wd_exp_sim(self):
#        wd = self.data_exp.apply(lambda row: wd_exp_sim(row, self.data_sim), axis=1)
#        self.wd = wd
#        return self.wd
#
#    def plot_dist(self):
#        colmns = ["X" + str(j) for j in range(1, self.p_sim['n_features'] + 1)] + ["tt"]
#        return sns.pairplot(self.data_sim[colmns], hue="tt", diag_kind="hist", height=6)
#
#    def plot_surv_true(self, patient=1):
#        t_max = max(self.data_sim["T_f"])
#        times = np.linspace(0, t_max, 100)
#        xbeta_p = self.data_sim["Xbeta"].values[patient]
#        T_0_p = self.data_sim["T_0"].values[patient].round(2)
import neptune.new as neptune

"""run = neptune.init(
    project="aabraich/APP0",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTllZGVjNy1jMWVmLTRjNzktYTIyNi0yM2JiNjIwZDkyZjgifQ==",
)  # your credentials
"""
## SEND TO NEPTUNE
class Neptune: 
    def __init__(self, experiment_name):
        self.project ='aabraich/APP0'
        self.api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5NTllZGVjNy1jMWVmLTRjNzktYTIyNi0yM2JiNjIwZDkyZjgifQ==",

        self.experiment_name = experiment_name
        self.experiment = None
        
    def create_experiment(self):
        self.experiment = neptune.create_experiment(
            name=self.experiment_name,
            project_name=self.project,
            api_token=self.api_token,
        )
        return self.experiment
    
    def send_data(self, df, name):
        df.to_csv("./data_exp/"+name)
        self.experiment.send_file("./data_exp/"+name, name)
        
    def send_param(self, param):
        self.experiment.send_config(param)
    def send_plot(self, fig, name):
        fig.savefig("./fig/"+name)
        self.experiment.send_file("./fig/"+name, name)
    
        
        
    