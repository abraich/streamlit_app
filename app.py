

import streamlit as st
from utils import *
from classes import *
from simulationnew import *
from sklearn.manifold import TSNE
import seaborn as sns
import SessionState

st.sidebar.header("Tasks")
tasks_choices = ['Simulation', 'Tunning', 'Benchmarking',
                 'Plots', 'WD vs n_samples', 'WD vs n_features']
task = st.sidebar.selectbox("Choose a task", tasks_choices)

st.title(f"SurvCaus- {task}")

session_state = SessionState.get(
    name='', params=None)  # cache the session state

session_state1 = SessionState.get(
    name='', params_best_survcaus=None)  # cache the session state

session_state2 = SessionState.get(
    name='', params_best_bart=None)  # cache the session state

ev_cache = SessionState.get(name='', Ev=None)
if task == 'Simulation':
    p_sim = {}
    p_sim['n_samples'] = 1000
    p_sim['n_features'] = 25
    idx = np.arange(p_sim['n_features'])
    #p_sim['beta'] = [0.01 * (p_sim['n_features'] - i) / p_sim['n_features'] for i in range(0, p_sim['n_features'])]

    # simule beta  as : (-1) ** idx * np.exp(-idx / 10.)
    p_sim['beta'] = (-1) ** idx * np.exp(-idx / 20.)
    beta_norm = np.linalg.norm(p_sim['beta'])

    p_sim['alpha'] = 3
    p_sim['lamb'] = 1
    p_sim['coef_tt'] = 1.8
    p_sim['rho'] = 0.0
    p_sim['kappa'] = 3.
    p_sim['wd_param'] = 3.  # 4.
    p_sim['path_data'] = "./sim_surv"
    params = p_sim.copy()

    n_samples = st.sidebar.number_input(
        "n_samples", min_value=1000, max_value=10000)
    n_features = st.sidebar.number_input(
        "n_features", min_value=25, max_value=30)

    coef_tt = st.sidebar.number_input("coef_tt", value=1.8)
    wd_param = st.sidebar.number_input("wd_param", value=3.)
    kappa = st.sidebar.number_input("kappa", 3.)
    rho = st.sidebar.number_input("rho", 0.1)
    scheme = st.sidebar.selectbox("scheme", ['linear', 'nonlinear'])
    coef_beta = st.sidebar.number_input("coef_beta", value=1.)
    coef_beta = coef_beta / beta_norm
    params['beta'] = coef_beta * p_sim['beta']
    beta_tcga = st.sidebar.selectbox("isbeta_tcga", [False, True])

    alpha = st.sidebar.number_input("alpha", min_value=3., max_value=5.)
    lamb = st.sidebar.number_input("lamb", min_value=1., max_value=3.)
    if scheme == 'linear':
        params['scheme'] = Scheme('linear')
    else:
        # select function :
        # sin(πXi1Xi2) + 2(Xi3 − 0.5)2 + Xi4 + 0.5Xi5
        def f1(x): return np.sin(
            np.pi*x[:, 0]*x[:, 1] + 2*(x[:, 2]-0.5)**2 + x[:, 3] + 0.5*x[:, 4])

        # 2 log(1 + exp(Xi1+Xi2+Xi3 )
        def f2(x): return np.log(1 + np.exp(x[:, 0] + x[:, 1] + x[:, 2]))

        #  (max{Xi1 + Xi2 + Xi3, 0} + max{Xi4 + Xi5, 0}) /2
        def f3(x): return (np.maximum(
            x[:, 0] + x[:, 1] + x[:, 2], 0) + np.maximum(x[:, 3] + x[:, 4], 0)) / 2

        #  max{Xi1 + Xi2, Xi3, 0} + max{Xi4 + Xi5, 0}
        def f4(x): return (np.maximum(
            x[:, 0] + x[:, 1], 0) - np.maximum(x[:, 3] + x[:, 4], 0))
        # print functions
        st.sidebar.markdown("## Functions")
        # use latex for display
        st.sidebar.markdown(
            r"$f_1$ : $sin\left(\pi\times X_1\times X_2\right) + 2\left(X_3 - 0.5\right)^2 + X_4 + 0.5X_5$")
        st.sidebar.markdown(
            r"$f_2$ : $2\log\left(1 + e^{X_1 + X_2 + X_3}\right)$")
        st.sidebar.markdown(
            r"$f_3$ : $\frac{\max\left(X_1 + X_2 + X_3, 0\right) + \max\left(X_4 + X_5, 0\right)}{2}$")
        st.sidebar.markdown(
            r"$f_4$ : $\max\left(X_1 + X_2, X_3, 0\right) + \max\left(X_4 + X_5, 0\right)$")
        f_choices = ['f1', 'f2', 'f3', 'f4']
        f_choice = st.sidebar.selectbox("Select function", f_choices)
        if f_choice == 'f1':
            params['scheme'] = Scheme('nonlinear', f1)
        elif f_choice == 'f2':
            params['scheme'] = Scheme('nonlinear', f2)
        elif f_choice == 'f3':
            params['scheme'] = Scheme('nonlinear', f3)
        elif f_choice == 'f4':
            params['scheme'] = Scheme('nonlinear', f4)

    patient = st.sidebar.number_input("patient", 0)

    if beta_tcga:
        params['beta'] = pd.read_csv(
            './beta_TGCA.csv')['x'].values[:n_features]

    params['n_samples'] = n_samples
    params['n_features'] = n_features
    params['alpha'] = alpha
    params['lamb'] = lamb
    params['coef_tt'] = coef_tt
    params['kappa'] = kappa
    params['wd_param'] = wd_param
    params['rho'] = rho

    session_state.params = params
    # Simulation of data

    simu = SimulationNew(params)

    df = simu.simulation_surv()
    st.dataframe(df)

    st.dataframe(df.describe())
    if scheme == 'linear':
        st.write("Beta = ", list(params['beta']))
    else:
        st.write("f_choice = ", f_choice)
    st.write("% event=1 : ", simu.perc_event)
    st.write("% tt=1 : ", simu.perc_treatement)
    st.write("WD : ", simu.wd)
    st.write('Cov matrix : ', simu.cov)
    # TSNE
    x = df.iloc[:, :p_sim['n_features']]
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    z = tsne.fit_transform(x)
    d = pd.DataFrame()
    d["tt"] = df[['tt']].values.squeeze()
    d["comp-1"] = z[:, 0]
    d["comp-2"] = z[:, 1]

    fig = plt.figure()
    sns.scatterplot(x="comp-1", y="comp-2", hue=d.tt.tolist(),
                    palette=sns.color_palette("hls", 2),
                    data=d).set(title="Sampled data T-SNE projection")
    fig.show()
    st.pyplot(fig)

    st.write('# plot survival function')

    fig_surv = simu.plot_surv_true(patient=patient)
    st.pyplot(fig_surv)


tunning_cache = SessionState.get(name='', tunning=None)

if task == 'Tunning':
    # Tunning og hyperparameters
    p_sim = session_state.params
    tunning = Tunning(p_sim)
    n_trials = st.sidebar.number_input("n_trials", min_value=1, max_value=1000)

    # run bouton for tunning
    if st.sidebar.button("Tunning Survcaus"):
        with st.spinner(text='Tunning Survcaus in progress'):
            p_survcaus_best = tunning.get_best_hyperparameter_survcaus(
                n_trials=n_trials)
            st.success('Done')
        st.write("Best hyperparameter for survcaus : ", p_survcaus_best)
        # cache tunning
        tunning_cache.tunning = tunning

        session_state1.params_best_survcaus = p_survcaus_best

    if st.sidebar.button("Tunning BART"):
        tunning = tunning_cache.tunning
        with st.spinner(text='Tunning BART in progress'):
            p_bart_best = tunning.get_best_hyperparameter_bart(
                n_trials=n_trials)
            st.success('Done')
        st.write("Best hyperparameter for BART : ", p_bart_best)
        session_state2.params_best_bart = p_bart_best


if task == 'Benchmarking':
    ev_cache = SessionState.get(name='', Ev=None)

    # cached_params
    p_sim = session_state.params
    p_survcaus_best = session_state1.params_best_survcaus
    p_bart_best = session_state2.params_best_bart
    # Evaluation - benchmark
    p_survcaus = p_survcaus_best.copy()

    p_survcaus['patience'] = 2
    p_survcaus['epochs'] = 40
    p_survcaus['batch_size'] = 256

    p_bart_best = session_state2.params_best_bart

    st.write("Best hyperparameter for survcaus : ", p_survcaus_best)
    st.write("Best hyperparameter for BART : ", p_bart_best)

    Ev = Evaluation(p_sim, p_survcaus)

    list_models = st.sidebar.multiselect('List of models', [
                                         "SurvCaus", "SurvCaus_0", 'CoxPH', 'BART'], default=["SurvCaus", "SurvCaus_0", 'CoxPH', 'BART'])
    st.write("Choosed models : ", list_models)
    if st.button('Train !'):
        with st.spinner(text='In progress'):
            Ev.All_Results(list_models=list_models,
                           is_train=False, params_bart=p_bart_best)
            st.success('Done')
        ev_cache.Ev = Ev

    Ev = ev_cache.Ev
    bilan_benchmark = Ev.bilan_benchmark

    st.write(bilan_benchmark)
    # st.write(pd.DataFrame(bilan_benchmark))
    st.pyplot(Ev.box_plot_cate)
    st.pyplot(Ev.box_plot_pehe)
    st.pyplot(Ev.box_plot_surv0)
    st.pyplot(Ev.box_plot_surv1)
    st.pyplot(Ev.box_plot_FSM)

    st.write('# plot survival function')
    patient = st.sidebar.number_input("patient", 1)
    for model_name in Ev.list_models[1:]:
        st.write('## '+model_name)
        fig_surv, fig_cate = plots(patient, Ev.d_list_models, model_name)
        st.pyplot(fig_surv)
        st.pyplot(fig_cate)
if task == 'Plots':
    Ev = ev_cache.Ev
    st.write('# plot survival function')
    patient = st.sidebar.number_input("patient", 1)
    for model_name in Ev.list_models[1:]:
        st.write('## '+model_name)
        fig_surv, fig_cate = plots(patient, Ev.d_list_models, model_name)
        st.pyplot(fig_surv)
        st.pyplot(fig_cate)

if task == 'WD vs n_samples':
    p_sim = {}
    p_sim['n_samples'] = 1000
    p_sim['n_features'] = 25
    p_sim['beta'] = [0.01 * (p_sim['n_features'] - i) / p_sim['n_features']
                     for i in range(0, p_sim['n_features'])]
    p_sim['alpha'] = 3
    p_sim['lamb'] = 1
    p_sim['coef_tt'] = 1.8
    p_sim['rho'] = 0.0
    p_sim['kappa'] = 3.
    p_sim['wd_param'] = 3.  # 4.
    p_sim['scheme'] = 'linear'  # 'linear'
    p_sim['path_data'] = "./sim_surv"
    params = p_sim.copy()
    wd_param = st.sidebar.number_input("wd_param", min_value=0.)
    p_sim['wd_param'] = wd_param
    n_samples_min = st.sidebar.number_input(
        "n_samples_min", min_value=1000)
    n_samples_max = st.sidebar.number_input(
        "n_samples_max", max_value=10000)
    steps = st.sidebar.number_input("steps", min_value=1)

    # Simulation of data
    i = n_samples_min
    L_wd = []
    L_n = []
    while i < n_samples_max:
        print(i)
        params["n_samples"] = i
        simu = Simulation(params)
        simu.simulation_surv()
        L_wd.append(simu.wd)
        L_n.append(i)
        i += steps

    fig1 = plt.figure(figsize=(18, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(L_n, L_wd, label='WD vs n_samples')
    plt.legend()
    plt.close()
    st.pyplot(fig1)


if task == 'WD vs n_features':
    p_sim = {}
    p_sim['n_samples'] = 1000
    p_sim['n_features'] = 25
    p_sim['beta'] = [0.01 * (p_sim['n_features'] - i) / p_sim['n_features']
                     for i in range(0, p_sim['n_features'])]
    p_sim['alpha'] = 3
    p_sim['lamb'] = 1
    p_sim['coef_tt'] = 1.8
    p_sim['rho'] = 0.0
    p_sim['kappa'] = 3.
    p_sim['wd_param'] = 3.  # 4.
    p_sim['scheme'] = 'linear'  # 'linear'
    p_sim['path_data'] = "./sim_surv"
    params = p_sim.copy()
    wd_param = st.sidebar.number_input("wd_param", min_value=0.)
    p_sim['wd_param'] = wd_param
    n_features_min = st.sidebar.number_input(
        "n_features_min", min_value=10)
    n_features_max = st.sidebar.number_input(
        "n_features_max", max_value=100)
    steps = st.sidebar.number_input("steps", min_value=1)

    # Simulation of data
    i = n_features_min
    L_wd = []
    L_n = []
    while i < n_features_max:
        print(i)
        params["n_features"] = i
        params['beta'] = [0.01 * (params['n_features'] - i) / params['n_features']
                          for i in range(0, params['n_features'])]
        simu = Simulation(params)
        simu.simulation_surv()
        L_wd.append(simu.wd)
        L_n.append(i)
        i += steps

    fig1 = plt.figure(figsize=(18, 10))
    ax1 = fig1.add_subplot(111)
    ax1.plot(L_n, L_wd, label='WD vs n_features')
    plt.legend()
    plt.close()
    st.pyplot(fig1)

 # X: multivariate, toplitz 0.2

# changer n ( 1000,2000,50000) , (voir johansson), scénarion lin/non linéaire,  wd

# tunning BART


# brca
# acgt


# FIXER N = 1000, N FEATURES = 25
#  1 ) notbook : input fonction non linaire  : devent topleit sigma = 0.25
# VENDREDI VOIR AGATHE SI POSSIBLE


# RDV MERCREDI : 9:30
# APR_S LES MARDIS
