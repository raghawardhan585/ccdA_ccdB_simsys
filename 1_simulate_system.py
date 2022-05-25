import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import copy
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
from sklearn.metrics import r2_score
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import os
import shutil
import pickle
from itertools import product

# k1f = 1.4
# k1i = 0.003
# k2f = 1.1
# k2i = 0.19
# k3f = 0.04
# k3i = 2.2
# k4f = 0.35 # Changed
# k4i = 2.2
# k5f = 0.01 # Assumed because it is not given
# k5i = 0.3 # Assumed because it is not given

k1f = 1.4#e6
k1i = 0.003
k2f = 1.1#e5
k2i = 0.19
k3f = 0.04
k3i = 2.2
k4f = 0.0035
k4i = 2.2
k5f = 0.14 # Assumed because it is not given
k5i = 0.13 # Assumed because it is not given

gamma_1 = 0.3
gamma_2 = 0.1
gamma_3 = 0.03
gamma_4 = 0.02
gamma_5 = 0.4
gamma_6 = 0.09
gamma_7 = 0.0

a1 = 0.8
a2 = 1.9
a3 = 4.
a4 = 0.7

K1 = 0.3
K2 = 2.
K3 = 4.
K4 = 0.5

n1 = 2
n2 = 5
n3 = 2
n4 = 3

d1 = 0.2
d2 = 0.03
d3 = 0.3
d4 = 0.1

mu_y = 10
Kya = 10
Kyr = 0.1

def ccdA_ccdB_Antitoxin_Toxin_system(x,t):
    xdot = np.zeros(len(x))
    xdot[0] = -k1f*x[0]*x[1] + k1i*x[2] - gamma_1*x[0]
    xdot[1] = -k1f*x[0]*x[1] - k2f*x[1]*x[2] - k5f*x[1]*x[4] + k1i*x[2] + k2i*x[3] +k5i*x[5] - gamma_2*x[1]
    xdot[2] = k1f*x[0]*x[1] - k2f*x[1]*x[2] - k1i*x[2] + k2i*x[3] - k4f*x[2] +k4i*x[4]*x[6] - gamma_3*x[2]
    xdot[3] = k2f*x[1]*x[2] - k2i*x[3] +k3i*x[5]*x[6] -k3f*x[3] - gamma_4*x[3]
    xdot[4] = k4f*x[2] -k4i*x[4]*x[6] -k5f*x[4] + k5i*x[1]*x[5] - gamma_5*x[4]
    xdot[5] = k5f*x[1]*x[4] - k5i*x[5] +k3f*x[3] -k3i*x[5]*x[6] - gamma_6*x[5]
    xdot[6] = k4f*x[2] -k4i*x[4]*x[6] + k3f*x[3] - k3i*x[5]*x[6] - gamma_7*x[6]
    xdot[7] = a1 * (x[6] / K1) ** n1 / (1 + (x[6] / K1) ** n1) - d1*x[7]
    xdot[8] = a2 * (x[6] / K2) ** n2 / (1 + (x[6] / K2) ** n2) - d2*x[8]
    xdot[9] = a3 * (x[6] / K3) ** n3 / (1 + (x[6] / K3) ** n3) - d3*x[9]
    xdot[10] = a4 * (x[6] / K4) ** n4 / (1 + (x[6] / K4) ** n4) - d4*x[10]
    return xdot



N_SIMULATIONS = 300
simulation_time = 100
sampling_time = 1
t = np.arange(0, simulation_time, sampling_time)
numpy_random_initial_condition_seed = 30
np.random.seed(numpy_random_initial_condition_seed)
ls_seed_for_initial_condition = np.random.randint(0,10000,(N_SIMULATIONS))


ls_data = []
for i in range(N_SIMULATIONS):
    # Simulation Parameters
    x0 = np.array([0.4,0.1,0.2,0.4,0.3,0.8,0.5])
    x0_proteins = np.array([0.3,0.8,0.1,1.8])
    np.random.seed(ls_seed_for_initial_condition[i])
    y0 = 0.02
    x0 = np.concatenate([x0,x0_proteins],axis=0)
    x0_i = x0 + np.random.uniform(0, 1, size=x0.shape) #0.15
    XT_i = odeint(ccdA_ccdB_Antitoxin_Toxin_system,x0_i,t)
    Y1_i = y0*np.exp(mu_y*(XT_i[:,7]/Kya)**1/(1 + (XT_i[:,7]/Kya)**1 + (XT_i[:,10]/Kyr)**2))
    # Y2_i = y0*np.exp(mu_y*XT_i[:,7]/(K_y + XT_i[:,7]))
    # Y3_i = y0*np.exp(mu_y/(K_y + X_i[:,10]))
    if np.max(Y1_i)>100:
        print('SYSTEM BLEW UP!')
    ls_data.append({'XT': XT_i, 'YT': Y1_i.reshape(-1,1)})
    # #
f,ax = plt.subplots(3,12, sharex=True, figsize = (36,9))
# ax = ax.reshape(-1)
for i in range(len(ls_data)):
    if i<N_SIMULATIONS/3:
        colorname='tab:blue'
        var_i = 0
    elif i<2*N_SIMULATIONS/3:
        colorname='tab:orange'
        var_i = 1
    else:
        colorname='tab:green'
        var_i = 2
    for state in range(11):
        ax[var_i,state].plot(ls_data[i]['XT'][:, state], color=colorname)
    ax[var_i,-1].plot(ls_data[i]['YT'][:, 0], color=colorname)
f.show()



## SYSTEM 2: Closed Form System

# a11_cfs = -0.4
# a21_cfs = -5
# a22_cfs = -3
# gamma_cfs = 0.7
#
# def closed_form_system(x,t):
#     xdot = np.zeros(len(x))
#     xdot[0] = a11_cfs*x[0]
#     xdot[1] = a21_cfs*x[0] + a22_cfs*x[1] + gamma_cfs *x[0]**2
#     return xdot
#
# N_SIMULATIONS = 300
# simulation_time = 3
# sampling_time = 0.1
# t = np.arange(0, simulation_time, sampling_time)
# numpy_random_initial_condition_seed = 10
# ls_seed_for_initial_condition = np.random.randint(0,10000,(N_SIMULATIONS))
#
#
# ls_data = []
# for i in range(N_SIMULATIONS):
#     # Simulation Parameters
#     x0 = np.array([2.4,2.8])
#     np.random.seed(ls_seed_for_initial_condition[i])
#     x0_i = x0 + np.random.normal(0, 0.5, size=x0.shape)
#     XT_i = odeint(closed_form_system,x0_i,t)
#     # Y1_i = XT_i[:,0:1] + gamma_cfs*XT_i[:,0:1]**2
#     Y1_i = XT_i[:,0:1]**3
#     ls_data.append({'XT': XT_i, 'YT': Y1_i})
#     # #
# f,ax = plt.subplots(3,3, sharex=True, figsize = (36,9))
# # ax = ax.reshape(-1)
# for i in range(len(ls_data)):
#     if i<N_SIMULATIONS/3:
#         colorname='tab:blue'
#         var_i = 0
#     elif i<2*N_SIMULATIONS/3:
#         colorname='tab:orange'
#         var_i = 1
#     else:
#         colorname='tab:green'
#         var_i = 2
#     for state in range(len(x0)):
#         ax[var_i,state].plot(ls_data[i]['XT'][:, state], color=colorname)
#     ax[var_i,-1].plot(ls_data[i]['YT'][:, 0], color=colorname)
# f.show()

##
# deepDMD formulation

for dict_data_i in ls_data[0:200]:
    try:
        XT = np.concatenate([XT,dict_data_i['XT']], axis=0)
        XpT = np.concatenate([XpT, dict_data_i['XT'][0:-1]], axis=0)
        XfT = np.concatenate([XfT, dict_data_i['XT'][1:]], axis=0)
        YpT = np.concatenate([YpT, dict_data_i['YT'][0:-1]], axis=0)
        YfT = np.concatenate([YfT, dict_data_i['YT'][1:]], axis=0)
        YT = np.concatenate([YT, dict_data_i['YT']], axis=0)
        XT_eq_all = np.concatenate([XT_eq_all, dict_data_i['XT'][-1:]], axis=0)
    except:
        XT = dict_data_i['XT']
        XpT = dict_data_i['XT'][0:-1]
        XfT = dict_data_i['XT'][1:]
        YpT = dict_data_i['YT'][0:-1]
        YfT = dict_data_i['YT'][1:]
        YT = dict_data_i['YT']
        XT_eq_all = dict_data_i['XT'][-1:]

class bias_lifting():
    def __init__(self, with_bias = True):
        self.with_bias = with_bias
        return
    def lift(self,XT):
        if self.with_bias:
            psiXT = np.concatenate([XT,np.ones(XT.shape[0]).reshape(-1,1)],axis=1)
        else:
            psiXT = XT
        return psiXT
    def unlift(self,psiXT):
        if self.with_bias:
            XT = psiXT[:,0:-1]
        else:
            XT = psiXT
        return XT

def fit_K(X, Y, fit_bias_of_X=True):
    if fit_bias_of_X:
        KT = np.linalg.pinv(X) @ Y
    else:
        KT = np.linalg.pinv(X) @ Y[:,0:-1]
        KT = np.concatenate([KT, np.zeros((KT.shape[0],1))],axis=1)
        KT[-1,-1] = 1
    return KT


# Attempting a linear fit

# Scaling
scaler_X = StandardScaler(with_mean=True, with_std=True)
scaler_X.fit(XT)
scaler_Y = StandardScaler(with_mean=True, with_std=True)
scaler_Y.fit(YT)
XTs = scaler_X.transform(XT)
XpTs = scaler_X.transform(XpT)
XfTs = scaler_X.transform(XfT)
YTs = scaler_Y.transform(YT)
YpTs = scaler_Y.transform(YpT)
YfTs = scaler_Y.transform(YfT)
# Lifting
lift_X = bias_lifting(with_bias=True)
psiXT = lift_X.lift(XTs)
psiXpT = lift_X.lift(XpTs)
psiXfT = lift_X.lift(XfTs)
# Fitting with a bias
KT = fit_K(psiXpT, psiXfT, fit_bias_of_X= False)
WhT = fit_K(psiXT, YTs, fit_bias_of_X= True)
# Plot eigen values
eig = np.linalg.eigvals(KT)
f,ax = plt.subplots()
colored_circle = plt.Circle(( 0. , 0. ), 1.0 ,color='cyan')
ax.set_aspect( 1 )
ax.add_artist( colored_circle )
ax.plot(np.real(eig), np.imag(eig),'*')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
f.show()

##
SYSTEM_NO = 5
storage_folder = '/Users/shara/Desktop/ccdA_ccdB_simsys/System_' + str(SYSTEM_NO)
if os.path.exists(storage_folder):
    shutil.rmtree(storage_folder)
    os.mkdir(storage_folder)
else:
    os.mkdir(storage_folder)
# Save the scaler
dict_Scaler = {'XT': scaler_X, 'YT': scaler_Y}
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
    pickle.dump(dict_Scaler, handle)
# Save the data for OC_deepDMD
dict_DATA_SCALED = {'XpT': XpTs, 'XfT': XfTs, 'YpT': YpTs, 'YfT': YfTs}
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DeepDMDdata_Scaled.pickle', 'wb') as handle:
    pickle.dump(dict_DATA_SCALED, handle)
# Save the original data
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
    pickle.dump(ls_data, handle)

##

# n - step Prediction
# ls_pred_list = list(range(40,len(ls_data)))
ls_pred_list = list(range(0,300))
XT_all = np.empty((0,ls_data[0]['XT'].shape[1]))
YT_all = np.empty((0,ls_data[0]['YT'].shape[1]))
XT_est_all = np.empty((0,ls_data[0]['XT'].shape[1]))
YT_est_all = np.empty((0,ls_data[0]['YT'].shape[1]))
XT_1step_est_all = np.empty((0,ls_data[0]['XT'].shape[1]))
YT_1step_est_all = np.empty((0,ls_data[0]['YT'].shape[1]))
for i in ls_pred_list:
    # Get X
    XTi = ls_data[i]['XT']
    # Scale the data
    XTsi = scaler_X.transform(XTi)
    # Lift the observables
    psiXTi = lift_X.lift(XTsi)
    # 1 - step prediction
    psiXTi_1step_est = np.concatenate([psiXTi[0:1], psiXTi[0:-1] @ KT],axis=0)
    # Get the n - step predictions
    psiXTi_est = psiXTi[0:1]
    for i in range(psiXTi.shape[0]-1):
        psiXTi_est = np.concatenate([psiXTi_est, psiXTi_est[-1:] @ KT], axis=0)
    YTsi_est = psiXTi_est @ WhT
    YTsi_1step_est = psiXTi_1step_est @ WhT
    # Unlift the observables
    XTsi_est = lift_X.unlift(psiXTi_est)
    XTsi_1step_est = lift_X.unlift(psiXTi_1step_est)
    # Inverse scale
    XTi_1step_est = scaler_X.inverse_transform(XTsi_1step_est)
    XTi_est = scaler_X.inverse_transform(XTsi_est)
    YTi_1step_est = scaler_Y.inverse_transform(YTsi_1step_est)
    YTi_est = scaler_Y.inverse_transform(YTsi_est)
    # Concatenate the corresponding variables
    XT_all = np.concatenate([XT_all, XTi],axis=0)
    XT_est_all = np.concatenate([XT_est_all, XTi_est], axis=0)
    XT_1step_est_all = np.concatenate([XT_1step_est_all, XTi_1step_est], axis=0)
    YT_all = np.concatenate([YT_all, ls_data[i]['YT']], axis=0)
    YT_est_all = np.concatenate([YT_est_all, YTi_est], axis=0)
    YT_1step_est_all = np.concatenate([YT_1step_est_all, YTi_1step_est], axis=0)




# print('r2 score of n-step prediction in psiX : ', r2_score(psiXT, psiXT_est,multioutput='uniform_average'))
# print('r2 score of n-step prediction in Xs : ', r2_score(XTs, XTs_est,multioutput='uniform_average'))
print('r2 score of 1-step prediction in X : ', r2_score(XT_all, XT_1step_est_all,multioutput='uniform_average'))
print('r2 score of 1-step prediction in Y : ', r2_score(YT_all, YT_1step_est_all,multioutput='uniform_average'))
print('r2 score of n-step prediction in X : ', r2_score(XT_all, XT_est_all,multioutput='uniform_average'))
print('r2 score of n-step prediction in Y : ', r2_score(YT_all, YT_est_all,multioutput='uniform_average'))
# f, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 15))
# ax[0].plot(t, XT_est[:, 0:7])
# ax[0].plot(t, XT[:, 0:7], '.')
# ax[0].legend(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'], ncol=2)
# ax[1].plot(t, XT_est[:, 7:])
# ax[1].plot(t, XT[:, 7:], '.')
# ax[1].legend(['x8', 'x9', 'x10', 'x11'], ncol=2)
# # ax[2].plot(t, Y1, linewidth=4)
# # ax[2].plot(t, Y2, linewidth=4)
# # ax[2].plot(t,Y3,linewidth = 4)
# # ax[2].legend(['Y[x8_act + x11_rep]', 'Y[x8_act]', 'Y[x11_rep]'], ncol=1)
# f.show()


##
## Bash Script Generation

N_NODES_PER_OBSERVABLE = 1.5

dict_hp={}
dict_hp['ls_dict_size'] = [16,20,24]
dict_hp['ls_nn_layers'] = [3,4]
dict_hp['System_no'] = []
dict_hp['System_no'] = dict_hp['System_no'] + [5]
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(1,7))   #mt
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(11,13))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(21,29))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(31,40))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(41,50))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(51,60))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(61,70))
# dict_hp['System_no'] = dict_hp['System_no'] + list(range(71,80))

system_running = 'goldentensor'
# system_running = 'optictensor'
# system_running = 'microtensor'
# system_running = 'quantensor'

file = open('/Users/shara/Desktop/ccdA_ccdB_simsys/' + system_running + '_run.sh','w')
if system_running in ['microtensor', 'quantensor', 'goldentensor', 'optictensor']:
    ls_device = [' \'/cpu:0\' ']
# elif system_running in ['goldentensor', 'optictensor']:
#     ls_device = [' \'/cpu:0\' ', ' \'/gpu:0\' ', ' \'/gpu:1\' ', ' \'/gpu:2\' ', ' \'/gpu:3\' ']

# For each system of interest
dict_system_next_run = {}
for system_no in dict_hp['System_no']:
    # Create a MyRunInfo folder
    runinfo_folder = 'System_' + str(system_no) + '/MyRunInfo'
    if not os.path.exists(runinfo_folder):
        os.mkdir(runinfo_folder)
    if not os.path.exists(runinfo_folder + '/dummy_proxy.txt'):
        with open(runinfo_folder + '/dummy_proxy.txt', 'w') as f:
            f.write('This is created so that git does not experience an issue with')
    # Get the latest run number for each system # TODO - Check this part of the code
    try:
        ls_all_run_files = os.listdir('System_' + str(system_no) + '/MyMac')
        ls_run_numbers = [np.int(i[4:]) for i in ls_all_run_files if 'RUN_' in i]
        next_run = np.int(np.max(ls_run_numbers)) +1 + 6
    except:
        next_run = 0
    dict_system_next_run[system_no] = next_run

file.write('#!/bin/bash \n')
# file.write('rm nohup.out \n')
file.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] \n')
ls_all_runs = []
n_devices = len(ls_device)
for system_no,n_x,n_l in product(dict_hp['System_no'],dict_hp['ls_dict_size'],dict_hp['ls_nn_layers']):
    run_number = dict_system_next_run[system_no]
    device_name = ls_device[np.mod(run_number,n_devices)]
    # Check if run file exists
    run_info_file = ' > System_' + str(system_no) + '/MyRunInfo/Run_' + str(run_number) + '.txt & \n'
    n_n = np.int(np.ceil(n_x*N_NODES_PER_OBSERVABLE))
    file.write('python3 deepDMD.py' + device_name + str(system_no) + ' ' + str(run_number) + ' ' + str(n_x) + ' ' + str(n_l) + ' ' + str(n_n) + run_info_file)
    if device_name == ls_device[-1]:
        file.write('wait\n')
    # Incrementing to the next run
    dict_system_next_run[system_no] = dict_system_next_run[system_no] + 1
file.write('echo "All sessions are complete" \n')
file.write('echo "=======================================================" \n')
file.close()



