import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from itertools import product
import random
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
import copy
import pickle
plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
import os
import shutil
import tensorflow as tf
import seaborn as sb

# Next set of functions requried

def get_dict_param(run_folder_name_curr,SYS_NO,sess):
    dict_p = {}
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_curr + '/System_' + str(SYS_NO) + '_DeepDMDdata_Scaled.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name_curr))
    try:
        psixpT = tf.get_collection('psixpT')[0]
        psixfT = tf.get_collection('psixfT')[0]
        xpT_feed = tf.get_collection('xpT_feed')[0]
        xfT_feed = tf.get_collection('xfT_feed')[0]
        KxT = tf.get_collection('KxT')[0]
        KxT_num = sess.run(KxT)
        dict_p['psixpT'] = psixpT
        dict_p['psixfT'] = psixfT
        dict_p['xpT_feed'] = xpT_feed
        dict_p['xfT_feed'] = xfT_feed
        dict_p['KxT_num'] = KxT_num
    except:
        print('State info not found')
    # try:
    ypT_feed = tf.get_collection('ypT_feed')[0]
    yfT_feed = tf.get_collection('yfT_feed')[0]
    dict_p['ypT_feed'] = ypT_feed
    dict_p['yfT_feed'] = yfT_feed
    WhT = tf.get_collection('WhT')[0]
    WhT_num = sess.run(WhT)
    dict_p['WhT_num'] = WhT_num
    # except:
    #     print('No output info found')
    return dict_p

##
SYSTEM_NO = 5
RUN_NO = 9
run_folder_name = 'System_' + str(SYSTEM_NO) + '/MyMac/RUN_' + str(RUN_NO)
sess = tf.InteractiveSession()
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)

# Load the data
simulation_data_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle'
# simulation_datainfo_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedDataInfo.pickle'
scaler_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle'
with open(simulation_data_file , 'rb') as handle:
    ls_data = pickle.load(handle)
# with open(simulation_datainfo_file , 'rb') as handle:
#     dict_data_info = pickle.load(handle)
with open(scaler_file , 'rb') as handle:
    dict_Scaler = pickle.load(handle)

# Generate n-step predictions for all the curves
ls_data_pred =[]
for data_i in ls_data:
    dict_data_i = {}
    # Generate the initial condition and the true dataset
    x0 = data_i['XT'][0:1,:]
    XT_true = data_i['XT']
    YT_true = data_i['YT']
    # Scale the x0
    x0s = dict_Scaler['XT'].transform(x0)
    XTs_true = dict_Scaler['XT'].transform(XT_true)
    psiXT_true = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: XTs_true})
    # Do 1-step prediction on the initial condition
    psiXT_est_1step = np.concatenate([psiXT_true[0:1],psiXT_true[1:] @ dict_model['KxT_num']], axis=0)
    # Do n-step prediction on the initial condition
    psiXT_est = dict_model['psixpT'].eval(feed_dict = {dict_model['xpT_feed']: x0s})
    for step_i in range(XT_true.shape[0]-1):
        psiXT_est = np.concatenate([psiXT_est, psiXT_est[-1:,:] @ dict_model['KxT_num']],axis=0)
    YTs_est = psiXT_est @ dict_model['WhT_num']
    YTs_est_1step = psiXT_est_1step @ dict_model['WhT_num']
    XTs_est = psiXT_est[:,0:x0.shape[1]]
    XTs_est_1step = psiXT_est_1step[:, 0:x0.shape[1]]
    XT_est = dict_Scaler['XT'].inverse_transform(XTs_est)
    XT_est_1step = dict_Scaler['XT'].inverse_transform(XTs_est_1step)
    YT_est = dict_Scaler['YT'].inverse_transform(YTs_est)
    YT_est_1step = dict_Scaler['YT'].inverse_transform(YTs_est_1step)
    dict_data_i = {'XT_true': XT_true, 'XT_est': XT_est, 'YT_true': YT_true, 'YT_est': YT_est, 'psiXT_true': psiXT_true, 'psiXT_est': psiXT_est, 'XTs_true': XTs_true, 'XTs_est': XTs_est, 'XT_est_1step': XT_est_1step, 'YT_est_1step': YT_est_1step}
    ls_data_pred.append(dict_data_i)

tf.reset_default_graph()
sess.close()

# Generate the prediction stats
XT_all = np.empty((0,ls_data_pred[0]['XT_true'].shape[1]))
YT_all = np.empty((0,ls_data_pred[0]['YT_true'].shape[1]))
XT_est_all = np.empty((0,ls_data_pred[0]['XT_true'].shape[1]))
YT_est_all = np.empty((0,ls_data_pred[0]['YT_true'].shape[1]))
XT_est_all_1step = np.empty((0,ls_data_pred[0]['XT_true'].shape[1]))
YT_est_all_1step = np.empty((0,ls_data_pred[0]['YT_true'].shape[1]))
for data_i in ls_data_pred:
    XT_all = np.concatenate([XT_all, data_i['XT_true']], axis=0)
    YT_all = np.concatenate([YT_all, data_i['YT_true']], axis=0)
    XT_est_all = np.concatenate([XT_est_all, data_i['XT_est']], axis=0)
    YT_est_all = np.concatenate([YT_est_all, data_i['YT_est']], axis=0)
    XT_est_all_1step = np.concatenate([XT_est_all_1step, data_i['XT_est_1step']], axis=0)
    YT_est_all_1step = np.concatenate([YT_est_all_1step, data_i['YT_est_1step']], axis=0)

print('Number of nonlinear observables :', psiXT_true.shape[1] - XT_true.shape[1] -1)
print('r2 score of 1-step prediction in X : ', r2_score(XT_all, XT_est_all_1step, multioutput='uniform_average'))
print('r2 score of 1-step prediction in Y : ', r2_score(YT_all, YT_est_all_1step, multioutput='uniform_average'))
print('r2 score of n-step prediction in X : ', r2_score(XT_all, XT_est_all, multioutput='uniform_average'))
print('r2 score of n-step prediction in Y : ', r2_score(YT_all, YT_est_all, multioutput='uniform_average'))

##
# data_sim_index = 5
# data_sim_index = 10
data_sim_index = 20
# data_sim_index = 30
# data_sim_index = 40
# data_sim_index = 60
# t = dict_data_info['t']
t = np.arange(0,100,1)
DS_FACTOR = 8
MARKERSIZE = 12
# ls_colors = ['#4DBBD5B2', '#E64B35B2', '#00A087B2', '#3C5488B2', '#F39B7FB2', '#8491B4B2', '#91D1C2B2', '#DC0000B2', '#7E6148B2']
# ls_colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499', '#44AA99', '#999933', '#882255', '#661100', '#661100', '#6699CC', '#888888']
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#332288', '#bcbd22', '#17becf', '#88CCEE', '#117733', '#888888']
f,ax = plt.subplots(2,1,figsize = (10,10), sharex=True)

for i in range(11):
    ax[0].plot([], linewidth= 3, label = '$x_{' +str(i+1) + '}$', color=ls_colors[i])
ax[0].plot([], linewidth= 3, label = '$y$', color=ls_colors[-1])
# for i in range(4):
#     ax[1].plot([], linewidth= 5, label = '$x_{' +str(i+8) + '}$', color=ls_colors[i])
ax[1].plot([], '.', markersize=MARKERSIZE, color=ls_colors[-1], label = 'Data')
ax[1].plot([], '--', color=ls_colors[-1], linewidth = 2, label = '1-step prediction')
ax[1].plot([], color=ls_colors[-1], linewidth = 2, label = 'n-step prediction')
for i in range(11):
    ax[0].plot(t[0:-1:DS_FACTOR], ls_data_pred[data_sim_index]['XT_true'][:, i][0:-1:DS_FACTOR], '.',
               markersize=MARKERSIZE, color=ls_colors[i])
    ax[0].plot(t, ls_data_pred[data_sim_index]['XT_est'][:, i], color=ls_colors[i])
    ax[0].plot(t, ls_data_pred[data_sim_index]['XT_est_1step'][:, i], '--', color=ls_colors[i])

ax[1].plot(t[0:-1:DS_FACTOR],ls_data_pred[data_sim_index]['YT_true'][:, 0][0:-1:DS_FACTOR]/ls_data_pred[data_sim_index]['YT_true'][0, 0], '.', markersize=MARKERSIZE, color=ls_colors[-1])
ax[1].plot(t,ls_data_pred[data_sim_index]['YT_est'][:, 0]/ls_data_pred[data_sim_index]['YT_true'][0, 0], color=ls_colors[-1])
ax[1].plot(t, ls_data_pred[data_sim_index]['YT_est_1step'][:, 0]/ls_data_pred[data_sim_index]['YT_est_1step'][0, 0], '--',color=ls_colors[-1])
ax[0].legend(ncol=6, fontsize = 16, loc = 'upper center')
ax[1].legend(ncol=1, fontsize = 16, loc = 'upper right')
ax[1].set_xlim([-0.5,100.])
ax[0].set_ylim([-0.1,3.3])
ax[0].set_yticks([0,1,2,3])
# ax[1].legend(ncol=1, fontsize = 14)
f.show()




##
# data_sim_index = 5
# data_sim_index = 10
data_sim_index = 20
# data_sim_index = 30
# data_sim_index = 40
# data_sim_index = 60
# t = dict_data_info['t']
t = np.arange(0,100,1)
DS_FACTOR = 8
MARKERSIZE = 12
# ls_colors = ['#4DBBD5B2', '#E64B35B2', '#00A087B2', '#3C5488B2', '#F39B7FB2', '#8491B4B2', '#91D1C2B2', '#DC0000B2', '#7E6148B2']
# ls_colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499', '#44AA99', '#999933', '#882255', '#661100', '#661100', '#6699CC', '#888888']
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#332288', '#bcbd22', '#17becf', '#88CCEE', '#117733', '#888888']
f,ax = plt.subplots(2,1,figsize = (7,8), sharex=True)

for i in range(11):
    ax[0].plot([], linewidth= 3, label = '$x_{' +str(i+1) + '}$', color=ls_colors[i])
ax[0].plot([], linewidth= 3, label = '$y$', color=ls_colors[-1])
# for i in range(4):
#     ax[1].plot([], linewidth= 5, label = '$x_{' +str(i+8) + '}$', color=ls_colors[i])
ax[1].plot([], '.', markersize=MARKERSIZE, color=ls_colors[-1], label = 'Data')
ax[1].plot([], '--', color=ls_colors[-1], linewidth = 2, label = '1-step\nprediction')
ax[1].plot([], color=ls_colors[-1], linewidth = 2, label = 'n-step\nprediction')
for i in range(11):
    ax[0].plot(t[0:-1:DS_FACTOR], ls_data_pred[data_sim_index]['XT_true'][:, i][0:-1:DS_FACTOR], '.',
               markersize=MARKERSIZE, color=ls_colors[i])
    ax[0].plot(t, ls_data_pred[data_sim_index]['XT_est'][:, i], color=ls_colors[i])
    ax[0].plot(t, ls_data_pred[data_sim_index]['XT_est_1step'][:, i], '--', color=ls_colors[i])
    # if i<7:
    #     ax[0].plot(t[0:-1:DS_FACTOR],ls_data_pred[data_sim_index]['XT_true'][:,i][0:-1:DS_FACTOR], '.', markersize=MARKERSIZE, color = ls_colors[i])
    #     ax[0].plot(t,ls_data_pred[data_sim_index]['XT_est'][:, i], color = ls_colors[i])
    #     ax[0].plot(t, ls_data_pred[data_sim_index]['XT_est_1step'][:, i], '--',color=ls_colors[i])
    # else:
    #     ax[1].plot(t[0:-1:DS_FACTOR],ls_data_pred[data_sim_index]['XT_true'][:, i][0:-1:DS_FACTOR], '.', markersize=MARKERSIZE, color=ls_colors[i-7])
    #     ax[1].plot(t,ls_data_pred[data_sim_index]['XT_est'][:, i], color=ls_colors[i-7])
    #     ax[1].plot(t, ls_data_pred[data_sim_index]['XT_est_1step'][:, i], '--', color=ls_colors[i-7])
ax[1].plot(t[0:-1:DS_FACTOR],ls_data_pred[data_sim_index]['YT_true'][:, 0][0:-1:DS_FACTOR]/ls_data_pred[data_sim_index]['YT_true'][0, 0], '.', markersize=MARKERSIZE, color=ls_colors[-1])
ax[1].plot(t,ls_data_pred[data_sim_index]['YT_est'][:, 0]/ls_data_pred[data_sim_index]['YT_true'][0, 0], color=ls_colors[-1])
ax[1].plot(t, ls_data_pred[data_sim_index]['YT_est_1step'][:, 0]/ls_data_pred[data_sim_index]['YT_est_1step'][0, 0], '--',color=ls_colors[-1])
ax[0].legend(ncol=4, fontsize = 16, loc = 'upper center')
ax[1].legend(ncol=1, fontsize = 16, loc = 'upper right')
ax[0].set_ylim([-0.1,4.3])
ax[0].set_yticks([0,1,2,3])
# ax[1].legend(ncol=1, fontsize = 14)
f.show()


## SAME AS ABOVE - But individual plots
# # f,ax = plt.subplots(1,3,figsize=(9,3))
# f,ax = plt.subplots(3,4,figsize=(16,12))
# ax = ax.reshape(-1)
# for i in range(x0.shape[1]):
#     ax[i].plot(ls_data_pred[data_sim_index]['XT_true'][:,i], '.', linewidth=3, color='tab:blue')
#     ax[i].plot(ls_data_pred[data_sim_index]['XT_est'][:,i], color='tab:blue')
#     ax[i].set_title('$x_{'+str(i+1)+'}$')
# ax[-1].plot(ls_data_pred[data_sim_index]['YT_true'][:,0], '.', linewidth=3, color='tab:blue')
# ax[-1].plot(ls_data_pred[data_sim_index]['YT_est'][:,0], color='tab:blue')
# ax[-1].set_title('$y$')
# f.show()
# # plt.figure()
# # plt.plot(ls_data_pred[data_sim_index]['XT_true'], '.', linewidth=3, color='tab:blue')
# # plt.plot(ls_data_pred[data_sim_index]['XT_est'], color='tab:blue')
# # plt.show()

##
nL = psiXT_true.shape[1]
WhT = dict_model['WhT_num']
KT = dict_model['KxT_num']
# Construct the observability matrix
OT = np.empty((WhT.shape[0],0))
for i in range(nL):
    OT = np.concatenate([OT, np.linalg.matrix_power(KT,i) @ WhT], axis=1)
O = OT.T
sb.heatmap(O,cmap='Blues')
plt.title('Original observability matrix')
plt.show()

# Decomposition of the observability matrix
U,S,VT = np.linalg.svd(O)
V = VT.T
plt.plot(np.arange(1,1+len(S)),100 - np.cumsum(S**2)/np.sum(S**2)*100)
plt.title('Scree plot of singular values \n of Observability matrix')
plt.show()

T = VT
K = KT.T
Wh = WhT.T

Ka = T @ K @ np.linalg.inv(T)
Wha = Wh @ np.linalg.inv(T)

f,ax = plt.subplots(1,2, figsize=(10,5))
sb.heatmap(Ka,cmap='Blues', ax=ax[1])
sb.heatmap(K,cmap='Blues', ax=ax[0])
ax[0].set_title('Original K')
ax[1].set_title('Linear observable \n decomposition of K')
f.show()


f,ax = plt.subplots(1,2, figsize=(10,5))
ax[0] = sb.heatmap(Wh,cmap='Blues', ax=ax[0])
ax[1] = sb.heatmap(Wha,cmap='Blues', ax=ax[1])
ax[0].set_title('Original Wh')
ax[1].set_title('Linear observable \n decomposition of Wh')
plt.show()

## Contribution of the observable dynamics to each gene
sess = tf.InteractiveSession()
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)

ls_labels = []
for i in range(ls_data_pred[0]['XT_true'].shape[1]):
    ls_labels.append('$x_{'+str(i+1)+'}$')
for i in range(ls_data_pred[0]['psiXT_true'].shape[1] - ls_data_pred[0]['XT_true'].shape[1]):
    ls_labels.append('$\\varphi_{'+ str(1+ i) + '}(x)$')


ls_nPC = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
ls_output_accuracy = []
n_cols = np.int(np.ceil(np.sqrt(len(ls_nPC))))
n_rows = np.int(np.ceil(len(ls_nPC)/n_cols))

f,ax = plt.subplots(nrows=n_rows,ncols=n_cols, sharex=True, sharey= True, figsize=(n_cols*15, n_rows*8))
ax = ax.reshape(-1)
print('r2 score of n-step prediction in Y : ')
for i in range(len(ls_nPC)):
    nPC = ls_nPC[i]
    Ko = Ka[0:nPC,0:nPC]
    Who = Wha[:,0:nPC]
    Tinv = np.linalg.inv(T)
    ls_transformed_data = []
    YT_actual = YT_pred = psiXT = psiXT_o = psiXT_hat = []
    for data_i in ls_data_pred:
        XTi = data_i['XT_est']
        XTis = dict_Scaler['XT'].transform(XTi)
        YTis = dict_Scaler['YT'].transform(data_i['YT_est'])
        psiXTi = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: XTis})
        psiXTi_ou = psiXTi @ T.T
        psiXTi_o = psiXTi_ou[:,0:nPC]
        # n - step prediction
        psiXTi_o_nstep = psiXTi_o[0:1]
        for j in range(psiXTi_o.shape[0]-1):
            psiXTi_o_nstep = np.concatenate([psiXTi_o_nstep, psiXTi_o_nstep[-1:] @ Ko.T], axis=0)
        YTi_o_nstep = psiXTi_o_nstep @ Who.T
        psiXTi_hat = psiXTi_o @ Tinv[:,0:nPC].T
        try:
            psiXT = np.concatenate([psiXT, psiXTi], axis=0)
            psiXT_o = np.concatenate([psiXT_o, psiXTi_o], axis=0)
            psiXT_hat = np.concatenate([psiXT_hat, psiXTi_hat], axis=0)
            YT_pred = np.concatenate([YT_pred, YTi_o_nstep], axis=0)
            YT_actual = np.concatenate([YT_actual, YTis], axis=0)
        except:
            psiXT = psiXTi
            psiXT_o = psiXTi_o
            psiXT_hat = psiXTi_hat
            YT_pred = YTi_o_nstep
            YT_actual = YTis
    ls_output_accuracy.append(r2_score(YT_actual, YT_pred, multioutput='uniform_average'))
    np_predictability = np.maximum(0, r2_score(psiXT, psiXT_hat, multioutput='raw_values') * 100)
    ax[i].bar(ls_labels[0:-1], np_predictability[:-1], color='tab:blue')
    # ax[i].set_xlabel('State number')
    if np.mod(i,n_cols) ==0:
        ax[i].set_ylabel('% of captured state dynamics')
    ax[i].set_title(str(nPC) + ' states in $\psi_o(x)$')
    ax[i].set_ylim([0, 100])
    print('# states : ', ls_nPC[i], ' r2 :', ls_output_accuracy[i])
f.show()
tf.reset_default_graph()
sess.close()
##
plt.figure(figsize =(8,3))
ax9 = plt.subplot(111)
MAX_NPC = 20
ax9.bar(ls_nPC[0:MAX_NPC], np.maximum(0,np.array(ls_output_accuracy)*100)[0:MAX_NPC], color = '#CDE0F1')
ax9.plot(ls_nPC[0:MAX_NPC], np.maximum(0,np.array(ls_output_accuracy)*100)[0:MAX_NPC], color='tab:blue', marker = '.', markersize=10, linestyle = '--')
ax9.plot(ls_nPC[10], np.maximum(0,np.array(ls_output_accuracy[10])*100), '*', color='tab:blue', markersize=20)
ax9.set_xlim([0.5,20.5])
ax9.set_ylim([0, 105])
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)
# plt.xlabel('Dimension of $\\psi_o(x)$')
# plt.ylabel('$\hat{y}$ reconcstruction \n accuracy $(\%)$')
plt.show()


## Plot the transformation matrix
nPC = 11

ls_xticks = []
for i in range(ls_data_pred[0]['XT_true'].shape[1]):
    ls_xticks.append('$x_{' +str(i+1)+'}$')
for i in range(ls_data_pred[0]['psiXT_true'].shape[1] - ls_data_pred[0]['XT_true'].shape[1]-1):
    ls_xticks.append('$\\varphi_{' + str(i+1) + '}$')
ls_xticks.append('$1$')

x_tick_pos = np.arange(0.5,ls_data_pred[0]['psiXT_true'].shape[1])

ls_yticks = []
for items in range(nPC):
    ls_yticks.append('$\psi_{o' +str(items+1)+'}$')
y_tick_pos = np.arange(nPC-0.5,0,-1)

# Simpler plot of the below one
# plt.figure(figsize=(10,8))
# sb.heatmap(T[0:nPC,:], cmap='RdBu')
# plt.ylim([0,nPC])
# plt.yticks(np.arange(nPC-0.5,0,-1),ls_yticks)
# plt.xticks(np.arange(0.5,ls_data_pred[0]['psiXT_true'].shape[1] ),ls_xticks)
# plt.show()



f,((ax0,dummy_ax),(ax1,cbar_ax1)) = plt.subplots(2,2,figsize=(16,10), sharex='col', gridspec_kw={'height_ratios': [1, 5], 'width_ratios': [20, 1]})
sb.heatmap(T[0:nPC,:], cmap='RdBu',ax=ax1, cbar_ax=cbar_ax1, vmax=0.8, vmin=-0.8)

ax1.set_xticks(x_tick_pos)
ax1.set_xticklabels(ls_xticks)
ax1.set_yticks(y_tick_pos)
ax1.set_yticklabels(ls_yticks)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

# Energy plot of psi_i(x) contributing to

ax0.bar(x_tick_pos, np.linalg.norm(T[0:nPC,:], axis=0, ord=2))
ax0.set_ylim([0,1])
# ax0.set_title('Energy of each observable')
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
dummy_ax.axis('off')
plt.tight_layout()
f.show()


## Sensitivity analysis of nonlinear observables

sess = tf.InteractiveSession()
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)

XTs_all = dict_Scaler['XT'].transform(XT_all)
sensitivity_matrix = np.empty((0,ls_data_pred[0]['XT_true'].shape[1]))
for i in range(ls_data_pred[0]['psiXT_true'].shape[1]):
    func = dict_model['psixpT'][:,i:i+1]
    func_grad = tf.gradients(func, dict_model['xpT_feed'])
    sensitivity_all_points_for_func = func_grad[0].eval(feed_dict={dict_model['xpT_feed']: XTs_all})
    sensitivity_matrix = np.concatenate([sensitivity_matrix ,np.max(sensitivity_all_points_for_func,axis=0).reshape(1,-1)],axis=0)

plt.figure(figsize=(10,10))
ax = sb.heatmap(sensitivity_matrix, cmap='Blues')
plt.yticks(np.arange(0.5,ls_data_pred[0]['psiXT_true'].shape[1]),ls_xticks)
plt.xticks(np.arange(0.5,ls_data_pred[0]['XT_true'].shape[1]),ls_xticks[0:ls_data_pred[0]['XT_true'].shape[1]])
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

tf.reset_default_graph()
sess.close()

## Sensitivity of the observation states with respect to the base states
nPC = 11

sess = tf.InteractiveSession()
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)
T_tensor = tf.constant(T[0:nPC,:], dtype=tf.float32)
psi_oT_tensor = tf.matmul(dict_model['psixpT'], tf.transpose(T_tensor))
psi_o_sensitivity_matrix = np.empty((0,ls_data_pred[0]['XT_true'].shape[1]))
for i in range(nPC):
    func = psi_oT_tensor[:,i:i+1]
    func_grad = tf.gradients(func, dict_model['xpT_feed'])
    sensitivity_all_points_for_func = func_grad[0].eval(feed_dict={dict_model['xpT_feed']: XTs_all})
    psi_o_sensitivity_matrix= np.concatenate([psi_o_sensitivity_matrix ,np.max(sensitivity_all_points_for_func,axis=0).reshape(1,-1)],axis=0)

ls_psi_o_labels = []
for i in range(nPC):
    ls_psi_o_labels.append('$\psi_{o'+str(i+1)+ '}$')

# plt.figure(figsize=(15,10))
# ax = sb.heatmap(psi_o_sensitivity_matrix, cmap='Blues', annot= True)
# plt.yticks(np.arange(0.5,nPC),ls_psi_o_labels,rotation=0)
# plt.xticks(np.arange(0.5,ls_data_pred[0]['XT_true'].shape[1]),ls_xticks[0:ls_data_pred[0]['XT_true'].shape[1]])
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()
#
# tf.reset_default_graph()
# sess.close()
#
# plt.figure(figsize=(15,10))
# ax = sb.heatmap(psi_o_sensitivity_matrix, cmap='Blues', annot= True)
# plt.yticks(np.arange(0.5,nPC),ls_psi_o_labels,rotation=0)
# plt.xticks(np.arange(0.5,ls_data_pred[0]['XT_true'].shape[1]),ls_xticks[0:ls_data_pred[0]['XT_true'].shape[1]])
# bottom, top = ax.get_ylim()
# ax.set_ylim(bottom + 0.5, top - 0.5)
# plt.show()

#
f,((ax0,dummy_ax),(ax1,cbar_ax1)) = plt.subplots(2,2,figsize=(14,10), sharex='col', gridspec_kw={'height_ratios': [2, 4], 'width_ratios': [20, 1]})
sb.heatmap(psi_o_sensitivity_matrix, cmap='Blues',ax=ax1, cbar_ax=cbar_ax1)

x_tick_pos = np.arange(0.5,ls_data_pred[0]['XT_true'].shape[1])

ax1.set_xticks(x_tick_pos)
ax1.set_xticklabels(ls_xticks[0:ls_data_pred[0]['XT_true'].shape[1]])
ax1.set_yticks(np.arange(0.5,nPC))
ax1.set_yticklabels(ls_psi_o_labels,rotation=0)
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)

# Energy plot of psi_i(x) contributing to

ax0.bar(x_tick_pos, np.linalg.norm(psi_o_sensitivity_matrix, axis=0, ord=2))
# ax0.set_title('Energy of each observable')
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
dummy_ax.axis('off')
plt.tight_layout()
f.show()

tf.reset_default_graph()
sess.close()

## Relative fitness plot of single gene knockouts
data_sim_index = 10
sess = tf.InteractiveSession()
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)


dict_gene_knockout_growth = {}
for gene_i in range(ls_data[data_sim_index]['XT'].shape[1]+1):
    XT0_i = ls_data[data_sim_index]['XT'][0:1]
    XTs_i = dict_Scaler['XT'].transform(XT0_i)
    if gene_i >0:
        XTs_i[0,gene_i-1] = 0
    psiXT_i = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: XTs_i})
    for t_i in range(ls_data[data_sim_index]['XT'].shape[0]-1):
        psiXT_i = np.concatenate([psiXT_i, psiXT_i[-1:] @ dict_model['KxT_num'] ])
        if gene_i > 0:
            psiXT_i[-1, gene_i - 1] = 0
    yhats_i = psiXT_i @ dict_model['WhT_num']
    yhat_i = dict_Scaler['YT'].inverse_transform(yhats_i)
    if gene_i ==0:
        dict_label = 'NC'
    else:
        dict_label = 'gene_' + str(gene_i)
    dict_gene_knockout_growth[dict_label] = yhat_i

#
plt.figure(figsize=(10,4))
for keys in dict_gene_knockout_growth:
    diff_growth = np.log2(dict_gene_knockout_growth[keys]/dict_gene_knockout_growth['NC'])
    plt.plot(diff_growth,label=keys)
plt.ylim([-0.15,0.1])
plt.legend(ncol=3, fontsize=11)
plt.show()





##

## ===========     EXTRAS      ===========

## Plotting the observed states, the actual states and the nonlinear observables
# f,ax = plt.subplots(1,5,figsize=(25,5))
# for i in range(5):
#     ax[i].plot(psiXTi_o[:,i])
#     ax[i].set_ylim([-2, 8])
# ax[2].set_title('psi_o')
# f.show()
#
# n_lifted_obs = psiXTi.shape[1] - XTis.shape[1]
# f,ax = plt.subplots(1,n_lifted_obs,figsize=(n_lifted_obs*5,5))
# for i in range(n_lifted_obs):
#     ax[i].plot(psiXTi[:,XTis.shape[1] + i])
#     ax[i].set_ylim([-2, 8])
# ax[2].set_title('varphi(x)')
# f.show()
#
#
# f,ax = plt.subplots(3,4,figsize=(16,9))
# ax = ax.reshape(-1)
# for i in range(x0.shape[1]):
#     ax[i].plot(XTis[:,i], color='tab:blue')
#     ax[i].set_title('x_'+str(i+1))
#     ax[i].set_ylim([-2,8])
# ax[-1].plot(ls_data[-1]['YT'][:,0], color='tab:blue')
# ax[-1].set_title('y')
# f.show()


## Code to make the network for this system


# from pyvis.network import Network
# import networkx as nx
#
# np_G = np.zeros((12,12))
# ls_edges = [ [1,3] , [2,3] ,[2,4] ,[2,6] ,[3,4] ,[3,5] ,[3,7] ,[4,6] ,[4,7] ,[5,6] ,[7,8] ,[7,9] ,[7,10] ,[7,11] ,[8,12] ,[11,12]]
# for edge_i in ls_edges:
#     np_G[edge_i[0]-1, edge_i[1]-1]=1
#     np_G[edge_i[1] - 1, edge_i[0] - 1] = 1
# ls_nodes = ['$x_{' + str(i) + '}$' for i in range(1,12)]
# ls_nodes.append('y')
# df_G = pd.DataFrame(np_G, columns=ls_nodes, index= ls_nodes)
#
# G = nx.from_pandas_adjacency(df_G)
# net = Network(notebook=True)
# net.from_nx(G)
# net.show("example.html")


## GENE KNOCKOUT
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
a = np.ones(11)
def ccdA_ccdB_Antitoxin_Toxin_system_knockout(x,t):
    xdot = np.zeros(len(x))
    xdot[0] = a[0]*(-k1f*x[0]*x[1] + k1i*x[2]) - gamma_1*x[0]
    xdot[1] = a[1]*(-k1f*x[0]*x[1] - k2f*x[1]*x[2] - k5f*x[1]*x[4] + k1i*x[2] + k2i*x[3] +k5i*x[5]) - gamma_2*x[1]
    xdot[2] = a[2]*(k1f*x[0]*x[1] - k2f*x[1]*x[2] - k1i*x[2] + k2i*x[3] - k4f*x[2] +k4i*x[4]*x[6]) - gamma_3*x[2]
    xdot[3] = a[3]*(k2f*x[1]*x[2] - k2i*x[3] +k3i*x[5]*x[6] -k3f*x[3]) - gamma_4*x[3]
    xdot[4] = a[4]*(k4f*x[2] -k4i*x[4]*x[6] -k5f*x[4] + k5i*x[1]*x[5]) - gamma_5*x[4]
    xdot[5] = a[5]*(k5f*x[1]*x[4] - k5i*x[5] +k3f*x[3] -k3i*x[5]*x[6]) - gamma_6*x[5]
    xdot[6] = a[6]*(k4f*x[2] -k4i*x[4]*x[6] + k3f*x[3] - k3i*x[5]*x[6]) - gamma_7*x[6]
    xdot[7] = a[7]*(a1 * (x[6] / K1) ** n1 / (1 + (x[6] / K1) ** n1)) - d1*x[7]
    xdot[8] = a[8]*(a2 * (x[6] / K2) ** n2 / (1 + (x[6] / K2) ** n2)) - d2*x[8]
    xdot[9] = a[9]*(a3 * (x[6] / K3) ** n3 / (1 + (x[6] / K3) ** n3)) - d3*x[9]
    xdot[10] = a[10]*(a4 * (x[6] / K4) ** n4 / (1 + (x[6] / K4) ** n4)) - d4*x[10]
    return xdot

simulation_time = 100
sampling_time = 1
t = np.arange(0, simulation_time, sampling_time)

x0 = np.array([0.4,0.1,0.2,0.4,0.3,0.8,0.5])
x0_proteins = np.array([0.3,0.8,0.1,1.8])
y0 = 0.02
x0 = np.concatenate([x0,x0_proteins],axis=0)
XT_nc = odeint(ccdA_ccdB_Antitoxin_Toxin_system_knockout,x0,t)
YT_nc = y0*np.exp(mu_y*(XT_nc[:,7]/Kya)**1/(1 + (XT_nc[:,7]/Kya)**1 + (XT_nc[:,10]/Kyr)**2))

YT_knockout = np.zeros((11, len(t)))

for i in range(11):
    x0_i = copy.deepcopy(x0)
    x0_i[i] = 0
    a[i] = 0
    XT_i = odeint(ccdA_ccdB_Antitoxin_Toxin_system_knockout,x0_i,t)
    YT_i = y0*np.exp(mu_y*(XT_i[:,7]/Kya)**1/(1 + (XT_i[:,7]/Kya)**1 + (XT_i[:,10]/Kyr)**2))
    YT_knockout[i,:] = np.log2(YT_i/YT_nc)
    a[i] = 1
plt.figure(figsize=(15,6))
plt.plot(YT_knockout[0:7,:].T)
plt.legend(['$x_{'+str(i+1)+'}$(' + str(np.round(np.sum(np.abs(YT_knockout[i,:])),2)) + ')' for i in range(11)], ncol=3, loc = 'upper left')
plt.show()


# TODO - YET TO COMPLETE Single gene knockout

# SYSTEM_NO = 5
# RUN_NO = 6

sess = tf.InteractiveSession()
run_folder_name = 'System_' + str(SYSTEM_NO) + '/MyMac/RUN_' + str(RUN_NO)
dict_model = get_dict_param(run_folder_name,SYSTEM_NO,sess)

# Load the data
simulation_data_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle'
# simulation_datainfo_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_SimulatedDataInfo.pickle'
scaler_file = 'System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle'
with open(simulation_data_file , 'rb') as handle:
    ls_data = pickle.load(handle)
# with open(simulation_datainfo_file , 'rb') as handle:

#     dict_data_info = pickle.load(handle)
with open(scaler_file , 'rb') as handle:
    dict_Scaler = pickle.load(handle)

t = np.arange(0, 100, 1)

x0 = np.array([0.4, 0.1, 0.2, 0.4, 0.3, 0.8, 0.5])
x0_proteins = np.array([0.3, 0.8, 0.1, 1.8])
y0 = 0.02
x0 = np.concatenate([x0, x0_proteins], axis=0).reshape(1,-1)
XTs_nc = dict_Scaler['XT'].transform(x0)
YT = 0
for gene_i in range(12):
    x0_i = copy.deepcopy(x0)
    if gene_i <11:
        x0_i[0,gene_i]= 0
    x0s = dict_Scaler['XT'].transform(x0_i)
    psiXT = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: x0s})
    for i in range(len(t)):
        psiXTi = psiXT[-1:,:]@dict_model['KxT_num']
        xTsi = psiXTi[:,0:11]
        if gene_i<11:
            xTsi[0, gene_i] = 0
        psiXTihat = dict_model['psixpT'].eval(feed_dict={dict_model['xpT_feed']: xTsi})
        psiXT = np.concatenate([psiXT,psiXTihat], axis=0)
        # psiXT = np.concatenate([psiXT, psiXTi], axis=0)
    ys = psiXT@dict_model['WhT_num']
    try:
        YT = np.concatenate([YT,dict_Scaler['YT'].inverse_transform(ys).T],axis=0)
    except:
        YT = dict_Scaler['YT'].inverse_transform(ys).T

YT_knockout_model = np.log2(YT[0:11,:]/YT[11,:])
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#332288', '#bcbd22', '#17becf', '#88CCEE', '#117733', '#888888']
plt.figure(figsize=(15,6))
for i in range(11):
    plt.plot(YT_knockout_model[i,:], color = ls_colors[i], label = '$x_{'+str(i+1)+'}$(' + str(np.round(np.sum(np.abs(YT_knockout_model[i,:])),2)) + ')')
plt.legend(ncol=3, loc = 'upper left')
plt.show()