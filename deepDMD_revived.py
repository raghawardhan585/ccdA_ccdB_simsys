##
# ALWAYS CONSIDERING WITH OUTPUT and STATE

# Required Packages
import pickle  # for data I/O
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy.linalg import pinv # for the least squares approach
import math;
import random;
import tensorflow as tf;
import os
import shutil
import pandas as pd

DEVICE_NAME = '/cpu:0'
SYSTEM_NO = 5

# RUN_NUMBER = 1
# max_epochs = 2000
# train_error_threshold = 1e-6
# valid_error_threshold = 1e-6;
# test_error_threshold = 1e-6;

#  Deep Learning Optimization Parameters ##

activation_flag = 1;  # sets the activation function type to RELU[0], ELU[1], SELU[2] (initialized a certain way,dropout has to be done differently) , or tanh()

DISPLAY_SAMPLE_RATE_EPOCH = 1000
TRAIN_PERCENT = 85.71429
keep_prob = 1.0;  # keep_prob = 1-dropout probability
res_net = 0;  # Boolean condition on whether to use a resnet connection.

# Neural network parameters

# ---- STATE OBSERVABLE PARAMETERS -------
x_deep_dict_size = 12
n_x_nn_layers = 4 # x_max_layers 3 works well
n_x_nn_nodes = 18 # max width_limit -4 works well

best_test_error = np.inf

# TODO - YET TO INCORPORATE REGULARIZATION
regularization_lambda =0

# Learning Parameters
batch_size = 1000#72#40#24#36
ls_dict_training_params = []
dict_training_params = {'step_size_val': 0.5, 'train_error_threshold': float(1e-20),'valid_error_threshold': float(1e-6), 'max_epochs': 50000, 'batch_size': batch_size} #80000
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.3, 'train_error_threshold': float(1e-10),'valid_error_threshold': float(1e-6), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-7), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.005, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.001, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 50000, 'batch_size': batch_size}
ls_dict_training_params.append(dict_training_params)

sess = tf.InteractiveSession()

# Required Functions

def estimate_K_stability(Kx, print_Kx=False):
    Kx_num = sess.run(Kx)
    np.linalg.eigvals(Kx_num)
    Kx_num_eigval_mod = np.abs(np.linalg.eigvals(Kx_num))
    if print_Kx:
        print(Kx_num)
    print('Eigen values: ')
    print(Kx_num_eigval_mod)
    unstable = True
    if np.max(Kx_num_eigval_mod) > 1:
        print('[COMP] The identified Koopman operator is UNSTABLE with ', np.sum(np.abs(Kx_num_eigval_mod) > 1),
              'eigenvalues greater than 1')
    else:
        print('[COMP] The identified Koopman operator is STABLE')
        unstable = False
    return unstable


def weight_variable(shape):
    std_dev = math.sqrt(3.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def bias_variable(shape):
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def initialize_Wblist(n_u, hv_list):
    # INITIALIZATION - going from input to first layer
    W_list = [weight_variable([n_u, hv_list[0]])]
    b_list = [bias_variable([hv_list[0]])]
    # PROPAGATION - consecutive layers
    for k in range(1,len(hv_list)):
        W_list.append(weight_variable([hv_list[k - 1], hv_list[k]]))
        b_list.append(bias_variable([hv_list[k]]))
    return W_list, b_list
def initialize_tensorflow_graph(param_list):
    # res_net = param_list['res_net'] --- This variable is not used!
    # TODO - remove the above variable if not required at all
    # u is the input of the neural network
    u = tf.placeholder(tf.float32, shape=[None,param_list['n_base_states']]);  # state/input node,# inputs = dim(input) , None indicates batch size can be any size
    z_list = [];
    n_depth = len(param_list['hidden_var_list']);
    # INITIALIZATION
    if param_list['activation flag'] == 1:  # RELU
        z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    if param_list['activation flag']== 2:  # ELU
        z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    if param_list['activation flag'] == 3:  # tanh
        z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    # PROPAGATION & TERMINATION
    for k in range(1, n_depth):
        prev_layer_output = tf.matmul(z_list[k - 1], param_list['W_list'][k]) + param_list['b_list'][k]
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    psiX = tf.concat([u, z_list[-1][:, 0:param_list['x_observables']]], axis=1)
    psiX = tf.concat([psiX, tf.ones(shape=(tf.shape(psiX)[0], 1))], axis=1)
    y = tf.concat([psiX, z_list[-1][:, param_list['x_observables']:]], axis=1)
    result = sess.run(tf.global_variables_initializer())
    return z_list, y, u

def get_variable_value(variable_name, prev_variable_value, reqd_data_type, lower_bound=0):
    # Purpose: This function is mainly to
    not_valid = True
    variable_output = prev_variable_value
    while (not_valid):
        print('Current value of ', variable_name, ' = ', prev_variable_value)
        variable_input = input('Enter new ' + variable_name + ' value [-1 or ENTER to retain previous entry]: ')
        # First check for -1
        if variable_input in ['-1', '']:
            not_valid = False
        else:
            # Second check for correct data type
            try:
                variable_input = reqd_data_type(variable_input)
                # Third check for the correct bound
                if not (variable_input > lower_bound):
                    print('Error! Value is out of bounds. Please enter a value greater than ', lower_bound)
                    not_valid = True
                else:
                    variable_output = variable_input
                    not_valid = False
            except:
                print('Error! Please enter a ', reqd_data_type, ' value, -1 or ENTER')
                not_valid = True
    return variable_output
def display_train_params(dict_run_params):
    print('======================================')
    print('CURRENT TRAINING PARAMETERS')
    print('======================================')
    print('Step Size Value            : ', dict_run_params['step_size_val'])
    print('Train Error Threshold      : ', dict_run_params['train_error_threshold'])
    print('Validation Error Threshold : ', dict_run_params['valid_error_threshold'])
    print('Maximum number of Epochs   : ', dict_run_params['max_epochs'])
    print('Batch Size   : ', dict_run_params['batch_size'])
    print('--------------------------------------')
    return
def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, dict_model_metrics, n_epochs_run, dict_run_params,with_output_train):
    training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
    validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
    training_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_train)
    validation_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_valid)
    dict_hp = {}
    dict_hp['x_hidden_variable_list'] = x_hidden_vars_list
    dict_hp['activation flag'] = activation_flag
    dict_hp['activation function'] = None
    if activation_flag == 1:
        dict_hp['activation function'] = 'relu'
    elif activation_flag == 2:
        dict_hp['activation function'] = 'elu'
    elif activation_flag == 3:
        dict_hp['activation function'] = 'tanh'
    dict_hp['no of epochs'] = n_epochs_run
    dict_hp['batch size'] = dict_run_params['batch_size']
    dict_hp['step size'] = dict_run_params['step_size_val']
    dict_hp['training error'] = training_error
    dict_hp['validation error'] = validation_error
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    dict_hp['r^2 X train accuracy'] = dict_model_metrics['accuracy_X'].eval(feed_dict=feed_dict_train)
    dict_hp['r^2 X valid accuracy'] = dict_model_metrics['accuracy_X'].eval(feed_dict=feed_dict_valid)
    if with_output_train:
        dict_hp['r^2 Y train accuracy'] = dict_model_metrics['accuracy_Y'].eval(feed_dict=feed_dict_train)
        dict_hp['r^2 Y valid accuracy'] = dict_model_metrics['accuracy_Y'].eval(feed_dict=feed_dict_valid)
    return dict_hp

def objective_func(dict_feed,dict_psi,dict_K):
    dict_model_perf_metrics ={}
    psiXf_predicted = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    psiXf_prediction_error = dict_psi['xfT'] - psiXf_predicted
    try:
        # If output is available
        Yf_prediction_error = dict_feed['ypT'] - tf.matmul(dict_psi['xpT'], dict_K['WhT'])
        prediction_error = tf.concat([psiXf_prediction_error,Yf_prediction_error],axis=1)
        all_value = tf.concat([dict_psi['xfT'],dict_feed['ypT']],axis=1)
        # Accuracy computation Y
        SST_Y = tf.math.reduce_sum(tf.math.square(dict_feed['ypT'] - tf.math.reduce_mean(dict_feed['ypT'])))
        SSE_Y = tf.math.reduce_sum(tf.math.square(Yf_prediction_error))
        dict_model_perf_metrics['accuracy_Y'] = (1 - tf.divide(SSE_Y, SST_Y)) * 100
    except:
        prediction_error = psiXf_prediction_error
        all_value = dict_psi['xfT']
    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics ['loss_fn'])
    # Mean Squared Error
    dict_model_perf_metrics ['MSE'] = tf.math.reduce_mean(tf.math.square(prediction_error))

    # Accuracy computation total
    SST = tf.math.reduce_sum(tf.math.square(all_value- tf.math.reduce_mean(all_value, axis=0)))
    SSE = tf.math.reduce_sum(tf.math.square(prediction_error))
    dict_model_perf_metrics['accuracy'] = (1 - tf.divide(SSE, SST)) * 100
    # Accuracy computation X
    SST_X = tf.math.reduce_sum(tf.math.square(dict_psi['xfT'] - tf.math.reduce_mean(dict_psi['xfT'], axis=0)))
    SSE_X = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error))
    dict_model_perf_metrics['accuracy_X'] = (1 - tf.divide(SSE_X, SST_X)) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics

def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics, all_histories = {'train error': [], 'validation error': []}, dict_run_info = {}):
    try:
        # Works if we have an output
        feed_dict_train = {dict_feed['xpT']: dict_train['XpT'],dict_feed['xfT']: dict_train['XfT'],dict_feed['ypT']: dict_train['YpT'],dict_feed['yfT']: dict_train['YfT']}
        feed_dict_valid = {dict_feed['xpT']: dict_valid['XpT'],dict_feed['xfT']: dict_valid['XfT'],dict_feed['ypT']: dict_valid['YpT'],dict_feed['yfT']: dict_valid['YfT']}
        with_output_train = True
    except:
        feed_dict_train = {dict_feed['xpT']: dict_train['XpT'], dict_feed['xfT']: dict_train['XfT']}
        feed_dict_valid = {dict_feed['xpT']: dict_valid['XpT'], dict_feed['xfT']: dict_valid['XfT']}
        with_output_train = False
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_train_params_i, all_histories,with_output_train)
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,dict_model_metrics,n_epochs_run, dict_train_params_i,with_output_train)
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        estimate_K_stability(KxT)
        run_info_index += 1
    return all_histories, dict_run_info

def train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_run_params, all_histories,with_output_train):
    # -----------------------------
    # Initialization
    # -----------------------------
    N_train_samples = len(dict_train['XpT'])
    runs_per_epoch = int(np.ceil(N_train_samples / dict_run_params['batch_size']))
    epoch_i = 0
    training_error = 100
    validation_error = 100
    # -----------------------------
    # Actual training
    # -----------------------------
    if with_output_train:
        while ((epoch_i < dict_run_params['max_epochs']) and (training_error > dict_run_params['train_error_threshold']) and (validation_error > dict_run_params['valid_error_threshold'])):
            epoch_i += 1
            # Re initializing the training indices
            all_train_indices = list(range(N_train_samples))
            # Random sort of the training indices
            random.shuffle(all_train_indices)
            for run_i in range(runs_per_epoch):
                if run_i != runs_per_epoch - 1:
                    train_indices = all_train_indices[run_i * dict_run_params['batch_size']:(run_i + 1) * dict_run_params['batch_size']]
                else:
                    # Last run with the residual data
                    train_indices = all_train_indices[run_i * dict_run_params['batch_size']: N_train_samples]
                feed_dict_train_curr = {dict_feed['xpT']: dict_train['XpT'][train_indices], dict_feed['xfT']: dict_train['XfT'][train_indices],dict_feed['ypT']: dict_train['YpT'][train_indices], dict_feed['yfT']: dict_train['YfT'][train_indices], dict_feed['step_size']: dict_run_params['step_size_val']}
                dict_model_metrics['optimizer'].run(feed_dict=feed_dict_train_curr)
            # After training 1 epoch
            training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
            validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
            all_histories['train error'].append(training_error)
            all_histories['validation error'].append(validation_error)
            if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
                print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
                print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
                # estimate_K_stability(Kx)
                print('---------------------------------------------------------------------------------------------------')
    else:
        while ((epoch_i < dict_run_params['max_epochs']) and (training_error > dict_run_params['train_error_threshold']) and (validation_error > dict_run_params['valid_error_threshold'])):
            epoch_i += 1
            # Re initializing the training indices
            all_train_indices = list(range(N_train_samples))
            # Random sort of the training indices
            random.shuffle(all_train_indices)
            for run_i in range(runs_per_epoch):
                if run_i != runs_per_epoch - 1:
                    train_indices = all_train_indices[run_i * dict_run_params['batch_size']:(run_i + 1) * dict_run_params['batch_size']]
                else:
                    # Last run with the residual data
                    train_indices = all_train_indices[run_i * dict_run_params['batch_size']: N_train_samples]
                feed_dict_train_curr = {dict_feed['xpT']: dict_train['XpT'][train_indices],
                                        dict_feed['xfT']: dict_train['XfT'][train_indices],
                                        dict_feed['step_size']: dict_run_params['step_size_val']}
                dict_model_metrics['optimizer'].run(feed_dict=feed_dict_train_curr)
            # After training 1 epoch
            training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
            validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
            all_histories['train error'].append(training_error)
            all_histories['validation error'].append(validation_error)
            if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
                print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
                print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
                # estimate_K_stability(Kx)
                print('---------------------------------------------------------------------------------------------------')
    return all_histories, epoch_i


def get_best_K_DMD(Xp_train,Xf_train,Xp_valid,Xf_valid):
    Xp_train = Xp_train.T
    Xf_train = Xf_train.T
    Xp_valid = Xp_valid.T
    Xf_valid = Xf_valid.T
    U,S,Vh = np.linalg.svd(Xp_train)
    V = Vh.T.conj()
    Uh = U.T.conj()
    A_hat = np.zeros(shape = U.shape)
    ls_error_train = []
    ls_error_valid = []
    for i in range(len(S)):
        A_hat = A_hat + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
        ls_error_train.append(np.mean(np.square((Xf_train - np.matmul(A_hat,Xp_train)))))
        ls_error_valid.append(np.mean(np.square((Xf_valid - np.matmul(A_hat, Xp_valid)))))
    ls_error = np.array(ls_error_train) + np.array(ls_error_valid)
    nPC_opt = np.where(ls_error==np.min(ls_error))[0][0] + 1
    A_hat_opt = np.zeros(shape = U.shape)
    for i in range(nPC_opt):
        A_hat_opt = A_hat_opt + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
    print('Optimal Linear model Error: ',np.mean(np.square((Xf_train - np.matmul(A_hat_opt, Xp_train)))))
    return  A_hat_opt.T

# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================




## Main Block
STATE_RUN_NUMBER_PROCURED = False

# import sys
# if len(sys.argv)>1:
#     DEVICE_NAME = sys.argv[1]
#     if DEVICE_NAME not in ['/cpu:0','/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
#         DEVICE_NAME = '/cpu:0'
# if len(sys.argv)>2:
#     SYSTEM_NO = sys.argv[2]
# if len(sys.argv) > 3:
#     RUN_NUMBER = np.int(sys.argv[3])
#     STATE_RUN_NUMBER_PROCURED = True
# if len(sys.argv) > 4:
#     x_deep_dict_size = np.int(sys.argv[4])
# if len(sys.argv)>5:
#     n_x_nn_layers = np.int(sys.argv[5])
# if len(sys.argv)>6:
#     n_x_nn_nodes = np.int(sys.argv[6])


# data_directory = os.path.normpath(os.getcwd() + os.sep + os.pardir) +'/koopman_data/'
data_directory = 'System_' + str(SYSTEM_NO)
data_suffix = 'System_' + str(SYSTEM_NO) + '_DeepDMDdata_Scaled.pickle'
file_path = data_directory + '/' +data_suffix
with open(file_path, 'rb') as handle:
    dict_data = pickle.load(handle)

dict_train = {'XpT': dict_data['XpT'][0:np.int(dict_data['XpT'].shape[0]/2)],
              'XfT': dict_data['XfT'][0:np.int(dict_data['XfT'].shape[0]/2)],
              'YpT': dict_data['YpT'][0:np.int(dict_data['YpT'].shape[0]/2)],
              'YfT': dict_data['YfT'][0:np.int(dict_data['YfT'].shape[0]/2)]}
dict_valid = {'XpT': dict_data['XpT'][np.int(dict_data['XpT'].shape[0]/2):],
              'XfT': dict_data['XfT'][np.int(dict_data['XfT'].shape[0]/2):],
              'YpT': dict_data['YpT'][np.int(dict_data['YpT'].shape[0]/2):],
              'YfT': dict_data['YfT'][np.int(dict_data['YfT'].shape[0]/2):]}
# dict_train = dict_data['train']
# dict_valid = dict_data['valid']
# EMBEDDING_NO = dict_data['embedding']
EMBEDDING_NO = 1
try:
    if dict_train['YpT'].shape[0] == 0:
        with_output = False
    else:
        with_output = True
except:
    with_output = False


num_bas_obs = dict_train['XpT'].shape[1]
num_train_samples = dict_train['XpT'].shape[0]
num_valid_samples = dict_valid['XpT'].shape[0]
if with_output:
    num_outputs = dict_train['YpT'].shape[1]

# Hidden layer list creation for state dynamics
x_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
x_hidden_vars_list[-1] = x_deep_dict_size

# Display info
print("[INFO] Number of total samples: " + repr(num_train_samples + num_valid_samples))
print("[INFO] Nonlinear system state dimension: " + repr(num_bas_obs))
if with_output:
    print("[INFO] Yf.shape (E-DMD): " + repr(num_outputs))
print("Number of training snapshots: " + repr(num_train_samples))
print("Number of validation snapshots: " + repr(num_valid_samples))
print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))

##
# ============================
# LEARNING THE STATE DYNAMICS
# ============================
with tf.device(DEVICE_NAME):
    dict_feed = {}
    dict_psi = {}
    dict_K ={}
    # Initialize the K and Wh matrices
    # Kx definition w/ bias
    KxT= weight_variable([x_deep_dict_size + num_bas_obs + 1, x_deep_dict_size + num_bas_obs])
    A_hat_opt = get_best_K_DMD(dict_train['XpT'], dict_train['XfT'], dict_valid['XpT'], dict_valid['XfT'])
    sess.run(tf.global_variables_initializer())
    KxT = tf.Variable(sess.run(KxT[0:num_bas_obs, 0:num_bas_obs].assign(A_hat_opt)))
    last_col = tf.constant(np.zeros(shape=(x_deep_dict_size + num_bas_obs, 1)), dtype=tf.dtypes.float32)
    last_col = tf.concat([last_col, [[1.]]], axis=0)
    KxT = tf.concat([KxT, last_col], axis=1)
    # sess.run(tf.global_variables_initializer())
    # Initialize the hidden layers
    Wx_list, bx_list = initialize_Wblist(num_bas_obs, x_hidden_vars_list)
    x_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x_hidden_vars_list, 'x_observables':x_deep_dict_size, 'W_list': Wx_list, 'b_list': bx_list,
                     'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
    psixpz_list, psixpT, xpT_feed = initialize_tensorflow_graph(x_params_list)
    psixfz_list, psixfT, xfT_feed = initialize_tensorflow_graph(x_params_list)
    dict_feed ['xpT'] = xpT_feed
    dict_feed ['xfT'] = xfT_feed
    if with_output:
        # Wh definition_
        ypT_feed = tf.placeholder(tf.float32, shape=[None, num_outputs])
        yfT_feed = tf.placeholder(tf.float32, shape=[None, num_outputs])
        WhT = weight_variable([x_deep_dict_size + num_bas_obs + 1, num_outputs])
        dict_feed['ypT'] = ypT_feed
        dict_feed['yfT'] = yfT_feed
        dict_K['WhT'] = WhT
    dict_psi ['xpT'] = psixpT
    dict_psi['xfT'] = psixfT
    dict_K['KxT'] = KxT
    dict_feed['step_size'] = tf.placeholder(tf.float32, shape=[])
    sess.run(tf.global_variables_initializer())
    dict_model_metrics = objective_func(dict_feed, dict_psi, dict_K)

    print('Training begins now!')
    all_histories, dict_run_info = static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics)
    print('---   TRAINING COMPLETE   ---')
estimate_K_stability(KxT)
if with_output:
    feed_dict_train = {xpT_feed: dict_train['XpT'], xfT_feed: dict_train['XfT'], ypT_feed:dict_train['YpT'], yfT_feed:dict_train['YfT'] }
    feed_dict_valid = {xpT_feed: dict_valid['XpT'], xfT_feed: dict_valid['XfT'], ypT_feed:dict_valid['YpT'], yfT_feed:dict_valid['YfT']}
else:
    feed_dict_train = {xpT_feed: dict_train['XpT'], xfT_feed: dict_train['XfT']}
    feed_dict_valid = {xpT_feed: dict_valid['XpT'], xfT_feed: dict_valid['XfT']}
train_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
valid_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)



## Saving the results of the run

# Find the runs in the folder
storage_folder = data_directory + '/MyMac'
if not os.path.exists(storage_folder):
    os.mkdir(storage_folder)
    if not STATE_RUN_NUMBER_PROCURED:
        RUN_NUMBER = 1
else:
    if not STATE_RUN_NUMBER_PROCURED:
    # Find the last run number and add 1 to it
        ls_all_run_files = os.listdir(storage_folder)
        ls_run_numbers = [np.int(i[4:]) for i in ls_all_run_files if 'RUN_' in i]
        # RUN_NUMBER = np.min(list(set(list(range(1,np.max(ls_run_numbers) + 2))) - set(ls_run_numbers)))
        RUN_NUMBER = np.int(np.max(ls_run_numbers)) + 1
FOLDER_NAME = storage_folder + '/RUN_' + str(RUN_NUMBER)
if os.path.exists(FOLDER_NAME):
    shutil.rmtree(FOLDER_NAME)
os.mkdir(FOLDER_NAME)

dict_dump = {}
dict_dump['Wx_list_num'] = [sess.run(W_temp) for W_temp in Wx_list]
dict_dump['bx_list_num'] =[sess.run(b_temp) for b_temp in bx_list]
dict_dump['KxT_num'] = sess.run(dict_K['KxT'])
if with_output:
    dict_dump['WhT_num'] = sess.run(dict_K['WhT'])

with open(FOLDER_NAME + '/constrainedNN-Model.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_dump, file_obj_swing)
with open(FOLDER_NAME + '/run_info.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_run_info, file_obj_swing)
with open(FOLDER_NAME + '/all_histories.pickle', 'wb') as file_obj_swing:
    pickle.dump(all_histories, file_obj_swing)

saver = tf.compat.v1.train.Saver()

all_tf_var_names =[]
for items in dict_psi.keys():
    tf.compat.v1.add_to_collection('psi'+items, dict_psi[items])
    all_tf_var_names.append('psi'+items)
for items in dict_feed.keys():
    tf.compat.v1.add_to_collection(items+'_feed', dict_feed[items])
    all_tf_var_names.append(items+'_feed')
for items in dict_K.keys():
    tf.compat.v1.add_to_collection(items, dict_K[items])
    all_tf_var_names.append(items)
for items in list(dict_model_metrics.keys()):
    all_tf_var_names.append(items)
    tf.compat.v1.add_to_collection(items, dict_model_metrics[items])

saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)
print('------ ------ -----')
print('----- Run Info ----')
print('------ ------ -----')
print(pd.DataFrame(dict_run_info))
print('------ ------ -----')


# Saving the hyperparameters
dict_hp = {'x_obs': x_deep_dict_size, 'x_layers': n_x_nn_layers, 'x_nodes': n_x_nn_nodes, 'regularization factor': regularization_lambda}
if with_output:
    dict_hp['r2 train'] = np.array([dict_run_info[list(dict_run_info.keys())[-1]]['r^2 X train accuracy'],
                                    dict_run_info[list(dict_run_info.keys())[-1]]['r^2 Y train accuracy']])
    dict_hp['r2 valid'] = np.array([dict_run_info[list(dict_run_info.keys())[-1]]['r^2 X valid accuracy'],
                                    dict_run_info[list(dict_run_info.keys())[-1]]['r^2 Y valid accuracy']])
else:
    dict_hp['r2 train'] = np.array([dict_run_info[list(dict_run_info.keys())[-1]]['r^2 X train accuracy']])
    dict_hp['r2 valid'] = np.array([dict_run_info[list(dict_run_info.keys())[-1]]['r^2 X valid accuracy']])

with open(FOLDER_NAME + '/dict_hyperparameters.pickle','wb') as handle:
    pickle.dump(dict_hp,handle)

##

