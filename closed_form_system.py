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


a11 = 0.5
a21 = 0.6
a22 = 0.3
gamma = 1.2

def closed_form_system(x,t):
    xdot = np.zeros(len(x))
    xdot[0] = a11*x[0]
    xdot[1] = a21*x[1] + a22*x[1]
    return xdot



