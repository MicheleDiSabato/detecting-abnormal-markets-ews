import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import kstest, ks_2samp, pearsonr
import seaborn as sns
import warnings
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import random

parent_dir = os.getcwd()
model_folder_path = os.path.join(parent_dir, "plots") 
if os.path.isdir(model_folder_path) == False:
    os.makedirs(model_folder_path) 

def plot_feature_stationary(feature_name, lw=3, figsize = (15, 10)):
    plt.figure(figsize = figsize)
    plt.plot(d_stationary[feature_name], linewidth = lw, color = "darkorange")
    anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
    lb = np.min(d_stationary[feature_name])-0.1
    ub = np.abs(lb) + np.max(d_stationary[feature_name])+0.1
    plt.bar(x=anomaly_indices, height=ub, bottom = lb)
    plt.show()

def plot_feature(feature_name, lw=3, figsize = (15, 10)):
    plt.figure(figsize = figsize)
    plt.plot(d[feature_name], linewidth = lw, color = "darkorange")
    anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
    lb = np.min(d[feature_name])-0.1
    ub = np.abs(lb) + np.max(d[feature_name])+0.1
    plt.bar(x=anomaly_indices, height=ub, bottom = lb)
    plt.show()


mat = scipy.io.loadmat('EWS.mat')
names = list(mat.keys())[3:-2]
y = mat["Y"]
y_cat = np.zeros_like(y, dtype = int)
y_cat[y[:,0]==1.0,0] = 1
col = []
times = np.arange(y.shape[0])
for ipsilon in y:
    if ipsilon == 1.0:
        col.append("red")
    else:
        col.append("blue")
d = {}
for i, name in enumerate(names):
    if name != "None":
        d[name] = mat[names[i]].reshape(-1,)


Indices_Currencies = ["XAUBGNL", "BDIY", "CRY", "Cl1", "DXY", "EMUSTRUU", "GBP", "JPY", "LF94TRUU", "LF98TRUU", "LG30TRUU", "LMBITR", "LP01TREU", "LUACTRUU", "LUMSTRUU", "MXBR", "MXCN", "MXEU", "MXIN", "MXJP", "MXRU", "MXUS", "VIX"]

InterestRates = ["EONIA", "GTDEM10Y", "GTDEM2Y", "GTDEM30Y", "GTGBP20Y", "GTGBP2Y", "GTGBP30Y", "GTITL10YR", "GTITL2YR", "GTITL30YR", "GTJPY10YR", "GTJPY2YR", "GTJPY30YR", "US0001M", "USGG3M", "USGG2YR", "GT10", "USGG30YR"]

response_cat = y_cat[1:, 0]
response = y[1:, 0]

d_stationary = {}
for feature_name in list(d.keys()):
    if feature_name in Indices_Currencies:
        temp = np.log(d[feature_name])
        d_stationary[feature_name] = temp[1:] - temp[:-1]
        d_stationary[feature_name] = d_stationary[feature_name].reshape(-1,)
        # d_stationary[feature_name] = np.log(d[feature_name][1:] / d[feature_name][1:])
    elif feature_name == "ECSURPUS":
        d_stationary[feature_name] = d[feature_name][1:]
        d_stationary[feature_name] = d_stationary[feature_name].reshape(-1,)
    elif feature_name in InterestRates:
        d_stationary[feature_name] = d[feature_name][1:] - d[feature_name][:-1]
        d_stationary[feature_name] = d_stationary[feature_name].reshape(-1,)

plot_names=["VIX", "XAUBGNL", "EONIA", "USGG2YR"]
df = pd.DataFrame(d)
fig, axs = plt.subplots(2, 2, figsize = (20,7))
count = 0
for i in range(2):
    for j in range(2):
        feature_name_2 = plot_names[count]
        axs[i,j].plot(d[feature_name_2], linewidth = 2, color = "darkorange")
        anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
        lb = np.min(d[feature_name_2])-0.1
        ub = np.abs(lb) + np.max(d[feature_name_2])+0.1
        axs[i,j].bar(x=anomaly_indices, height=ub, bottom = lb)
        axs[i,j].set_title(feature_name_2, fontsize=15)
        count += 1
# plt.show()
fig.savefig('plots' + os.sep + 'VIX_XAUBGNL_EONIA_USGG2YR', bbox_inches='tight')

df_stat = pd.DataFrame(d_stationary)
fig, axs = plt.subplots(2, 2, figsize = (20,7))
count = 0
for i in range(2):
    for j in range(2):
        feature_name_2 = plot_names[count]
        axs[i,j].plot(df_stat[feature_name_2], linewidth = 2, color = "darkorange")
        anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
        lb = np.min(df_stat[feature_name_2])-0.1
        ub = np.abs(lb) + np.max(df_stat[feature_name_2])+0.1
        axs[i,j].bar(x=anomaly_indices, height=ub, bottom = lb)
        axs[i,j].set_title(feature_name_2, fontsize=15)
        count += 1
# plt.show()
fig.savefig('plots' + os.sep + 'VIX_XAUBGNL_EONIA_USGG2YR_stationary', bbox_inches='tight')

plot_names=['BDIY', 'CRY', 'EONIA', 'GTDEM2Y', 'GTGBP2Y', 'MXBR', 'MXEU', 'MXUS', 'USGG2YR', 'VIX']

fig, axs = plt.subplots(2, 5, figsize = (20,7))
count = 0
for i in range(2):
    for j in range(5):
        feature_name_2 = plot_names[count]
        axs[i,j].plot(d[feature_name_2], linewidth = 2, color = "darkorange")
        anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
        lb = np.min(d[feature_name_2])-0.1
        ub = np.abs(lb) + np.max(d[feature_name_2])+0.1
        axs[i,j].bar(x=anomaly_indices, height=ub, bottom = lb)
        axs[i,j].set_title(feature_name_2, fontsize=15)
        count += 1
plt.show()
fig.savefig('plots' + os.sep + 'final_features', bbox_inches='tight')

fig, axs = plt.subplots(2, 5, figsize = (20,7))
count = 0
for i in range(2):
    for j in range(5):
        feature_name_2 = plot_names[count]
        axs[i,j].plot(df_stat[feature_name_2], linewidth = 1, color = "darkorange")
        anomaly_indices=np.argwhere(y_cat[:,0] == 1)[:,0]
        lb = np.min(df_stat[feature_name_2])-0.1
        ub = np.abs(lb) + np.max(df_stat[feature_name_2])+0.1
        axs[i,j].bar(x=anomaly_indices, height=ub, bottom = lb)
        axs[i,j].set_title(feature_name_2, fontsize=15)
        count += 1
plt.show()
fig.savefig('plots' + os.sep + 'final_features_stationary', bbox_inches='tight')





























































