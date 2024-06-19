import matplotlib.pyplot as plt
import numpy as np
from preprocessing import log_to_csv
from pathlib import Path
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
from commonClass import MyCommonClass
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn import tree

plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(20, 10))

# Create a random dataset
def decisionTree(fileid,endc,mode):
    common = MyCommonClass()
    filename = common.path2fileonly(fileid)
    filesave = common.removeExt(fileid)
    depth1 = 2
    depth2 = 7

    dataCsv = pd.read_csv(fileid)
    #print(dataCsv[["Lte_RSRQ(dBm)","PowerMeas(mW)"]])
    #selectData = dataCsv[["Lte_RSRQ(dBm)", "PowerMeas(mW)"]]
    if endc == 1 and mode == 'UL':
        X = dataCsv["NRULpwr(dBm)"].to_numpy()
        data1name = "NR_ULpwr(dBm)"
    elif endc == 1 and mode == 'DL':
        X = dataCsv["NRDLpwr(dBm)"].to_numpy()
        data1name = "NR_DLpwr(dBm)"
        #print(X)
        #X = dataCsv["NR_SINR(dBm)"].to_numpy()
        #data1name = "NR_SINR(dBm)"
    elif endc == 1 and mode == 'deg':
        X = dataCsv["H_degree"].to_numpy()
        data1name = "H_degree"
    elif endc == 0 and mode == 'UL':
        X = dataCsv["LTEULpwr(dBm)"].to_numpy()
        data1name = "LTE_ULpwr(dBm)"
    elif endc == 0 and mode == 'DL':
        X = dataCsv["LTEDLpwr(dBm)"].to_numpy()
        data1name = "LTE_DLpwr(dBm)"
    elif endc == 0 and mode == 'deg':
        X = dataCsv["H_degree"].to_numpy()
        data1name = "H_degree"

    X = X.reshape(-1, 1)
    #print(X)
    Y = dataCsv["UE_PowerMeas(mW)"].to_numpy()
    data0name = "UE_PowerMeas(mW)"

    Y = Y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)  # (X,Y)
    regr_1 = DecisionTreeRegressor(max_depth=depth1)
    regr_2 = DecisionTreeRegressor(max_depth=depth2)
    regr_1.fit(x_train, y_train)
    regr_2.fit(x_train, y_train)

    # Predict
    if endc == 1 and mode == 'UL':
        x_plot = np.arange(-39, 23, 0.1)[:, np.newaxis]
    elif endc == 1 and mode == 'DL':
        x_plot = np.arange(-45, -18, 0.1)[:, np.newaxis] # NR DL
        #x_plot = np.arange(11, 24, 0.1)[:, np.newaxis] # SINR
    elif endc == 1 and mode == 'deg':
        x_plot = np.arange(0, 180, 0.1)[:, np.newaxis] # NR DL
        #x_plot = np.arange(0, 180, 0.1)[:, np.newaxis]  # NR DL
    elif endc == 0 and mode == 'UL':
        x_plot = np.arange(-9, 23, 0.1)[:, np.newaxis]
    elif endc == 0 and mode == 'DL':
        x_plot = np.arange(-45, -18, 0.1)[:, np.newaxis]
    elif endc == 0 and mode == 'deg': # degree
        x_plot = np.arange(-180, 180, 0.1)[:, np.newaxis] # NR DL

    y_1 = regr_1.predict(x_plot)
    y_2 = regr_2.predict(x_plot)
    y_pred = regr_2.predict(x_test)

    # r2 score
    print(x_test.shape)
    print(y_test.shape)
    r2score_1 = regr_1.score(x_test, y_test)
    r2score_2 = regr_2.score(x_test, y_test)

    # Plot the results
    plt.figure()
    plt.scatter(x_train, y_train, s=10, edgecolor="black", c="darkorange", label='Train ')
    #plt.plot(X, Y, color="blue", label='Train ',linewidth=2)

    plt.scatter(x_test, y_pred, s=10, edgecolor="red", c="purple", label='Test ')
    #plt.plot(x_plot, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(x_plot, y_2, color="yellowgreen", label="Test max_depth=" +str(depth2) , linewidth=2)
    plt.xlabel(data1name, fontsize=10)
    plt.ylabel(data0name, fontsize=10)
    plt.title("Single feature of Decision Tree Regression \n" + filename + ' r2_score depth=(' +str(depth1)+ ','+str(depth2)+')=(' + str(round(r2score_1,2)) + ',' + str(round(r2score_2,2)) +')')
    plt.legend(prop = {"size": 8})
    plt.grid()
    plt.savefig(filesave + '_decisionTree_singleFea', dpi=300, bbox_inches='tight')

    export_graphviz(regr_2, out_file=filesave+'_dot.dot',
                    feature_names=[data1name])



if __name__ == "__main__":
    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    endc = 1 # o-lte 1-endc
    mode = 'deg' # UL, DL, deg
    decisionTree(file_path,endc,mode)