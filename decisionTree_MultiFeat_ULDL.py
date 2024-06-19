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

plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(60, 45))
from sklearn.tree import export_graphviz

def decisionTree(fileid,endc,mode):
    common = MyCommonClass()
    filename = common.path2fileonly(fileid)
    filesave = common.removeExt(fileid)
    h = 360
    casetitle = filename
    dataCsv = pd.read_csv(fileid)
    data0 = dataCsv["UE_PowerMeas(mW)"].to_numpy()
    data0Name  = "UE_PowerMeas(mW)"
    #data0 = data0.reshape(-1,1)

    if endc == 1 and mode == 'UL':
        data1 = dataCsv["NRULpwr(dBm)"].to_numpy()
        data1Name = "NRULpwr(dBm)"
        data2 = dataCsv["UL_NrThp%"].to_numpy()
        data2Name = "UL_NrThp%"
        data3 = dataCsv["UL_NrBLER%"].to_numpy()
        data3Name = "UL_NrBLER%"

    elif endc == 1 and mode == 'DL':
        data1 = dataCsv["NRDLpwr(dBm)"].to_numpy()
        data1Name = "NRDLpwr(dBm)"
        data2 = dataCsv["NR_RSRQ(dBm)"].to_numpy()
        data2Name = "NR_RSRQ(dBm)"
        data3 = dataCsv["NR_SINR(dBm)"].to_numpy()
        data3Name = "NR_SINR(dB)"

    elif endc == 1 and mode == 'deg':
        data1 = dataCsv["H_degree"].to_numpy()
        data1Name = "H_degree"
        #data1 = dataCsv["NR_RSRP(dBm)"].to_numpy() # 147
        #data1Name = "NR_RSRP(dBm)"
        #data1 = dataCsv["UL_NrDtx%"].to_numpy() # 146
        #data1Name = "UL_NrDtx%" # 146
        data2 = dataCsv["DL_NrAck%"].to_numpy()
        data2Name = "DL_NrAck%"
        data3 = dataCsv["UL_NrAck%"].to_numpy()
        data3Name = "UL_NrAck%"

    elif endc == 0 and mode == 'UL':
        data1 = dataCsv["LTEULpwr(dBm)"].to_numpy()
        data1Name = "LTEULpwr(dBm)"
        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data2Name = "Lte_RSRP(dBm)"
        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data3Name = "Lte_RSRQ(dBm)"

    elif endc == 0 and mode == 'deg':
        data1 = dataCsv["H_degree"].to_numpy()
        data1Name = "H_degree"
        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data2Name = "Lte_RSRP(dBm)"
        #data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        #data3Name = "Lte_RSRQ(dBm)"
        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data3Name = "Lte_RSRQ(dBm)"


    dataX = np.stack((data1, data2, data3), axis=-1)
    print(dataX)
    #dataX = dataX.reshape(-1, 1)  # reshape it as a ndarray

    ############ Plot ###############
    depth1 = 2
    depth2 = 7
    x_train, x_test, y_train, y_test = train_test_split(dataX, data0, test_size=0.2, random_state=2)  # (X,Y)
    regr_1 = DecisionTreeRegressor(max_depth=depth1)
    regr_2 = DecisionTreeRegressor(max_depth=depth2)
    regr_1.fit(x_train, y_train)
    regr_2.fit(x_train, y_train)

    # Predict x line
    if endc == 1 and mode == 'UL':
        x_plot = np.arange(-39, 23, 0.1)[:, np.newaxis]
    elif endc == 1 and mode == 'DL':
        x_plot = np.arange(-45, -18, 0.1)[:, np.newaxis] # NR DL
        #x_plot = np.arange(11, 24, 0.1)[:, np.newaxis] # SINR
    elif endc == 1 and mode == 'deg':
        x_plot = np.arange(0, 180, 0.1)[:, np.newaxis] # NR DL
    elif endc == 0 and mode == 'deg':
        x_plot = np.arange(0, h , 0.1)[:, np.newaxis]
    #y_1 = regr_1.predict(x_plot)
    #y_2 = regr_2.predict(x_plot)
    y_pred2 = regr_2.predict(x_test)
    y_pred1 = regr_1.predict(x_test)

    index = x_test[:, 0].argsort()

    # r2 score
    print(x_test.shape)
    print(y_test.shape)
    r2score_1 = regr_1.score(x_test, y_test)
    r2score_2 = regr_2.score(x_test, y_test)
    print('r2 score reg1 ' + str(r2score_1))
    print('r2 score reg2 ' + str(r2score_2))
    print(x_test[:,0].shape)
    print(y_pred1.shape)
    # Plot the results
    plt.figure()
    plt.scatter(x_train[:,0], y_train, s=10, edgecolor="black", c="darkorange", label='Train ' + data1Name)
    plt.scatter(x_test[:,0], y_test, s=10, edgecolor="red", c="purple", label='Test ' + data1Name)
    plt.plot(x_test[index,0], y_pred1[index], color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(x_test[index,0], y_pred2[index], color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel(data1Name, fontsize=10)
    plt.ylabel(data0Name, fontsize=10)
    treefea = data1Name + ', ' + data2Name + ', ' + data3Name
    plt.title("3 Features of Decision Tree Regression \n " + treefea + " \n" + filename + '\n r2_score depth=(' +str(depth1)+ ','+str(depth2)+')=(' + str(round(r2score_1,2)) + ',' + str(round(r2score_2,2)) +')')
    plt.legend(prop = { "size": 8 })
    plt.grid()
    plt.savefig(filesave + '_decisionTree_3Feat', dpi=300, bbox_inches='tight')

    print("loss of len1:" + str(sum(abs(x_test[index, 0] - y_pred1[index]))))
    print("loss of len2:" + str(sum(abs(x_test[index, 0] - y_pred2[index]))))

if __name__ == "__main__":
    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    endc = 0 # o-lte 1-endc
    mode = 'deg' # UL, DL, deg
    decisionTree(file_path,endc,mode)