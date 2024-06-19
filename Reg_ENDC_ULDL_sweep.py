import matplotlib.pyplot as plt
import numpy as np
from preprocessing import log_to_csv
from pathlib import Path
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from plotnmodel import plotclass
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pandas as pd
from commonClass import MyCommonClass

plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(60, 45))



def plot(mode):
    common = MyCommonClass()
    #define including variable
    #listSelection = ["Power Meas ","Lte_RSRP(dBm)","NR_RSRP(dBm)"]


    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(file_path)


    filename = os.path.splitext(os.path.basename(file_path))[0]


    figname = common.removeExt(file_path)

    #casetitle = "UL power sweeps in LTE-only in a good channel condition (at 0 degree)"
    #casetitle = "UL power sweeps in LTE-only in a bad channel condition (at 60 degree)"
    casetitle = "UL power sweeps in ENDC at H112 degree \n"
    data = pd.read_csv(file_path)

    ################## selecting data #############

    data0 = data["UE_PowerMeas(mW)"].to_numpy()
    length = len(data0)

    data0Name = "UE_PowerMeas(mW)"
    if mode == 'UL':
        data1 = data["NRULpwr(dBm)"].to_numpy()
        data1Name = "NRULpwr(dBm)"
        data2 = data["NR_RSRQ(dBm)"].to_numpy()
        data2Name = "NR_RSRQ(dBm)"
        data3 = data["UL_NrThp%"].to_numpy()
        data3Name = "UL_NrThp%"
    elif mode == 'DL':
        data1 = data["NRDLpwr(dBm)"].to_numpy()
        data1Name = "NRDLpwr(dBm)"
        data2 = data["NR_RSRQ(dBm)"].to_numpy()
        data2Name = "NR_RSRQ(dBm)"
        data3 = data["NR_SINR(dBm)"].to_numpy()
        data3Name = "NR_SINR(dBm)"
        data4 = data["DL_NrBLER%"].to_numpy()
        data4Name = "DL_NrBLER%"


    ############ Plot ###############
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('NR RF Downlink power (dBm)', fontsize=8)
    ax1.set_ylabel('RF power (dBm) i.e. RSRP, RSRQ, UL, DL power', color=color, fontsize=8)
    #ax1.plot(data1, data1, 'r.')
    ax1.plot(data1, data2, 'r-',markersize=2)
    ax1.plot(data1, data3, 'ro',markersize=2)
    ax1.plot(data1, data4, 'm-',markersize=2)

    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('UE Power (mW)', color=color, fontsize=8)  # we already handled the x-label with ax1
    ax2.plot(data1, data0, 'bo',markersize=2)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend([data2Name,data3Name,data4Name], prop = { "size": 8 }, loc='upper left')
    ax2.legend(["UE Power (mW)"], prop = { "size": 8 },loc='upper right')
    plt.title(casetitle)
    plt.grid()
    plt.savefig(figname + '_singleplot', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


    ###################### polynomial 3 features
    print(type(data1))
    dataX = data1 #np.stack((data1,data2,data3),axis=-1)
    #print(dataX)
    dataX = data1.reshape(-1,1) # reshape it as a ndarray
    #print(data1)
    x_train, x_test, y_train, y_test = train_test_split(dataX, data0, test_size=0.2, random_state=2) #(X,Y)
    print(len(y_test))
    r2_score_orders =[]
    arr_mse = []
    for i in range(10): # for benchmark nth order
        # applying polynomial regression degree 2
        poly = PolynomialFeatures(degree=(i+1), include_bias=True)
        x_train_trans = poly.fit_transform(x_train)
        x_test_trans = poly.transform(x_test)
        # include bias parameter
        lr = LinearRegression()
        lr.fit(x_train_trans, y_train)
        y_pred = lr.predict(x_test_trans)
        mse = sum(np.square(abs(y_pred-y_test)))/length
        print(mse)
        arr_mse.append(mse)
        r2_score_orders.append(r2_score(y_test, y_pred,multioutput='variance_weighted')) # regression scpre multioutput : string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’]
    print(r2_score_orders)
    arr_mse = arr_mse/max(arr_mse)
    print(arr_mse)

    degree1 = 30
    # just one order
    poly = PolynomialFeatures(degree=30, include_bias=True)
    x_train_trans = poly.fit_transform(x_train)
    x_test_trans = poly.transform(x_test)
    # include bias parameter
    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)
    y_pred = lr.predict(x_test_trans)
    r2Score = r2_score(y_test, y_pred, multioutput='variance_weighted')  # regression scpre multioutput : string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’]

    print('coeff',lr.coef_)
    print('intercept',lr.intercept_)


    # visualize the fitting line and actual training, testing point
    if mode == 'UL':
        X_new1 = np.linspace(-39, 23, 200).reshape(200, 1)
    elif mode == 'DL':
        X_new1 = np.linspace(-45, -18, 200).reshape(200, 1)
    X_new_poly = poly.transform(X_new1)
    y_new = lr.predict(X_new_poly)

    fig, objplot = plt.subplots(2, layout="constrained")
    #fig.suptitle(casetitle)
    objplot[0].plot(X_new1, y_new, "r-", linewidth=2, label="Predictions")
    objplot[0].plot(x_train, y_train, "b.", label='Training points')
    objplot[0].plot(x_test, y_test, "g.", label='Testing points')
    objplot[0].legend(["Predictions","Training points","Testing points"], prop = { "size": 8 })
    objplot[0].set_xlabel('RF Uplink power (dBm)', fontsize=8)
    objplot[0].set_ylabel('UE power (mW)', fontsize=8)
    objplot[0].set_title(casetitle + ', single feature, '+str(degree1)+'th order, r2_score=' + str(round(r2Score,4)))
    objplot[0].grid()
    # plot
    x = list(i+1 for i in range(10))
    objplot[1].plot(x, r2_score_orders)
    objplot[1].plot(x, arr_mse)
    objplot[1].set_xlabel("Polynomial order nth", fontsize=8)
    objplot[1].set_ylabel("MSE, r2 scores", fontsize=8)
    objplot[1].legend(["r2 score", "normalized MSE"], prop = { "size": 8 })
    objplot[1].set_title("MSE and R2 score of nth order of polynomial regression ")
    objplot[1].grid()
    #plt.figure(figsize=(25, 14))
    plt.savefig(figname + 'singlefea_reg_plot', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


if __name__ == "__main__":
    print('..in main..')
    mode = 'DL' # UL Deg
    start = plot(mode) # 1 select manual 0-auto read file name

# LTEonly UL sweep py case 142,143,144


