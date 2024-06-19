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

plt.rcParams.update({'font.size': 12})


def get_var_name(var):
    for name, value in locals().items():
        if value is var:
            return name
def readcsv(fileid):
    with open(fileid, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data_firstrow = np.array(data)[0]
        data = np.delete(data,0,0)
        data_array = np.array(data, dtype=str)
    return data_array, data_firstrow



def plot(type,endc):
    common = MyCommonClass()
    #define including variable

    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    print(file_path)

    data = pd.read_csv(file_path)

    data0 = data["UE_PowerMeas(mW)"].to_numpy()
    data0Name = "UE_PowerMeas(mW)"

    if type == "DL" and endc == 1:
        data1 = data["NRDLpwr(dBm)"].to_numpy()
        data1Name = "NRDLpwr(dBm)"
        data2 = data["NR_RSRQ(dBm)"].to_numpy()
        data2Name = "NR_RSRQ(dBm)"
        data3 = data["NR_SINR(dBm)"].to_numpy()
        data3Name = "NR_SINR(dB)"
    else:
        data1 = data["NRDLpwr(dBm)"].to_numpy()
        data1Name = "NRDLpwr(dBm)"
        data2 = data["NR_RSRQ(dBm)"].to_numpy()
        data2Name = "NR_RSRQ(dBm)"
        data3 = data["NR_SINR(dB)"].to_numpy()
        data3Name = "NR_SINR(dB)"



    nameonly = common.path2fileonly(file_path)
    figname = common.removeExt(file_path)


    ###################### polynomial 3 features

    dataX = np.stack((data1,data2,data3),axis=-1)
    #print(dataX.shape)
    #dataX = data1.reshape(-1,1) # reshape it as a ndarray
    #print(dataX)
    x_train, x_test, y_train, y_test = train_test_split(dataX, data0, test_size=0.2, random_state=2) #(X,Y)
    print(x_train.shape)
    print(x_train)
    r2_score_orders =[]
    arr_mse = []
    length = len(y_test)
    for i in range(4): # for benchmark nth order
        # applying polynomial regression degree 2
        poly = PolynomialFeatures(degree=(i+1), include_bias=True)
        x_train_trans = poly.fit_transform(x_train)
        x_test_trans = poly.transform(x_test)
        # include bias parameter
        lr = LinearRegression()
        lr.fit(x_train_trans, y_train)
        y_pred = lr.predict(x_test_trans)
        mse = sum(np.square(abs(y_pred-y_test)))/length

        arr_mse.append(mse)
        r2_score_orders.append(r2_score(y_test, y_pred,multioutput='variance_weighted')) # regression scpre multioutput : string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’]
    print(r2_score_orders)
    arr_mse = arr_mse/max(arr_mse)
    print(arr_mse)

    # just one order
    poly = PolynomialFeatures(degree=4, include_bias=True)
    x_train_trans = poly.fit_transform(x_train)
    x_test_trans = poly.transform(x_test)
    print(x_train_trans.shape)
    print(x_train_trans)
    print(x_test_trans.shape)
    print(x_test_trans)
    # include bias parameter
    lr = LinearRegression()
    lr.fit(x_train_trans, y_train)
    y_pred = lr.predict(x_test_trans)
    r2Score = r2_score(y_test, y_pred, multioutput='variance_weighted')  # regression scpre multioutput : string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’]

    #print('coeff',lr.coef_)
    #print('intercept',lr.intercept_)
    #print(x_train[:,0])
    #print(x_test[:,0])

    # visualize the fitting line and actual training, testing point
    X_new1 = np.linspace(-45, -18, 200).reshape(200, 1)

    #X_new_poly = poly.transform(X_new1)
    #y_new = lr.predict(X_new_poly)

    fig, objplot = plt.subplots(2, layout="constrained")
    #fig.suptitle(casetitle)
    #print(x_test.shape)
    #print(y_pred.shape)
    index = x_test[:, 0].argsort()
    #testPred = np.append(x_test, y_pred, axis=1)
    #testPred[testPred[:, 0].argsort()]
    objplot[0].plot(x_test[index,0], y_pred[index], "r-", linewidth=2, label="Predictions")
    objplot[0].plot(x_train[:,0], y_train, "b.", label='Training points')
    objplot[0].plot(x_test[:,0], y_test, "g.", label='Testing points')
    objplot[0].legend()
    objplot[0].set_xlabel('Downlink power (dBm)', fontsize=8)
    objplot[0].set_ylabel('UE power (mW)', fontsize=8)
    objplot[0].set_title(nameonly + '\n 3 features (NR_DL_pwr(dBm), NR_RSRQ(dBm), NR_SINR(dB)),\n 6th order, r2_score=' + str(round(r2Score,4)))
    objplot[0].grid()
    # plot
    x = list(i+1 for i in range(4))
    objplot[1].plot(x, r2_score_orders)
    objplot[1].plot(x, arr_mse)
    objplot[1].set_xlabel("Polynomial order nth", fontsize=8)
    objplot[1].set_ylabel("MSE, r2 scores", fontsize=8)
    objplot[1].legend(["r2 score", "normalized MSE"], prop = { "size": 8 })
    objplot[1].set_title("MSE and R2 score of nth order of polynomial regression ")
    objplot[1].grid()
    #plt.figure(figsize=(25, 14))
    plt.savefig(figname + 'multifea_reg_plot', dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close()


if __name__ == "__main__":
    print('..in main..')
    type = "DL"
    endc = 1
    start = plot(type,endc) # 1 select manual 0-auto read file name

# LTEonly UL sweep py case 142,143,144


