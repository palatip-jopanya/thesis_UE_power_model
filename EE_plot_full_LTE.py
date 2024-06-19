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



def plotFull(fileid,endc,mode):
    common = MyCommonClass()
    filename = common.path2fileonly(fileid)
    filesave = common.removeExt(fileid)
    h = 360
    casetitle = filename
    dataCsv = pd.read_csv(fileid)

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
        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data2Name = "Lte_RSRP(dBm)"
        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data3Name = "Lte_RSRQ(dBm)"

    elif endc == 1 and mode == 'deg':
        data1 = dataCsv["H_degree"].to_numpy()
        data1Name = "H_degree"
        data2 = dataCsv["NR_RSRP(dBm)"].to_numpy()
        data2Name = "NR_RSRP(dBm)"
        data3 = dataCsv["DL_NrAck%"].to_numpy()
        data3Name = "DL_NrAck%"

    elif endc == 0 and mode == 'UL':
        data1 = dataCsv["LTEULpwr(dBm)"].to_numpy()
        data1Name = "LTE UL power(dBm)"

        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data2Name = "Lte_RSRP(dBm)"

        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data3Name = "Lte_RSRQ(dBm)"

        data4 = dataCsv["DL_LteAck%"].to_numpy()
        data4Name = "DL_LteAck%"
        data5 = dataCsv["UL_LteAck%"].to_numpy()
        data5Name = "UL_LteAck%"

        data6 = dataCsv["DL_LteNack%"].to_numpy()
        data6Name = "DL_LteNack%"
        data7 = dataCsv["DL_LteDtx%"].to_numpy()
        data7Name = "DL_LteDtx%"
        data8 = dataCsv["DL_LteBLER%"].to_numpy()
        data8Name = "DL_LteBLER%"

        data9 = dataCsv["UL_LteNack%"].to_numpy()
        data9Name = "UL_LteNack%"
        data10 = dataCsv["UL_LteDtx%"].to_numpy()
        data10Name = "UL_LteDtx%"
        data11 = dataCsv["UL_LteBLER%"].to_numpy()
        data11Name = "UL_LteBLER%"

    elif endc == 0 and mode == 'deg':
        data1 = dataCsv["H_degree"].to_numpy()
        data1Name = "Degree in horizon"

        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data2Name = "Lte_RSRP(dBm)"

        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data3Name = "Lte_RSRQ(dBm)"

        data4 = dataCsv["DL_LteAck%"].to_numpy()
        data4Name = "DL_LteAck%"
        data5 = dataCsv["UL_LteAck%"].to_numpy()
        data5Name = "UL_LteAck%"

        data6 = dataCsv["DL_LteNack%"].to_numpy()
        data6Name = "DL_LteNack%"
        data7 = dataCsv["DL_LteDtx%"].to_numpy()
        data7Name = "DL_LteDtx%"
        data8 = dataCsv["DL_LteBLER%"].to_numpy()
        data8Name = "DL_LteBLER%"

        data9 = dataCsv["UL_LteNack%"].to_numpy()
        data9Name = "UL_LteNack%"
        data10 = dataCsv["UL_LteDtx%"].to_numpy()
        data10Name = "UL_LteDtx%"
        data11 = dataCsv["UL_LteBLER%"].to_numpy()
        data11Name = "UL_LteBLER%"

    LTE_DLrate = dataCsv["DL_LteThp(bps)"].to_numpy()
    LTE_DLrateName = "LTE_DL_rate"
    LTE_ULrate = dataCsv["UL_LteTph(bps)"].to_numpy()
    LTE_ULrateName = "LTE_UL_rate"
    sumrate = LTE_DLrate + LTE_ULrate
    sumrateName = "Transfer data rate"
    data0 = dataCsv["UE_PowerMeas(mW)"].to_numpy()
    data0Name = "Joule per bit"
    DLULratio = LTE_DLrate/LTE_ULrate
    DLULratioName = "DL:UL ratio"
    EE = data0/sumrate*1e6
    EEName = "Energy efficiency (micro joule/bit)"

    #print(sumrate)
    #print(DLULratio)
    #print(EE)


    ############ Plot ###############
    if endc == 0 and mode == 'UL':
        marker = 6
        linewid = 2
        ssize = 30
        labelsize = 12
        titlesize = 18
        legendsize = 12
    elif endc == 0 and mode == 'deg':
        marker = 2 #line
        linewid = 1
        ssize = 10
        labelsize = 12
        titlesize = 18
        legendsize = 12
    ################################

    fig, axs = plt.subplots(3, sharex=True, figsize=(20, 10))
    fig.suptitle("Energy efficiency " + casetitle,fontsize=titlesize)
    axs[0].plot(data1, EE, color='olivedrab', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='darkgreen', markersize=marker, label=EEName) # UL sweep main feat
    axs[0].grid()
    axs[0].legend(fontsize=legendsize)
    axs[0].set_ylabel("micro Joule/bit")

    axs[1].plot(data1, LTE_DLrate, color='navy', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='navy', markersize=marker, label=LTE_DLrateName) # RSRP
    axs[1].plot(data1, LTE_ULrate, color='steelblue', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='steelblue', markersize=marker, label=LTE_ULrateName)  # RSRP
    axs[1].plot(data1, sumrate, color='orange', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='orange', markersize=marker, label=sumrateName)  # RSRP
    axs[1].grid()
    axs[1].legend(fontsize=legendsize)
    axs[1].set_ylabel("bit/s", fontsize=labelsize)

    axs[2].plot(data1, DLULratio, color='gold', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='gold', markersize=marker, label=DLULratioName) # RSRQ
    axs[2].grid()
    axs[2].legend(fontsize=legendsize)
    axs[2].set_ylabel("ratio", fontsize=labelsize)


    axs[2].set_xlabel(data1Name, fontsize=labelsize)


    #plt.show()
    plt.savefig(filesave + '_plt_full_LTE', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    endc = 0 # o-lte 1-endc
    mode = 'deg' # UL, DL, deg
    plotFull(file_path,endc,mode)
