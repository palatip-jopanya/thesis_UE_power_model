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



def plotFull():
    fileid1 = "C:/Users/ejoppal/OneDrive - Ericsson/Documents/UEpowerModel_pythonProject/measurementsLogs/141_Mk4_Lteonly_ULpwrSweep_64QAM_H0_processed.csv"
    fileid2 = "C:/Users/ejoppal/OneDrive - Ericsson/Documents/UEpowerModel_pythonProject/measurementsLogs/142_Mk4_Lteonly_ULpwrSweep_64QAM_H90_processed.csv"
    fileid3 = "C:/Users/ejoppal/OneDrive - Ericsson/Documents/UEpowerModel_pythonProject/measurementsLogs/143_Mk4_Lteonly_ULpwrSweep_64QAM_H60_processed.csv"

    common = MyCommonClass()
    filename = common.path2fileonly(fileid1)
    filesave = common.removeExt(fileid1)

    casetitle = "Mk4 LTEonly UL power sweep at 3 different orientations"

    dataCsv1 = pd.read_csv(fileid1)
    data10 = dataCsv1["UE_PowerMeas(mW)"].to_numpy()
    data10Name  = "H_degree = 0"
    data11 = dataCsv1["LTEULpwr(dBm)"].to_numpy()
    data11Name = "LTEULpwr(dBm)"

    dataCsv2 = pd.read_csv(fileid2)
    data20 = dataCsv2["UE_PowerMeas(mW)"].to_numpy()
    data20Name = "H_degree = 90"
    data21 = dataCsv2["LTEULpwr(dBm)"].to_numpy()
    data21Name = "LTEULpwr(dBm)"

    dataCsv3 = pd.read_csv(fileid3)
    data30 = dataCsv3["UE_PowerMeas(mW)"].to_numpy()
    data30Name = "H_degree = 60"
    data31 = dataCsv3["LTEULpwr(dBm)"].to_numpy()
    data31Name = "LTEULpwr(dBm)"




    ############ Plot ###############

    marker = 6
    linewid = 2
    ssize = 30
    labelsize = 12
    titlesize = 18
    legendsize = 12


    fig, axs = plt.subplots(1, sharex=True, figsize=(20, 10))
    fig.suptitle(casetitle,fontsize=titlesize)
    axs.plot(data11, data10, color='darkred', linestyle='dashed', linewidth=linewid, marker='o', markerfacecolor='red', markersize=marker, label=data10Name) # UL sweep main feat
    axs.plot(data31, data30, color='darkblue', linestyle='dashed', linewidth=linewid, marker='*',
             markerfacecolor='blue',
             markersize=marker, label=data30Name)
    axs.plot(data21, data20, color='darkgreen', linestyle='dashed', linewidth=linewid, marker='d', markerfacecolor='green',
                markersize=marker, label=data20Name)  # UL sweep main feat
    axs.set_xlabel(data11Name, fontsize=labelsize)

    axs.grid()
    axs.legend(fontsize=legendsize)
    axs.set_ylabel("mW")




    #plt.show()
    plt.savefig(filesave + '_plt_full_LTE', dpi=300, bbox_inches='tight')


if __name__ == "__main__":

    plotFull()

