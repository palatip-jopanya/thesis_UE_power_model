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



def plotFull(file_path1,file_path2):
    common = MyCommonClass()
    filename = common.path2fileonly(file_path1)
    filesave = common.removeExt(file_path1)
    casetitle = filename

    dataCsv1 = pd.read_csv(file_path1)
    data0 = dataCsv1["Lte_RSRP(dBm)"].to_numpy()
    data0Name  = "Lte_RSRP (dBm)"

    dataCsv2 = pd.read_csv(file_path2)
    data1 = dataCsv2["NR_RSRP(dBm)"].to_numpy()
    data1Name = "NR_RSRP (dBm)"

    data2 = dataCsv2["NR_SINR(dBm)"].to_numpy()
    data2Name = "NR_SINR (dB))"


    ############ Plot ###############

    marker = 6
    linewid = 2
    ssize = 30
    labelsize = 12
    titlesize = 18
    legendsize = 12

    ################################

    fig, axs = plt.subplots(1,2, subplot_kw=dict(projection='polar'), figsize=(20, 10))
    #fig.suptitle(casetitle,fontsize=titlesize)
    len1 = len(data0)
    x1 = np.linspace(0, 2*np.pi,len1)
    axs[0].scatter(x1, data0,  s=ssize, edgecolor="darkred", c="red", label=data0Name) # UL sweep main feat
    axs[0].grid()
    axs[0].legend(fontsize=legendsize)
    axs[0].set_ylabel("dBm", fontsize=labelsize)
    axs[0].grid()
    axs[0].set_title("LTE-64QAM RSRP (dBm) in horizontal degree for full circle")

    len2 = len(data1)
    x2 = np.linspace(0, 1*np.pi, len2)
    axs[1].scatter(x2, data1, s=ssize, edgecolor="darkblue", c="navy", label=data1Name) # RSRP
    axs[1].grid()
    axs[1].legend(fontsize=legendsize)
    axs[1].set_ylabel("dBm", fontsize=labelsize)
    axs[1].grid()
    axs[1].set_title("NR-64QAM RSRP (dBm) in horizontal degree for half circle")


    #plt.show()
    plt.savefig(filesave + '_plt_full_LTE', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path1 = filedialog.askopenfilename()
    file_path2 = filedialog.askopenfilename()
    plotFull(file_path1,file_path2)
