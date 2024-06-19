import pandas as pd
from commonClass import MyCommonClass
from preprocessing import log_to_csv
from pathlib import Path
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def csvTocorr(fileID):

    print(fileID)
    common = MyCommonClass()
    #data, firstrow = common.readCSVarr(fileID) # data, string first row
    #print(data)
    dataFrame = pd.read_csv(fileID)
    #filenameOnly = common.path2fileonly(fileID)
    fileonly = common.path2fileonly(fileID)

    #corr_matrix = np.corrcoef(np.transpose(data)).round(decimals=2)
   # {‘pearson’, ‘kendall’, ‘spearman’} or callable
    #type = "Kendall"
    #type = "Spearman"
    type = "Pearson"

    if fileonly.find("ENDC") == -1:
        dataFrameSel = dataFrame[["UE_PowerMeas(mW)","H_degree","LTEULpwr(dBm)","LTEDLpwr(dBm) ","Lte_RSRP(dBm)","Lte_RSRQ(dBm)","DL_LteAck%","DL_LteNack%","DL_LteDtx%","DL_LteBLER%","UL_LteAck%","UL_LteNack%","UL_LteDtx%","UL_LteBLER%"]]
        corr_matrix = dataFrameSel.corr(method=type.lower())
    else:
        dataFrameSel = dataFrame[
            ["UE_PowerMeas(mW)", "H_degree", "LTEULpwr(dBm)", "LTEDLpwr(dBm) ", "Lte_RSRP(dBm)", "Lte_RSRQ(dBm)",
             "DL_LteAck%", "DL_LteNack%", "DL_LteDtx%", "DL_LteBLER%", "UL_LteAck%", "UL_LteNack%", "UL_LteDtx%",
             "UL_LteBLER%","NRULpwr(dBm)","NRDLpwr(dBm)","NR_RSRP(dBm)","NR_RSRQ(dBm)","NR_SINR(dBm)","DL_NrAck%","DL_NrNack%","DL_NrDtx%","DL_NrBLER%","UL_NrAck%","UL_NrNack%","UL_NrDtx%","UL_NrBLER%"]]
        corr_matrix = dataFrameSel.corr( method=type.lower())


    #sns.heatmap(corr_matrix, annot=True)

    fileout = common.removeExt(fileID) + '_' + type + '_corrMatrix'
    fileout_uepwr = common.removeExt(fileID) + '_' + type + '_UE_pwr_corr'

    plt.figure(figsize=(25, 14))
    #mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool_))
    heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='BrBG') # mask=mask,
    heatmap.set_title(type + ' correlation heatmap ' + fileonly, fontdict={'fontsize': 18}, pad=12)
    plt.savefig(fileout, dpi=300, bbox_inches='tight')
    plt.close()
    #plt.show()

    ##plot only power measurement feature
    powermeas_corr = dataFrameSel.corr()[['UE_PowerMeas(mW)']].sort_values(by='UE_PowerMeas(mW)', ascending=False)
    plt.figure(figsize=(8, 12))
    heatmap = sns.heatmap(powermeas_corr, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title(type + ' correlation heatmap \n' + fileonly, fontdict={'fontsize': 18}, pad=12)
    plt.savefig(fileout_uepwr, dpi=300, bbox_inches='tight')
    plt.close()


def callfunc(mode):
    root = tk.Tk()  # select measurement file
    root.withdraw()
    if mode == 's':
        path_path = filedialog.askopenfilename()
        #fullpath = path_path + '/' + file
        csvTocorr(path_path)
    elif mode == 'm':
        path_path = filedialog.askdirectory()
        for file in os.listdir(path_path):
            if file.find('csv') != -1 :
                fullpath = path_path + '/' + file
                csvTocorr(fullpath)



if __name__ == "__main__":
    endc = 0
    callfunc('m')  # s-single m-multi

