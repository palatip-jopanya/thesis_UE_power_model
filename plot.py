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

def selection(firstrow,arr,listSelection):
    m = 0
    for i in listSelection:
        index = firstrow.tolist().index(i)
        selected = arr[:,index]
        print(selected)
        if m == 1:
            selected = np.stack(old,selected)
        old = selected
        m = 1
        #combined = np.append(selected,tobeadded,axis=1)
    return selected
def plot():
    #define including variable
    #listSelection = ["Power Meas ","Lte_RSRP(dBm)","NR_RSRP(dBm)"]

    root = tk.Tk()  # select measurement file
    root.withdraw()
    file_path = filedialog.askopenfilename()
    arr, firstrow = readcsv(file_path)
    print(arr.shape)
    length = len(arr)

    t = np.arange(1, length+1, 1)

    data0_time = arr[:, 0]
    data1_DL_LteThpbps = arr[:, 1]
    data2_DL_NrDlThpbps = arr[:, 2]
    data3_DL_LteAck = arr[:, 3]
    data4_DL_LteNack = arr[:, 4]
    data5_DL_LteDtx = arr[:, 5]
    data6_DL_LteBLER = arr[:, 6]
    data7_DL_LteThp = arr[:, 7]
    data8_DL_NrAck = arr[:, 8]
    data9_DL_NrNack = arr[:, 9]
    data10_DL_NrDtx = arr[:, 10]
    data11_DL_NrBLER = arr[:, 11]
    data12_DL_NrThp = arr[:, 12]
    data13_UL_LteAck = arr[:, 13]
    data14_UL_LteNack = arr[:, 14]
    data15_UL_LteDtx = arr[:, 15]
    data16_UL_LteBLER = arr[:, 16]
    data17_UL_NrAck = arr[:, 17]
    data18_UL_NrNack = arr[:, 18]
    data19_UL_NrDtx = arr[:, 19]
    data20_UL_NrBLER = arr[:, 20]
    data21_UL_LteThp = arr[:, 21]
    data22_UL_LteTphbps = arr[:, 22]
    data23_UL_NrThp = arr[:, 23]
    data24_UL_NrTphbps = arr[:, 24]
    data25_Lte_RSRPdBm = arr[:, 25]
    data26_Lte_RSRQdBm = arr[:, 26]
    data27_NR_RSRPdBm = arr[:, 27]
    data28_NR_RSRQdBm = arr[:, 28]
    data29_NR_SINRdBm = arr[:, 29]
    data30_PowerMeasmW = arr[:, 30]
    data31_NRULpwrdBm = arr[:, 31]
    data32_NRDLpwrdBm = arr[:, 32]
    data33_LTEULpwrdBm = arr[:, 33]
    data34_LTEDLpwrdBm = arr[:, 34]

    #print(type(data34_LTEDLpwrdBm))
    #[float(string) for string in string_array]
    data0 = np.array(data30_PowerMeasmW,dtype=float)  # [float(string) for string in data30_PowerMeasmW] #data30_PowerMeasmW.astype(np.float)  #arr[:,2] #UE Power

    data1 = np.array(data25_Lte_RSRPdBm,dtype=float) #[float(string) for string in data25_Lte_RSRPdBm] #.astype(np.float) #arr[:,0] #LTE RSRP
    data2 = np.array(data26_Lte_RSRQdBm,dtype=float) #[float(string) for string in data26_Lte_RSRQdBm] #data27_NR_RSRPdBm.astype(np.float) #arr[:,1] #NR RSRP
    data3 = np.array(data33_LTEULpwrdBm,dtype=float) #[float(string) for string in data33_LTEULpwrdBm] #data31_NRULpwrdBm.astype(np.float) #arr[:, 3]  # NR UL power

    #print(get_var_name())
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('report')
    ax1.set_ylabel('RSRP (dBm)', color=color)
    ax1.plot(t, data1, 'ro')
    ax1.plot(t, data2, 'm*')
    #ax1.plot(t, data3, 'y*')
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('UE Power (mW)', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data0, 'bh')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    ax1.legend(["a","b","c"])
    ax2.legend(["UE Power (mW)"])
    plt.show()




if __name__ == "__main__":
    print('..in main..')
    start = plot()




