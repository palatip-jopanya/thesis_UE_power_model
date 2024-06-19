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
def csvRmvColLte(filepath, fileout):
    data = pd.read_csv(filepath)
    print(data)
    data.drop('DL_LteThp(bps)', inplace=True, axis=1)
    data.drop('DL_NrDlThp(bps)', inplace=True, axis=1)
    data.drop('UL_LteTph(bps)', inplace=True, axis=1)
    data.drop('UL_NrTph(bps)', inplace=True, axis=1)
    print(data)
    data.to_csv(fileout, index=False)


def csvRmvColENDC(filepath, fileout):
    data = pd.read_csv(filepath)
    print(data)
    data.drop('DL_LteThp(bps)', inplace=True, axis=1)
    data.drop('DL_NrDlThp(bps)', inplace=True, axis=1)
    data.drop('UL_LteTph(bps)', inplace=True, axis=1)
    data.drop('UL_NrTph(bps)', inplace=True, axis=1)
    data.to_csv(fileout, index=False)


if __name__ == "__main__":
    common = MyCommonClass()
    root = tk.Tk()  # select measurement file
    root.withdraw()

    mode = 1 # single =1 multi=0
    if mode == 1:
        file_path = filedialog.askopenfilename()
        fileout = common.removeExt(file_path) + '_rmvCol.csv'
        csvRmvColLte(file_path, fileout)

    elif mode == 0:
        # file_path = filedialog.askopenfilename()
        path_path = filedialog.askdirectory()
        for file in os.listdir(path_path):

            fullpath = path_path + '/' + file
            fileout = common.removeExt(fullpath) + '_rmvCol.csv'

            if file.find('csv') != -1 and file.find('LTEonly') != -1:
                csvRmvColLte(fullpath, fileout)
            elif file.find('csv') != -1 and file.find('Lteonly') != -1:
                csvRmvColLte(fullpath, fileout)
            elif file.find('csv') != -1 and file.find('ENDC') != -1:
                csvRmvColENDC(fullpath, fileout)