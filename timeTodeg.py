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

def time2Deg(fileid, deg):
    common = MyCommonClass()
    noext = common.removeExt(fileid)
    print(noext)
    data = pd.read_csv(fileid)
    #arr = data["H_degree"].to_numpy()
    length = len(data["H_degree"])
    if deg == 360:
        degree = np.linspace(-180, 180, num=length)
    elif deg == -180:
        degree = np.linspace(-180, 0, num=length)
    else:
        degree = np.linspace(0, deg, num=length)

    #data_updated = data["H_degree"].replace(degree)
    data_updated = data.assign(H_degree=degree)
    data_updated.to_csv(noext + '_deg.csv',index=False)





if __name__ == "__main__":
    common = MyCommonClass()

    root = tk.Tk()  # select measurement file
    root.withdraw()
    mode = 's' # m
    if mode == 's':
        file_path = filedialog.askopenfilename()
        if file_path.find('csv') != -1 and file_path.find('_180') != -1:
            time2Deg(file_path, 180)
        elif file_path.find('csv') != -1 and file_path.find('-180') != -1:
            time2Deg(file_path, -180)
        elif file_path.find('csv') != -1 and file_path.find('360') != -1:
            time2Deg(file_path, 360)
        elif file_path.find('csv') != -1 and file_path.find('80') != -1:
            time2Deg(file_path, 80)

    else:
        path_path = filedialog.askdirectory()
        for file in os.listdir(path_path):

            fullpath = path_path + '/' + file
            #fileout = common.removeExt(fullpath) + '_rmvCol.csv'

            if file.find('csv') != -1 and file.find('_180') != -1:
                time2Deg(fullpath, 180)
            elif file.find('csv') != -1 and file.find('-180') != -1:
                time2Deg(fullpath, -180)
            elif file.find('csv') != -1 and file.find('360') != -1:
                time2Deg(fullpath, 360)
            elif file.find('csv') != -1 and file.find('80') != -1:
                time2Deg(fullpath, 80)

