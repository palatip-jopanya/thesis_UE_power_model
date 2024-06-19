import csv
import numpy as np
from pathlib import Path
import os

class MyCommonClass():

    def __init__(self):
        pass

    def readCSV(self, filename):
        file_open = open(filename, 'r')
        read_lines = file_open.readlines()
        # print(self.fileid)
        return read_lines

    def readCSVarr(self, fileid): # data, string first row
        with open(fileid, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
            data_firstrow = np.array(data)[0]
            data = np.delete(data, 0, 0)
            data_array = np.array(data, dtype=float)
        return data_array, data_firstrow

    def readCSVall(self, fileid): # data, string first row
        with open(fileid, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        return data

    def path2fileonly(self, fullpath):
        filenameout = os.path.splitext(os.path.basename(fullpath))[0]
        filenameWext, ext = os.path.splitext(fullpath)
        return filenameout

    def removeExt(self,filename):
        filename, file_extension = os.path.splitext(filename)
        return filename

    def findmid(self,arr):
        max = np.max(arr)
        min = np.min(arr)
        mid = (max - min)/2 + min
        return mid


