from preprocessing import log_to_csv
from pathlib import Path
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import tkinter.messagebox

class Mainclass:
    def __init__(self):
        pass

def runGUI():
    # setting offsetpwrtime unit is in seconds
    #offsetpwrtime = 1.8 # case 131
    #offsetpwrtime = 2.415 # case 132
    #offsetpwrtime = 5 # case 133
    #offsetpwrtime = 1.63 # case 134
    #offsetpwrtime = 1.34 # case 135
    #offsetpwrtime = 1.524 # case 136
    #offsetpwrtime = 2.72 # case 137
    #offsetpwrtime = 1.53 # case 138
    #offsetpwrtime = 0 # case 139
    #offsetpwrtime = 0.5 # case 140
    #offsetpwrtime = 0.84 # case 141
    #offsetpwrtime = 2.35 # case 142
    #offsetpwrtime = 1.7 # case 143
    offsetpwrtime = 0

    hdegree = 0  #112

    root1 = tk.Tk() # create root window
    root1.title("UE Power Pre-processing GUI")
    root1.geometry('1000x500')
    root1.configure(bg='#006BE3')
    l = Label(root1, text="Pre-processing log from CMX to CSV ")
    l.config(font=("Courier", 14))
    l.pack()
    def onClick1():
        root = tk.Tk()  # select CMX500 log file
        root.withdraw()
        file_path = filedialog.askopenfilename()
        root.destroy()
        root2 = tk.Tk()  # select HV Monsoon power file
        root2.withdraw()
        file_path2 = filedialog.askopenfilename()
        root2.destroy()

        logdir = str(Path(file_path).parent) + '\\'
        #pwrdir = str(Path(file_path).parent) + '\\power_log\\'
        filenameout = os.path.splitext(os.path.basename(file_path))[0] + '_processed.csv'

        log2csvobj = log_to_csv.LogtoCsv(file_path, logdir, filenameout,file_path2,offsetpwrtime,hdegree) # define object
        log2csvobj.LogtoCsvDef() # execute program
        # text after run successfully
        outputtext = Text(root1, height=5, width=30)
        outputtext.insert(tk.END, 'Log to CSV has been done!!')
        outputtext.pack()



    # Create a log2CSV button
    btnSel = Button(root1, text="Select raw file", bg='white', command=onClick1)
    btnSel.place(x=300,y=200)
    # close button
    exit_button = Button(root1, text="Exit", bg='white', command=root1.destroy)
    exit_button.place(x=950, y=10)
    # execute program
    root1.mainloop()

    #fileid = str(Path(__file__).parent) + '\\measurementsLogs\\monitor_logging_Always[31]-129-att1.ptc'
    #logdir = str(Path(__file__).parent) + '\\measurementsLogs\\'


if __name__ == "__main__":
    print('..in main..')
    start = runGUI()

