import matplotlib.pyplot as plt
import numpy
import numpy as np
from preprocessing import log_to_csv
from pathlib import Path
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
import csv
from sklearn.tree import DecisionTreeRegressor
from commonClass import MyCommonClass
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

def decisionTree(fileid,endc,mode,selected):
    common = MyCommonClass()
    filename = common.path2fileonly(fileid)
    filesave = common.removeExt(fileid)
    h = 360

    dataCsv = pd.read_csv(fileid)

    # independent
    dataY = dataCsv["UE_PowerMeas(mW)"].to_numpy()
    data1Name  = "UE_PowerMeas(mW)"
    mid = common.findmid(dataY)
    #print(mid)
    #print(dataY)
    ######### adjust value ##########
    dataY = [i-mid for i in dataY]
    dataY = numpy.array(dataY)

    # dependent data
    if selected == 0 and mode == "deg":
        #dataCsv.drop('UE_PowerMeas(mW)', inplace=True, axis=1) # now dataCsv is independent metrix
        dataX = dataCsv["H_degree"].to_numpy()
        data0Name = "H_degree"
    elif selected == 0 and endc == "1" and mode == "UL":
        dataX = dataCsv["NRULpwr(dBm)"].to_numpy()
        data0Name = "NR UL pwr(dBm)"
    elif selected == 0 and endc == "1" and mode == "DL":
        dataX = dataCsv["NRDLpwr(dBm)"].to_numpy()
        data0Name = "NR DL pwr(dBm)"
    elif selected == 0 and endc == "0" and mode == "UL":
        dataX = dataCsv["LTEULpwr(dBm)"].to_numpy()
        data0Name = "LTE UL pwr(dBm)"
    elif selected == 1 and endc == 1 and mode == "deg":
        data0Name = "H_degree"
        data1 = dataCsv["H_degree"].to_numpy()
        data2 = dataCsv["DL_NrAck%"].to_numpy()
        data3 = dataCsv["UL_NrAck%"].to_numpy()
        data4 = dataCsv["UL_NrDtx%"].to_numpy()
        dataX = np.stack((data1, data2, data3, data4), axis=-1)
    elif selected == 1 and endc == 1 and mode == "UL":
        data0Name = "NR UL pwr(dBm)"
        data1 = dataCsv["NRULpwr(dBm)"].to_numpy()
        data2 = dataCsv["NR_RSRP(dBm)"].to_numpy()
        data3 = dataCsv["NR_RSRQ(dBm)"].to_numpy()
        #data4 = dataCsv["UL_NrThp%"].to_numpy()
        dataX = np.stack((data1, data2, data3), axis=-1)
    elif selected == 1 and endc == 1 and mode == "DL":
        data0Name = "NR DL pwr(dBm)"
        data1 = dataCsv["NRDLpwr(dBm)"].to_numpy()
        data2 = dataCsv["NR_SINR(dBm)"].to_numpy()
        data3 = dataCsv["NR_RSRQ(dBm)"].to_numpy()
        data4 = dataCsv["NRDLpwr(dBm)"].to_numpy()
        dataX = np.stack((data1, data2, data3, data4), axis=-1)
    elif selected == 1 and endc == 0 and mode == "deg": # not works
        data0Name = "H_degree"
        data1 = dataCsv["H_degree"].to_numpy()
        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        data4 = dataCsv["DL_LteThp%"].to_numpy()
        dataX = np.stack((data1, data2, data3, data4), axis=-1)
    elif selected == 1 and endc == 0 and mode == "UL":
        data0Name = "LTEULpwr(dBm)"
        data1 = dataCsv["LTEULpwr(dBm)"].to_numpy()
        data2 = dataCsv["Lte_RSRP(dBm)"].to_numpy()
        data3 = dataCsv["Lte_RSRQ(dBm)"].to_numpy()
        #data4 = dataCsv["DL_LteThp%"].to_numpy()
        dataX = np.stack((data1, data2, data3), axis=-1)


    #layer = tf.keras.layers.Dense(units=1)  # nr of output = 1

    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=2)  # (X,Y)
    #print(X_train.shape[1])
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(data_scaled, target, test_size=0.2, random_state=42)

    # Define the model architecture
    nr_features = X_train.shape[1]
    row_train_data = X_train.shape[0]
    ######################### SETTING #####################
    n1 = 64
    n2 = n1
    act = 'relu' # 'linear' 'relu'
    epoch = 1000
    batch_size = 32
    ########################################################
    model = models.Sequential([
        layers.Dense(n1, activation=act, input_shape=(X_train.shape[1],)),
        layers.Dense(n2, activation=act),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_split=0.2)
    loss = history.history['loss']
    lenloss = len(loss)
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    Y_pred = model.predict(X_test)
    print('Test MAE:', test_mae)
    print('Test loss:', test_loss)


    ################
    y_train = [i + mid for i in y_train]
    y_test = [i + mid for i in y_test]
    Y_pred = [i + mid for i in Y_pred]
    ###############
    plt.rcParams.update({'font.size': 15})
    marker = 6
    linewid = 2
    ssize = 40
    labelsize = 15
    titlesize = 18
    legendsize = 15
    ############################

    plt.figure(figsize=(20, 10))
    plt.scatter(X_train[:, 0], y_train, s=ssize, edgecolor="black", c="darkorange", label='Train ')
    plt.scatter(X_test[:, 0], y_test, s=ssize, edgecolor="olivedrab", c="olivedrab", label='Test')
    plt.scatter(X_test[:, 0], Y_pred, s=ssize, edgecolor="red", c="purple", label='Predict')
    # plt.plot(x_test[:,0], y_pred1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    # plt.plot(x_test[:,0], y_pred2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel(data0Name, fontsize=labelsize)
    plt.ylabel(data1Name, fontsize=labelsize)
    casetitle = filename + '\n nr_features=' + str(nr_features) + ', Node_layer1=' +str(n1)+ ', Node_layer2=' +str(n2)+ ', row_training=' + str(row_train_data) + ',activation=' + act + ' \n epochs=' + str(epoch) + ' ,batch size =' + str(batch_size) + ', Test MAE=' + str(round(test_mae,2)) + ', Test loss=' + str(round(test_loss,2))
    plt.title(casetitle,fontsize=titlesize)
    plt.legend(fontsize=legendsize)
    plt.grid()
    if selected == 0:
        nameSave = filesave + '_' + mode + '_NeuralNet_SingleFeature'
    elif selected == 1:
        nameSave = filesave + '_' + mode + '_NeuralNet_MultiFeatures'
    #plt.show()
    plt.savefig(nameSave, dpi=300, bbox_inches='tight')

    #plt.figure(figsize=(10, 5))
    titlesize = 13
    plt.figure()
    x = np.linspace(1, lenloss, lenloss)
    plt.semilogy(x, loss, label="loss", linewidth=linewid, markersize=marker)
    plt.title(casetitle,fontsize=titlesize)
    plt.xlabel('epoch', fontsize=labelsize)
    plt.ylabel('loss', fontsize=labelsize)
    plt.legend(fontsize=legendsize)
    plt.grid()
    plt.savefig(nameSave + '_loss', dpi=300, bbox_inches='tight')
    #plt.show()


    #plt.savefig(filesave + '_NeuralNet_allfeatures_', dpi=300, bbox_inches='tight')
    #plt.savefig(filesave + '_NeuralNet_allfeatures_')


    #plt.savefig(filesave + '_decisionTree_3Feat', dpi=300, bbox_inches='tight')

    #plt.subplots(2,1,1)
    #plt.plot(history)







if __name__ == "__main__":
    root = tk.Tk()  # select measurement file
    root.withdraw()
    manual = 1 # 0-with path 1-select file manually
    if manual == 0:
        #file_path = "C:/Users/ejoppal/OneDrive - Ericsson/Documents/UEpowerModel_pythonProject/measurementsLogs/136_Mk4_ENDC_H180sweep_Lte64QAM_NrQPSK_processed_deg_rmvCol.csv"
        file_path = "C:/Users/ejoppal/OneDrive - Ericsson/Documents/UEpowerModel_pythonProject/measurementsLogs/137_Mk4_ENDC_H180sweep_Lte64QAM_Nr64QAM_processed_deg_rmvCol.csv"
    else:
        file_path = filedialog.askopenfilename()

    endc = 1 # o-lte 1-endc
    mode = 'deg' # UL, DL, deg
    selected = 1 # 0-single features, 1-selected features
    decisionTree(file_path,endc,mode,selected)


