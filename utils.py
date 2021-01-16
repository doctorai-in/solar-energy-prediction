
import numpy as np
import math
import matplotlib as mpl
from matplotlib.image import imread
from random import randint
import theano
import keras
import pandas

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
import keras.utils
import keras.layers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import r2_score
import copy
import csv


#Set y values of data to lie between 0 and 1
def normalize_data(dataset, data_min, data_max):
    data_std = (dataset - data_min) / (data_max - data_min)
    test_scaled = data_std * (np.amax(data_std) - np.amin(data_std)) + np.amin(data_std)
    return test_scaled

#Import and pre-process data for future applications
def import_data(train_dataframe, dev_dataframe, test_dataframe):
    
    dataset = train_dataframe.values
    dataset = dataset.astype('float32')

    #Include all 12 initial factors (Year ; Month ; Hour ; Day ; Cloud Coverage ; Visibility ; Temperature ; Dew Point ;
    #Relative Humidity ; Wind Speed ; Station Pressure ; Altimeter
    max_test = np.max(dataset[:,12])
    min_test = np.min(dataset[:,12])
    scale_factor = max_test - min_test
    max = np.empty(13)
    min = np.empty(13)

    #Create training dataset
    for i in range(0,13):
        min[i] = np.amin(dataset[:,i],axis = 0)
        max[i] = np.amax(dataset[:,i],axis = 0)
        dataset[:,i] = normalize_data(dataset[:, i], min[i], max[i])

    train_data = dataset[:,0:12]
    train_labels = dataset[:,12]

    # Create dev dataset
    dataset = dev_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    dev_data = dataset[:,0:12]
    dev_labels = dataset[:,12]

    # Create test dataset
    dataset = test_dataframe.values
    dataset = dataset.astype('float32')

    for i in range(0, 13):
        dataset[:, i] = normalize_data(dataset[:, i], min[i], max[i])

    test_data = dataset[:, 0:12]
    test_labels = dataset[:, 12]

    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, scale_factor

#Save output predictions for graphing and inspection
def write_to_csv(prediction, filename):
    print("Writing to CSV...")
    with open(filename, 'w') as file:
        for i in range(prediction.shape[0]):
            file.write("%.5f" % prediction[i][0][0])
            file.write('\n')
    print("...finished!")

# Return MSE error values of all three data sets based on a single model
def evaluate(model, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, scale_factor):
    scores = model.evaluate(X_train, Y_train, verbose = 0) * scale_factor * scale_factor
    print("train: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_dev, Y_dev, verbose = 0) * scale_factor * scale_factor
    print("dev: ", model.metrics_names, ": ", scores)
    scores = model.evaluate(X_test, Y_test, verbose = 0) * scale_factor * scale_factor
    print("test: ", model.metrics_names, ": ", scores)

# Calculate MSE between two arrays of values
def mse(predicted, observed):
    return np.sum(np.multiply((predicted - observed),(predicted - observed)))/predicted.shape[0]