###################################################################################################
############################################# MODULES #############################################
###################################################################################################
import numpy as np
import csv
import matplotlib.pyplot as plt

PLOT_1 = True
PLOT_2 = True

###################################################################################################
########################################### DATAS IMPORT ##########################################
###################################################################################################
with open('Datasets/Dataset_1_test.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_test = data[:, 0]
    y1_test = data[:, 1]

with open('Datasets/Dataset_1_train.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_train = data[:, 0]
    y1_train = data[:, 1]

with open('Datasets/Dataset_1_valid.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_valid = data[:, 0]
    y1_valid = data[:, 1]

###################################################################################################
############################################ FUNCTIONS ############################################
###################################################################################################
def transform(X):
    vandermonde = np.ones((50, 1))
    for j in range(1, 21):
        x_pow = np.power(X, j)
        vandermonde = np.append(vandermonde, x_pow.reshape(-1, 1), axis=1)
    return vandermonde

def fit(X, y, epoch=200):
    w = np.zeros(21)
    for i in range(epoch):
        vandermonde = transform(X)
        y_pred = predict(X, w)
        error = y_pred - y
        w = w - (1/50)*(vandermonde.T @ error)
    return w

def fit_L2(X, y, L2_factor, epoch=1):
    w = np.zeros(21)
    for i in range(epoch):
        vandermonde = transform(X)
        y_pred = predict(X, w)
        error = y_pred - y
        L2_term = np.sum(np.square(w))
        w = w - (1/50)*(vandermonde.T @ error) + L2_factor*w
    return w

def predict(X, w):
    return np.dot(transform(X), w)

def MSE(y_pred, y_test):
    return np.square(np.subtract(y_pred, y_test)).mean()

###################################################################################################
############################################## CODE ###############################################
###################################################################################################
w = fit(X1_train, y1_train)
y1_valid_pred = predict(X1_valid, w)
y1_train_pred = predict(X1_train, w)

MSE_valid = MSE(y1_valid_pred, y1_valid)
MSE_train = MSE(y1_train_pred, y1_train)
print(f'MSE for train set : {MSE_train}\nMSE for valid set : {MSE_valid}')

############################################### L2 ################################################
y1_valid_pred_L2s, y1_train_pred_L2s = [], []
MSE_valids, MSE_trains = [], []

for i in range(1000):
    L2_factor = i/1000
    w_L2 = fit_L2(X1_train, y1_train, L2_factor)
    y1_valid_pred_L2 = predict(X1_valid, w_L2)
    y1_train_pred_L2 = predict(X1_train, w_L2)

    y1_valid_pred_L2s.append(y1_valid_pred_L2)
    y1_train_pred_L2s.append(y1_train_pred_L2)

    MSE_valids.append(MSE(y1_valid_pred_L2, y1_valid))
    MSE_trains.append(MSE(y1_train_pred_L2, y1_train))

###################################################################################################
############################################# GRAPHS ##############################################
###################################################################################################
if PLOT_1:
    figure, axis = plt.subplots(2, 1) 
    axis[0].scatter(X1_train, y1_train, label='y_train')
    axis[0].scatter(X1_train, y1_train_pred, label='y_pred')
    axis[0].legend()
    axis[0].set_title(f'Regression Polynomiale De Degrée 20 Sur Train\nMSE={MSE_train}')

    axis[1].scatter(X1_valid, y1_valid, label='y_valid')
    axis[1].scatter(X1_valid, y1_valid_pred, label='y_pred')
    axis[1].legend()
    axis[1].set_title(f'Regression Polynomiale De Degrée 20 Sur Valid\nMSE={MSE_valid}')
    plt.show()

############################################### L2 ################################################
if PLOT_2:
    pass