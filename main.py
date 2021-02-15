###################################################################################################
############################################# MODULES #############################################
###################################################################################################
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

PLOT_1 = False
PLOT_2 = False

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
    X1_train / np.linalg.norm(X1_train)
    y1_train = data[:, 1]

with open('Datasets/Dataset_1_valid.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_valid = data[:, 0]
    X1_valid / np.linalg.norm(X1_valid)
    y1_valid = data[:, 1]

with open('Datasets/Dataset_2_test.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X2_test = data[:, 0]
    X2_test / np.linalg.norm(X2_test)
    y2_test = data[:, 1]

with open('Datasets/Dataset_2_train.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X2_train = data[:, 0]
    X2_train / np.linalg.norm(X1_train)
    y2_train = data[:, 1]

with open('Datasets/Dataset_2_valid.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X2_valid = data[:, 0]
    X2_valid / np.linalg.norm(X1_valid)
    y2_valid = data[:, 1]

#########################################################################################################
############################################ FUNCTIONS EXO 1 ############################################
#########################################################################################################
def transform(X):
    vandermonde = np.ones((50, 1))
    for j in range(1, 21):
        x_pow = np.power(X, j)
        vandermonde = np.append(vandermonde, x_pow.reshape(-1, 1), axis=1)
    return vandermonde

def fit(X, y, epoch=100):
    w = np.zeros(21)
    vandermonde = transform(X)
    for i in range(epoch):
        y_pred = predict(X, w)
        w = w - (1/50)*(vandermonde.T @ (y_pred - y))
    return w

def fit_L2(X, y, L2, epoch=50):
    w = np.zeros(21)
    vandermonde = transform(X)
    for i in range(epoch):
        y_pred = predict(X, w)
        w = w - (0.5*(vandermonde.T @ (y_pred - y)) + (L2 * w))/50
    return w

def predict(X, w):
    return transform(X) @ w

def MSE(y_pred, y):
    return np.square(y_pred - y).mean()

#########################################################################################################
############################################ FUNCTIONS EXO 2 ############################################
#########################################################################################################
class LinearReg:
    def __init__(self, epochs=20, learning_rate=1e-6):
        #y = a*x + b
        self.a = 0
        self.b = 0
        self.epochs = epochs
        self.lr = learning_rate
        self.MSE_global = list()

    def predict(self, X):
        return self.a*X + self.b

    def MSE(self, y_pred, y):
        return np.square(y_pred - y).mean()

    def fit(self, X, y):
        MSE_tmp = list()
        N = len(X)
        for epoch in tqdm(range(self.epochs)):
            y_pred = self.predict(X)

            grad_a = sum(X*(y - y_pred))/N
            grad_b = sum(y - y_pred)/N

            self.a = self.a - self.lr * grad_a
            self.b = self.b - self.lr * grad_b
            
            MSE = self.MSE(y_pred, y)
            MSE_tmp.append(MSE)
            print(f'MSE={MSE}')
        
        self.MSE_global.append(MSE_tmp)

#########################################################################################################
############################################ FUNCTIONS EXO 3 ############################################
#########################################################################################################

###################################################################################################
############################################## CODE ###############################################
###################################################################################################
def exo1():
    w = fit(X1_train, y1_train)
    y1_valid_pred = predict(X1_valid, w)
    y1_train_pred = predict(X1_train, w)

    MSE_valid = MSE(y1_valid_pred, y1_valid)
    MSE_train = MSE(y1_train_pred, y1_train)


    ############################################### L2 ################################################
    y1_valid_pred_L2s, y1_train_pred_L2s = [], []
    MSE_valids, MSE_trains = [], []
    N = 100
    L2_factors = [i/N for i in range(N)]

    for i in range(N):
        L2_factor = i/N
        w_L2 = fit_L2(X1_train, y1_train, L2_factor)
        y1_valid_pred_L2 = predict(X1_valid, w_L2)
        y1_train_pred_L2 = predict(X1_train, w_L2)

        y1_valid_pred_L2s.append(y1_valid_pred_L2)
        y1_train_pred_L2s.append(y1_train_pred_L2)

        MSE_valids.append(MSE(y1_valid_pred_L2, y1_valid))
        MSE_trains.append(MSE(y1_train_pred_L2, y1_train))

    '''L2 = L2_factors[MSE_valids.index(min(MSE_valids))]
    w_L2 = fit_L2(X1_test, y1_test, L2)
    y1_test_pred_L2 = predict(X1_test, w_L2)
    MSE_test = MSE(y1_test_pred_L2, y1_test)'''

    ###################################################################################################
    ############################################# GRAPHS ##############################################
    ###################################################################################################
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
    plt.scatter(L2_factors, MSE_valids, label='valid_set')
    plt.scatter(L2_factors, MSE_trains, label='train_set')
    plt.legend()
    plt.title(f'MSE en fonction de l\'hyperparametre $lambda$')
    plt.xlabel('$lambda$')
    plt.ylabel('MSE')
    plt.show()
    
def exo2():
    model = LinearReg()
    