###################################################################################################
############################################# MODULES #############################################
###################################################################################################
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

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

names = pd.read_csv('Datasets/attributes.csv', delim_whitespace=True)
data = pd.read_csv('Datasets/communities.data', names=names['names']).replace('?', np.nan)
#feat_miss = data.columns[data.isnull().any()]

print(data.shape)
print(data.head())

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
    def __init__(self, epochs=20, learning_rate=1e-6, X_valid = [], y_valid = []):
        #y = a*x + b
        self.a = 0
        self.b = 0
        self.epochs = epochs
        self.lr = learning_rate
        self.MSE_global = list()
        self.X_valid = X_valid
        self.y_valid = y_valid

    def predict(self, X):
        return self.a*X + self.b

    def MSE(self, y_pred, y):
        return np.square(y_pred - y).mean()

    def SGD(self, X, y):
        MSE_tmp = list()
        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            error = y_pred - y 

            grad_a = X*error/self.N
            grad_b = error/self.N

            self.a = self.a - self.lr * grad_a
            self.b = self.b - self.lr * grad_b
            
            MSE = self.MSE(y_pred, y)
            MSE_tmp.append(MSE)
            #print(f'    Epoch: {epoch+1}/{self.epochs}, MSE={MSE}')
        
        self.MSE_global.append(MSE_tmp)

    def SGD_plt(self, X, y):
        MSE_tmp = list()
        for epoch in range(self.epochs):
            y_pred = self.predict(X)

            error = y_pred - y 

            grad_a = X*error/self.N
            grad_b = error/self.N

            self.a = self.a - self.lr * grad_a
            self.b = self.b - self.lr * grad_b
            
            MSE = self.MSE(y_pred, y)
            MSE_tmp.append(MSE)

        self.MSE_global.append(MSE_tmp)
    
    def fit(self, X, y):
        self.N = len(X)
        for Xi, yi, i in zip(X, y, range(self.N)):
            self.SGD(Xi, yi)

    def fit_and_MSE_valid(self, X, y):
        self.N = len(X)
        for Xi, yi, i in zip(X, y, range(self.N)):
            self.SGD(Xi, yi)
            y_pred_valid = self.predict(self.X_valid)
            mse_valid = self.MSE(y_pred_valid, self.y_valid)
            print(f"MSE pour le jeu de validation après {i+1} données d'entrainement, 50 epochs par donnée et un pas de {self.lr} = {mse_valid}")

    def fit_and_plot_valid(self, X, y):
        self.N = len(X)
        figure_exo2_2, axis_exo2_2 = plt.subplots(3, 2)
        axis_exo2_2[0][0].scatter(self.X_valid, self.y_valid, label = "y_valid")
        axis_exo2_2[0][1].scatter(self.X_valid, self.y_valid, label = "y_valid")
        axis_exo2_2[1][0].scatter(self.X_valid, self.y_valid, label = "y_valid")
        axis_exo2_2[1][1].scatter(self.X_valid, self.y_valid, label = "y_valid")
        axis_exo2_2[2][0].scatter(self.X_valid, self.y_valid, label = "y_valid")
        for Xi, yi, i in zip(X, y, range(self.N)):
            self.SGD_plt(Xi, yi)
            y_valid_pred = self.predict(self.X_valid)
            
            if i == 0:
                axis_exo2_2[0][0].scatter(self.X_valid, y_valid_pred, label = "y_pred")
                axis_exo2_2[0][0].legend()
                axis_exo2_2[0][0].set_title(f"y_pred après entrainement sur {i+1} données")
            elif i == 75:
                axis_exo2_2[0][1].scatter(self.X_valid, y_valid_pred, label = "y_pred")
                axis_exo2_2[0][1].legend()
                axis_exo2_2[0][1].set_title(f"y_pred après entrainement sur {i+1} données")
            elif i == 150:
                axis_exo2_2[1][0].scatter(self.X_valid, y_valid_pred, label = "y_pred")
                axis_exo2_2[1][0].legend()
                axis_exo2_2[1][0].set_title(f"y_pred après entrainement sur {i+1} données")
            elif i == 225:
                axis_exo2_2[1][1].scatter(self.X_valid, y_valid_pred, label = "y_pred")
                axis_exo2_2[1][1].legend()
                axis_exo2_2[1][1].set_title(f"y_pred après entrainement sur {i+1} données")
            elif i == 299:
                axis_exo2_2[2][0].scatter(self.X_valid, y_valid_pred, label = "y_pred")
                axis_exo2_2[2][0].legend()
                axis_exo2_2[2][0].set_title(f"y_pred après entrainement sur {i+1} données")
        plt.show(False)


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
    print("-----------------")
    print("QUESTION 1")
    print("-----------------")
    model = LinearReg(epochs=50, learning_rate=1e-6, X_valid = X2_valid, y_valid = y2_valid)
    model.fit_and_MSE_valid(X2_train, y2_train)
    y2_train_pred = model.predict(X2_train)
    y2_train_MSE = MSE(y2_train_pred, y2_train)

    y2_valid_pred = model.predict(X2_valid)
    y2_valid_MSE = MSE(y2_valid_pred, y2_valid)

    print("\nOn remarque que le MSE diminue plus le nombre de données d'entrainement est grand, cependant le pas semble trop petit pour voir une nette diminution de la MSE")
    
    # question 2
    print("-----------------")
    print("QUESTION 2")
    print("-----------------")
    pas = [i/100 for i in range(20, 200, 5)]
    _MSE = []
    for i in pas :
        model = LinearReg(learning_rate = i, X_valid = X2_valid, y_valid = y2_valid)
        model.fit(X2_train, y2_train)

        y2_valid_pred = model.predict(X2_valid)
        y2_valid_MSE = MSE(y2_valid_pred, y2_valid)

        _MSE.append(y2_valid_MSE)
    
    for i in range(len(pas)):
        print(f"Jeu de validation, Epochs = 50, Pas = {pas[i]} --> MSE = {_MSE[i]}")

    figure_exo2_1, axis_exo2_1 = plt.subplots(1, 1) 
    axis_exo2_1.plot(pas, _MSE)
    axis_exo2_1.set_xlabel("pas")
    axis_exo2_1.set_ylabel("MSE")
    axis_exo2_1.set_title("MSE selon le pas utilisé et avec 50 epochs")
    plt.show(False)
    
    print("\nLe pas qui nous donne la MSE la plus faible est 0.95, nous le garderons donc par la suite.")
    print("-----")
    modelFinal = LinearReg(epochs = 1, learning_rate= 0.95, X_valid= X2_valid, y_valid= y2_valid)
    modelFinal.fit(X2_train, y2_train)

    y2_valid_pred = modelFinal.predict(X2_valid)
    y2_valid_MSE = MSE(y2_valid_pred, y2_valid)
    print(f"La MSE du jeu de test pour le modèle final avec 50 epochs et un pas de 0.9 est de : {y2_valid_MSE}")
    

    # question 3
    print("-----------------")
    print("QUESTION 3")
    print("-----------------")
    modelFinal.fit_and_plot_valid(X2_train, y2_train)

    plt.show()


'''if __name__ == '__main__':
    #exo1()
    #exo2()'''
    