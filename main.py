import numpy as np
import csv

with open('Datasets\Dataset_1_test.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_test = data[:, 0]
    y1_test = data[:, 1]

with open('Datasets\Dataset_1_train.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_train = data[:, 0]
    y1_train = data[:, 1]

with open('Datasets\Dataset_1_valid.csv') as csvfile:
    data = np.array(list(csv.reader(csvfile)))
    data = data[:, :2].astype(np.float)
    X1_valid = data[:, 0]
    y1_valid = data[:, 1]

def poly_20(x, weight_list):
    acc = weight_list[0]
    for n in range(1, 21):
        acc += (x**n)*weight_list[n]
    return acc