# Import of the model
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import csvprocessing as csv
import pandas as pd
import joblib
from numpy import genfromtxt

# Initialisation of the model
my_model = RandomForestRegressor(n_estimators = 100, random_state = 0)

# Definition of the input data
import torch
#my_data = np.array([[0,1,2.4],[1,2,1],[4,2,0.2],[8,3,0.4], [4,1,0.4]])
#my_label = np.array([0,1,0,0,1])

data = genfromtxt("/home/mininet/network_topo/labeled_datasets/training/all.csv", delimiter=',', skip_footer=0, encoding="utf-8-sig", dtype='float32')
# data = data[0:10000, :]
# data = np.array(random.sample(list(data),100000))
my_data = data[:, [0, 1, 2, 3, 4, 5, 6]]
my_label = data[:, [10]]
# One Hot Encode
my_data = (pd.get_dummies(pd.DataFrame(my_data), columns=[0,1,2,4,5]))
my_data = my_data / my_data.max(axis=0)
# my_label = my_label.flatten()
print(my_label.shape)
print(my_data.shape)

# Fitting function
my_model.fit(my_data, my_label)

# Save model
joblib.dump(my_model, "./random_forest.joblib")

# # Prediction function
# data = genfromtxt("data1shuffled.csv", delimiter=',', skip_header=10000, encoding="utf-8-sig", dtype='float32')
# print(data.shape)
# test_inputs = data[:, [0, 1, 2, 3, 4, 5]]
# #targets = data2[:, [6, 7]]
# test_labels = data[:, [7]]

# total = 0
# count = 0
# correct = 0
# for row in test_inputs:
#     result = my_model.predict(test_inputs[count].reshape(1,-1))
#     if (test_labels[count] and result >= 0.5):
#         correct = correct + 1
#     elif (not test_labels[count] and result < 0.5):
#         correct = correct + 1
#     total = total + 1
#     count = count + 1

# print("Accuracy: ", correct / total)