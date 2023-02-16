# Import of the model
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import linearregression as lr
import csvprocessing as csv
from numpy import genfromtxt

# Initialisation of the model
my_model = RandomForestRegressor(n_estimators = 100, random_state = 0)

# Definition of the input data
import torch
#my_data = np.array([[0,1,2.4],[1,2,1],[4,2,0.2],[8,3,0.4], [4,1,0.4]])
#my_label = np.array([0,1,0,0,1])
process = csv.csvProcessor("data1shuffled.csv")
my_data, my_label = process.datatoArr()
my_label = my_label.flatten()
print(my_label.shape)

# Fitting function
my_model.fit(my_data, my_label)

# Prediction function
data = genfromtxt("data1shuffled.csv", delimiter=',', skip_header=10000, encoding="utf-8-sig", dtype='float32')
print(data.shape)
test_inputs = data[:, [0, 1, 2, 3, 4, 5, 6]]
#targets = data2[:, [6, 7]]
test_labels = data[:, [7]]

total = 0
count = 0
correct = 0
for row in test_inputs:
    result = my_model.predict(test_inputs[count].reshape(1,-1))
    if (test_labels[count] and result >= 0.5):
        correct = correct + 1
    elif (not test_labels[count] and result < 0.5):
        correct = correct + 1
    total = total + 1
    count = count + 1

print("Accuracy: ", correct / total)