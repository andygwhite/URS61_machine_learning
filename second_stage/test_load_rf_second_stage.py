# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import csv
import linearregression as lr
# import csvprocessing as csvproc
import csv
import pandas as pd
import joblib
from numpy import genfromtxt

def accuracyCheck(rf_model, test_inputs, test_labels):
    total = 0
    count = 0
    correct = 0
    result_mat = []

    for row in test_inputs:
        result = rf_model.predict(test_inputs[count].reshape(1,-1))
        result_mat.append(result)
        # error = abs((test_labels[count] - result) / (test_labels[count] + 1))
        # if (error <= 0.15):
        #     correct = correct + 1
        # # print(f"Input: {row}, Result: {result}")
        # total = total + 1
        # count = count + 1
        
    with open('networkresultsdatarf.csv', 'w', ) as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in range(len(test_labels)):
                wr.writerow([test_labels[i], result_mat[i]])

    # print("Accuracy: ", correct / total)

# Prediction function
data2 = genfromtxt("/home/mininet/network_topo/second_stage_labeled_datasets/validation/networkvalidationdata.csv", delimiter=',', skip_header=10000, encoding="utf-8-sig", dtype='float32')
print(data2.shape)
test_inputs = data2[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
test_labels = data2[:, [16,17,18,19]]

accuracyCheck(joblib.load("./second_stage_rf.joblib"), test_inputs, test_labels)