# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import csv
import first_stage.linearregression_trainer as lr
import csvprocessing as csv
import pandas as pd
import joblib
from numpy import genfromtxt

model = joblib.load("./random_forest.joblib")
MAPPING = {0: "cpu", 1: "network", 2: "memory"}
process = csv.csvProcessor("/home/mininet/network_topo/labeled_datasets/validation/all.csv")
inputs, targets = process.datatoArr()
inputs = (pd.get_dummies(pd.DataFrame(inputs), columns=[0,1,2,4,5]))
inputs = (inputs / inputs.max(axis=0))
print(inputs)
correct = 0
count = 0
conf_matrix = {label: {lb: 0 for lb in ["cpu", "network", "memory"]} for label in ["cpu", "network", "memory"]}
for in_arr, target_arr in zip(inputs.values, targets):
    # print(in_arr)
    pred = model.predict(in_arr.reshape(1,-1)).flatten()
    count += 1
    conf_matrix[MAPPING[target_arr.argmax()]][MAPPING[pred.argmax()]] += 1
    correct += 1 if target_arr.argmax() == pred.argmax() else 0
print("Confusion Matrix:")
print(pd.DataFrame.from_dict(conf_matrix).to_markdown())
print("Num correct:", correct)
print("Total count:", count)
print("Accuracy:", 100*correct/count, "%")