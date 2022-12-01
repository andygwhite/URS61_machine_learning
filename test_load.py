# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import csv
import linearregression as lr
import csvprocessing as csv
import pandas as pd
from numpy import genfromtxt

model = torch.load('./model_DEMO.pt')
model.eval()

process = csv.csvProcessor("/home/mininet/network_topo/labeled_datasets_DEMO/validation/all.csv")
inputs, targets = process.datatoArr()
inputs = (pd.get_dummies(pd.DataFrame(inputs), columns=[0,1,2,5]))
inputs = (inputs / inputs.max(axis=0))
correct = 0
count = 0
for in_arr, target_arr in zip(inputs.values, targets):
    pred = model(torch.from_numpy(in_arr).float()).data.numpy()
    # print(in_arr)
    # print(pred, pred.argmax())
    count += 1
    if pred.argmax() == target_arr.argmax():
        correct += 1
print("Num correct:", correct)
print("Total count:", count)
print("Accuracy:", 100*correct/count, "%")