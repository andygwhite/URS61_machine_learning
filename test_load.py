# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import linearregression as lr
import csvprocessing as csv
from numpy import genfromtxt

model = torch.load('./model.pt')
model.eval()

process = csv.csvProcessor("all.csv")
inputs, targets = process.datatoArr()
inputs = inputs / inputs.max(axis=0)
for i in range(300):
    
    pred = model(torch.from_numpy(inputs[i])).data.numpy()
    print(i, pred, pred.argmax())