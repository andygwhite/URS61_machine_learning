# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import linearregression as lr
import csvprocessing as csv
from numpy import genfromtxt

process = csv.csvProcessor("/home/mininet/network_topo/labeled_datasets_DEMO/training/all.csv")
inputs, targets = process.datatoArr()

lr_model = lr.linearRegression(inputs, targets)
lr_model.trainModel()
lr_model.saveModel(path="./model_DEMO.pt")
model = lr_model.getModel()

process.writetoCSV(model)