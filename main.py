# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import linearregression as lr
import csvprocessing as csv
from numpy import genfromtxt

process = csv.csvProcessor("data1shuffled.csv")
inputs, targets = process.datatoArr()

lr_model = lr.linearRegression(inputs, targets)
lr_model.trainModel()
lr_model.saveModel()
model = lr_model.getModel()

process.writetoCSV(model)