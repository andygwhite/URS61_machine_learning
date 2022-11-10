# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
from numpy import genfromtxt

class csvProcessor:
    def __init__(self, filename):
        self.filename = filename

    def datatoArr(self):
        data = genfromtxt(self.filename, delimiter=',', skip_footer=0, encoding="utf-8-sig", dtype='float32')
        # data = data[0:10000, :]
        data = np.array(random.sample(list(data),100000))
        inputs = data[:, [0, 1, 2, 3, 4, 5]]
        targets = data[:, [6, 7, 8]]
        return inputs, targets

    def writetoCSV(self, model):
        weights_arr = model.weight.cpu().detach().numpy()
        bias_arr = model.bias.cpu().detach().numpy()
        weights_arr.tofile("weights.csv", sep=',')
        bias_arr.tofile("bias.csv", sep=',')