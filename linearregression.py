# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from numpy import genfromtxt
from torchmetrics import ConfusionMatrix

class linearRegression(nn.Module):
    def __init__(self, input_arr, target_arr):
        super(linearRegression, self).__init__()
        # Format inputs
        # Use One Hot Encoding
        input_arr = (pd.get_dummies(pd.DataFrame(input_arr), columns=[0,1,2,3,5]))
        print(input_arr)
        input_arr = input_arr / input_arr.max(axis=0)
        self.inputs = torch.from_numpy(input_arr.values).float()
        print(self.inputs)
        self.targets = torch.from_numpy(target_arr)
        self.model = nn.Linear(np.shape(self.inputs)[1], np.shape(self.targets)[1])

    # Define a utility function to train the model. Trains for a given number of epochs.
    def __fit(self, train_dl, num_epochs, model, loss_fn, opt):
        for epoch in range(num_epochs):
            for xb, yb in train_dl:
                # Generate predictions
                pred = model(xb)
                loss = loss_fn(pred, yb)
                # Perform gradient descent
                loss.backward()
                opt.step()
                opt.zero_grad()
            print('Epoch: ', epoch, 'Loss: ', loss)
        print('Training loss: ', loss_fn(model(self.inputs), self.targets))

    def setWeights(self, weights):
        self.model.weight = torch.from_numpy(weights)

    def setBias(self, biases):
        self.model.bias = torch.from_numpy(biases)

    def trainModel(self):
        # Import tensor dataset & data loader
        from torch.utils.data import TensorDataset, DataLoader

        # Define dataset
        train_ds = TensorDataset(self.inputs, self.targets)
        train_ds[0:3]
        # Define data loader. Allows us to split data into batches and access rows from our inputs/targets as tuples
        batch_size = 64
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        next(iter(train_dl))

        print(self.model.weight)
        print(self.model.bias)

        # Define optimizer
        opt = torch.optim.SGD(self.model.parameters(), lr=5e-2)
        # Define loss function
        loss_fn = F.mse_loss
        loss = loss_fn(self.model(self.inputs), self.targets)
        print(loss)

        # Train the model for 1000 epochs
        self.__fit(train_dl, 100, self.model, loss_fn, opt)

        # Generate predictions
        preds = self.model(self.inputs)
        print(preds)
        # Compare with targets
        print(self.targets.int())
        confmat = ConfusionMatrix(3)
        print(confmat(preds, self.targets.int().contiguous()))



    def getModel(self):
        return self.model

    def saveModel(self, path='./model.pt'):
        torch.save(self.model, path)
