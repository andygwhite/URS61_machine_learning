# Import Numpy & PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from numpy import genfromtxt
# from torchmetrics import ConfusionMatrix

class linearRegression(nn.Module):
    def __init__(self, input_arr, target_arr):
        super(linearRegression, self).__init__()
        # Format inputs
        input_arr = input_arr / input_arr.max(axis=0)
        self.inputs = torch.from_numpy(input_arr).float()
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
        self.__fit(train_dl, 10, self.model, loss_fn, opt)

        # Generate predictions
        preds = self.model(self.inputs)
        print(preds)
        # Compare with targets
        print(self.targets.int())
        # confmat = ConfusionMatrix(3)
        # print(confmat(preds, self.targets.int().contiguous()))



    def getModel(self):
        return self.model

    def saveModel(self, path='./second_stage_lr_model_epochs_10.pt'):
        torch.save(self.model, path)

data = genfromtxt("/home/mininet/machine_learning/second_stage/networktestdata_encoded.csv", delimiter=',', skip_footer=0, encoding="utf-8-sig", dtype='float32')
# data = data[0:10000, :]
# data = np.array(random.sample(list(data),100000))

inputs = data[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
targets = data[:, [17,18,19,20]]

lr_model = linearRegression(inputs, targets)
lr_model.trainModel()
lr_model.saveModel(path='./second_stage_lr_model_epochs_10.pt')
model = lr_model.getModel()

data_test = genfromtxt("/home/mininet/machine_learning/networkvalidationdata.csv", delimiter=',', skip_footer=0, encoding="utf-8-sig", dtype='float32')
inputs = data_test[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
targets = data_test[:, [16,17,18,19]]
inputs = inputs / inputs.max(axis=0)
model.eval()
correct = 0
for pt, target in zip(inputs, targets):
    pred = model(torch.from_numpy(pt).float()).data.numpy()
    if pred.argmax() == target.argmax():
        correct += 1
print(correct/len(inputs))
