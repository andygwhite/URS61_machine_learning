# Import of the model
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
from numpy import genfromtxt

# Get input and label data from CSV
data = genfromtxt("networktestdata.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
my_data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
print(my_data)
my_label = data[:, [16]]
my_label = my_label.flatten()
print(my_label)


# Scale and split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(my_data, my_label, random_state=42)

# Create and fit the model
my_model = Pipeline([('scaler', StandardScaler()), ('my_model', LogisticRegression(C=1000))])
my_model.fit(X_train, y_train)  # apply scaling on training data
print(my_model.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.

joblib.dump(my_model, "./second_stage_logistic_regression.joblib")

# Uncomment for hyperparameter tuning. 
""" # defining parameter range
param_grid = {'my_model__C': [0.1, 1, 10, 100, 1000], 
            'my_model__tol': [1e-4, 1e-5, 1e-6],
            'my_model__max_iter': [1, 10, 100, 1000, 10000],
            'my_model__solver': ['lbfgs', 'newton-cholesky']} 

grid = GridSearchCV(my_model, param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
print("Best Score: ", grid.best_estimator_)
print("Best Params: ", grid.best_params_) """

# Uncomment to run against a unique test set of data entries and print results to CSV for comparison.
""" data = genfromtxt("exampletest.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
my_data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
print(my_data)
my_label = data[:, [16]]
my_label = my_label.flatten()
print(my_label)

with open('lrresults.csv', 'w', ) as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    count = 0
    total = 0
    for i in range(len(my_data)):
        prediction = my_model.predict(my_data[i].reshape(1,-1)).flatten()
        if(my_label[i] == prediction):
            count = count + 1
        total = total + 1
        wr.writerow([my_label[i], prediction])
    print(count / total) """