# Import of the model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
from numpy import genfromtxt
import joblib

# Get input and label data from CSV
train_data = genfromtxt("/home/mininet/network_topo/labeled_datasets/training/all.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
X_train = train_data[:, [0, 1, 2, 3, 4, 5, 6]]
# One hot encode
X_train = (pd.get_dummies(pd.DataFrame(X_train), columns=[0,1,2,4,5]))
X_train = X_train / X_train.max(axis=0)
print(X_train.shape)
print(X_train)
y_train = train_data[:, 10]
y_train = y_train.flatten()
print(y_train)

# Create and fit the model
my_model = Pipeline([('scaler', StandardScaler()), ('my_model', SVC(C=100, degree=5, gamma=1, kernel='linear'))])
my_model.fit(X_train, y_train)  # apply scaling on training data
# print(my_model.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.

joblib.dump(my_model, "./first_stage_svm.joblib")

# Uncomment for hyperparameter tuning. 
# defining parameter range
"""
param_grid = {'my_model__C': [0.1, 1, 10, 100, 1000], 
            'my_model__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'my_model__degree': [5],
            'my_model__decision_function_shape': ['ovr', 'ovo'],
            'my_model__kernel': ['linear', 'poly']} 

grid = GridSearchCV(my_model, param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)
print("Best Score: ", grid.best_estimator_)
print("Best Params: ", grid.best_params_)
print(grid.cv_results)
"""

# Uncomment to run against a unique test set of data entries and print results to CSV for comparison.
""" data = genfromtxt("exampletest.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
my_data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
print(my_data)
my_label = data[:, [16]]
my_label = my_label.flatten()
print(my_label)

with open('svmresults.csv', 'w', ) as myfile:
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