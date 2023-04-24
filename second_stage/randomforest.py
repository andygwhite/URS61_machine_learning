# Import of the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
from numpy import genfromtxt
import joblib

def accuracyCheck(rf_model, inputdata, labeldata):
    total = 0
    count = 0
    correct = 0
    result_mat = []

    for row in inputdata:
        result = rf_model.predict(inputdata[count].reshape(1,-1))
        result_mat.append(result)
        error = abs((labeldata[count] - result) / (labeldata[count] + 1))
        if (error <= 0.15):
            correct = correct + 1

        total = total + 1
        count = count + 1
        
    with open('networkresultsdatarf.csv', 'w', ) as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for i in range(len(labeldata)):
                wr.writerow([labeldata[i], result_mat[i]])

    print("Accuracy: ", correct / total)

# Initialisation of the model
# my_model = RandomForestRegressor(n_estimators = 100, random_state = 0)

data = genfromtxt("networktestdata.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
my_data = data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
print(my_data)
my_label = data[:, [16]]
print(my_label)
my_label = my_label.flatten()
# print(my_label.shape)

# Fitting function
# my_model.fit(my_data, my_label)

# Prediction function
# data2 = genfromtxt("networkvalidationdata.csv", delimiter=',', skip_header=0, encoding="utf-8-sig", dtype='float32')
# print(data.shape)
# test_inputs = data2[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]
# test_labels = data2[:, [16]]

# Scale Data
X_train, X_test, y_train, y_test = train_test_split(my_data, my_label, random_state=42)
my_model = Pipeline([('scaler', StandardScaler()), ('my_model', RandomForestClassifier(max_depth=9, max_features=9,
                                        min_samples_split=2,
                                        n_estimators=1000))])
my_model.fit(X_train, y_train)  # apply scaling on training data
print(my_model.score(X_test, y_test))  # apply scaling on testing data, without leaking training data.

joblib.dump(my_model, "./second_stage_rf.joblib")

# Uncomment for hyperparameter tuning. 
# defining parameter range
""" param_grid = {'my_model__n_estimators': [1, 10, 100, 1000], 
            'my_model__max_depth': [1, 3, 5, 7, 9],
            'my_model__min_samples_split': [0.01, 0.1, 0.4, 0.6, 0.8, 1],
            'my_model__max_features': [1, 3, 5, 7, 9, 11]} 

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