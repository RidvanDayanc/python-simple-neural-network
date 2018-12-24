from ANN import *
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

num_iterations = 10000
learning_rate = 0.001

data = pd.read_csv("titanic3.csv")

data["sex_cleaned"] = np.where(data["sex"] == "male", 0, 1)

cleanedData = data[["survived", "sex_cleaned",
                    "age", "fare"]].dropna(axis=0, how='any')

selectedData = cleanedData[["sex_cleaned", "age", "fare"]]
labels = cleanedData[["survived"]]

train, test,  trainLabels, testLabels = train_test_split(
    selectedData, labels, test_size=0.20)

X_train = train.values.T
X_test = test.values.T
Y_train = trainLabels.values.reshape(-1, 1).T
Y_test = testLabels.values.reshape(-1, 1)

d = model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate)
