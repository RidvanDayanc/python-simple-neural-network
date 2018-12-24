# coding=utf-8
# Package imports
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model


def sigmoid_func(z):
    s = 1 / (1 + np.exp(-z))

    return s

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)

def paramaters(n_x, n_h, n_y):

    np.random.seed(2)

    W1 = 0.01 * np.random.random((n_h, n_x))
    b1 = np.zeros((n_h, 1))
    W2 = 0.01 * np.random.random((n_y, n_h))
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid_func(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):


    m = Y.shape[1]

    logprobs = (np.multiply(Y, np.log(
        A2)) + np.multiply((np.ones(Y.shape) - Y), np.log(np.ones(A2.shape) - A2))) / m
    cost = - np.sum(logprobs)
    cost = np.squeeze(cost)

    assert (isinstance(cost, float))

    return cost


def backward(parameters, cache, X, Y):

    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.power(A1, 2)))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def updateparam(parameters, grads, learning_rate=1.2):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):

    parameters = paramaters(*layer_sizes(X_train, Y_train))
    for x in range(num_iterations):
        A2, cache = forward(X_train, parameters)
        cost = compute_cost(A2, Y_train, parameters)
        grads = backward(parameters, cache, X_train, Y_train)
        parameters = updateparam(
            parameters, grads, learning_rate=learning_rate)
    Y_prediction_train = (A2 >= 0.5) * 1.0
    A2, cache = forward(X_test, parameters)
    Y_prediction_test = (A2 >= 0.5) * 1.0
    print("train accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(
        100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": cost,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
