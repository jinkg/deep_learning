import numpy as np
from sample_data import load_train_data, load_test_data


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = (A, Z)
    return A, cache


def sigmoid_backward(dA, cache):
    A, _ = cache
    return dA * A * (1 - A)


def relu(Z):
    A = Z * (Z >= 0)
    cache = (A, Z)
    return A, cache


def relu_forward(dA, cache):
    A, _ = cache
    return dA * (A >= 0)


def initialize_parameters(n_x, n_h, n_y):
    parameters = {}

    parameters['W1'] = np.random.randn(n_h, n_x)
    parameters['b1'] = np.zeros((n_h, 1))
    parameters['W2'] = np.random.randn(n_y, n_h)
    parameters['b2'] = np.zeros((n_y, 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    return A, (linear_cache, activation_cache)


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    elif activation == 'relu':
        dZ = relu_forward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def predict(X, parameters, Y):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    A1, _ = linear_activation_forward(X, W1, b1, activation='relu')
    A2, _ = linear_activation_forward(A1, W2, b2, activation='sigmoid')

    Y_hat = A2 >= 0.5
    accuracy = np.mean(Y_hat == Y)
    return accuracy


def two_layer_model(X, Y, parameters, num_iteration, learing_rate=0.05):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    m = X.shape[1]
    for i in range(num_iteration):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')

        accuracy = predict(X, parameters, Y)
        print("train accuracy = ", accuracy)

        # dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        # dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = np.dot(W2.T, dZ2)
        _, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')

        W1 -= learing_rate * dW1
        b1 -= learing_rate * db1
        W2 -= learing_rate * dW2
        b2 -= learing_rate * db2

    return parameters


X, Y = load_train_data()
X_test, Y_test = load_test_data()

# print(X, Y)
parameters = initialize_parameters(X.shape[0], 10, Y.shape[0])
parameters = two_layer_model(X, Y, parameters, 200, 0.005)

test_accuracy = predict(X_test, parameters, Y_test)
print("test accuracy = ", test_accuracy)
