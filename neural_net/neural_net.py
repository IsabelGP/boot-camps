import numpy as np

def linear_activation(x, gradient = False):
    if gradient:
        return (x == x)
    return x

def relu_activation(x, gradient = False):
    if gradient:
        return 1 * (x > 0)
    return x * (x > 0)

def sigmoid_activation(x, gradient = False):
    if gradient:
        return x * (1 - x)
    return 1/(1+np.exp(-x))

w_in_hidden = np.random.normal((n_features, n_hidden))
b_hidden = np.zeros(n_hidden)
w_hidden_out = np.random.normal((n_hidden, 1))
b_out = np.zeros(1)
