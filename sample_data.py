import numpy as np


def generate_new_test_data(size=50):
    datas = []
    for i in range(size):
        x1 = np.random.randint(-50, 50)
        x2 = np.random.randint(-50, 50)
        y = int(3 * x1 + 6 > x2)
        datas.append((x1, x2, y))
        print(x1, x2, y)
    return datas


def load_train_data():
    return load_data('sample_data_train.txt')


def load_test_data():
    return load_data('sample_data_test.txt')


def load_data(file):
    f = open(file)
    X = []
    Y = []
    line = f.readline()
    while line:
        x1, x2, y = line.split(' ')
        X.append([int(x1), int(x2)])
        Y.append([int(y)])
        line = f.readline()
    f.close()
    return np.array(X).T, np.array(Y).T

# generate_new_test_data(500)
