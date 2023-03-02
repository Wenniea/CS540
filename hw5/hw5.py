import sys
import csv
import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath):
    x = []
    y = []
    with open(filepath, mode='r') as f:
        file = csv.DictReader(f)
        for row in file:
            x.append(int(row['year']))
            y.append(int(row['days']))
    return x, y


def x_vector(x):
    X = np.array([[1, x[0]]])

    for i in range(len(x) - 1):
        year = np.array([[1, x[i + 1]]])
        X = np.concatenate((X, year))
    return X


def y_vector(y):
    Y = np.array(y)
    return Y


def z_vector(X):
    transposeX = np.transpose(X)
    Z = np.dot(transposeX, X)
    return Z


def inverse():
    return np.linalg.inv(Z)


def pseudo_inverse():
    transposeX = np.transpose(X)
    PI = np.dot(I, transposeX)
    return PI


def beta():
    B = np.dot(PI, Y)
    return B


def y_test():
    x_test = 2022
    prediction = hat_beta[0] + hat_beta[1] * x_test
    return prediction


def interpretation():
    if hat_beta[1] > 0:
        sign = '>'
        explanation = 'Number of ice days at Lake Mendota is increasing.'
    elif hat_beta[1] < 0:
        sign = '<'
        explanation = 'Number of ice days at Lake Mendota is decreasing.'
    else:
        sign = '='
        explanation = 'Number of ice days at Lake Mendota is staying the same.'
    return sign, explanation


def limitation():
    prediction = (0 - hat_beta[0]) / hat_beta[1]
    return prediction


if __name__ == "__main__":
    x, y = load_data(sys.argv[1])

    plt.plot(x, y)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")

    X = x_vector(x)
    print("Q3a:")
    print(X)

    Y = y_vector(y)
    print("Q3b:")
    print(Y)

    Z = z_vector(X)
    print("Q3c:")
    print(Z)

    I = inverse()
    print("Q3d:")
    print(I)

    PI = pseudo_inverse()
    print("Q3e:")
    print(PI)

    hat_beta = beta()
    print("Q3f:")
    print(hat_beta)

    print("Q4: " + str(y_test()))

    sign, explanation = interpretation()
    print("Q5a: " + sign)
    print("Q5b: " + explanation)

    print("Q6a: " + str(limitation()))
    print(
        "Q6b: x* is a compelling prediction based on the trend because the head-beta is a negative "
        "number, which means the number of frozen days is decreasing. And since the graph is going "
        "in a downward trend, and it's currently on around 70 days, by 2455 it's possible to have "
        "no frozen days at all.")
