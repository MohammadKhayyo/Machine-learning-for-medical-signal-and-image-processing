import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def loadData(filename, sheet_name):
    """
    Loads Excel file into arrays
    :param filename:  Excel file to load
    :param sheet_name: sheet name to load
    :return: arrays of X and Y
    """
    x = list()
    y = list()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=[1, 2])
        for row in range(df.shape[0] - 1):
            x.append(df.iat[row + 1, 0])
            y.append(df.iat[row + 1, 1])
    except Exception as e:
        print(f'error : {e}')
        exit(1)
    return np.array(x), np.array(y)


def train(x, y, alpha=1e-4, min_d_cost=1e-6, max_iter_num=1e+6):
    """
    Trains our model with the given data X and Y, in addition to alpha, and max iteration number
    :param x: dataset
    :param y: label set
    :param alpha: learning rate
    :param min_d_cost: minimum diff cost
    :param max_iter_num: maximum number of iteration
    :return: weights, cost, number of iterations passed
    """
    m = x.shape[0]
    x = np.c_[np.ones(m), x]
    T = np.random.rand(x.shape[1])
    i = 0
    d_cost = min_d_cost + 1
    last_cost = min_d_cost + 1
    cost = list()
    while i < max_iter_num and d_cost > min_d_cost:
        h = x @ T
        loss = h - y
        S = loss.T @ x
        T = T - (alpha / m) * S
        cost.append(np.sum(np.square(loss)) / (2 * m))
        d_cost = np.abs(last_cost - cost[i])
        last_cost = cost[i]
        i += 1
    return T, cost, i


def predict(data_for_train, labels, weights):
    """
    takes the input and gives return an estimate based on our training set
    :param data_for_train: training dataset
    :param labels: labels for dataset
    :param weights: theta0 , theta1
    :return: predicted answer, MSE
    """
    data_for_train = np.c_[np.ones(data_for_train.shape[0]), data_for_train]
    m = data_for_train.shape[0]
    h = data_for_train @ weights
    final_cost = np.sum(np.square(np.array(h) - labels)) / (2 * m)
    return h, final_cost


def scipy_linear(data_for_train, labels):
    """
    Calculates linear regression using prebuilt function in scipy
    :param data_for_train: training dataset
    :param labels: labels for dataset
    :return: hypothesis, slope, bias, cost
    """
    slope, intercept, r, p, stderr = stats.linregress(data_for_train, labels)

    def Hypothesis(x):
        return slope * x + intercept

    m = data_for_train.shape[0]
    h = list(map(Hypothesis, data_for_train))
    final_cost = np.sum(np.square(np.array(h) - labels)) / (2 * m)
    return h, [intercept, slope], final_cost


if __name__ == '__main__':
    _x, _y = loadData('בדיקות_מעבדה.xlsx', 'בדיקות')
    _T, _costs, number_of_iter = train(_x, _y)
    Our_model, final_cost_Our_model = predict(_x, _y, _T)
    scipy_model, _T_scipy_model, final_cost_scipy_model = scipy_linear(_x, _y)
    plt.subplot(121)
    plt.scatter(_x, _y)
    plt.plot(_x, Our_model, color="red")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Our Model")
    plt.subplot(122)
    plt.scatter(_x, _y)
    plt.plot(_x, scipy_model, color="orange")
    plt.xlabel('X')
    plt.title("Scipy Model")
    plt.show()
    plt.plot([i for i in range(number_of_iter)], _costs, color="black")
    plt.xlabel('Number of iteration')
    plt.ylabel('Cost Function')
    plt.title("Our Model")
    plt.show()
    print(
        f'Scipy Model is the best with MSE value: {final_cost_scipy_model}'
        f' While Our Model has MSE value of: {final_cost_Our_model}') \
        if final_cost_scipy_model < final_cost_Our_model else print(
        f'Our Model is the best with cost value: {final_cost_Our_model} '
        f'While Scipy Model has MSE value of: {final_cost_scipy_model}')
    print("Our Model Thetas: ", _T)
    print("Scipy Model Thetas: ", _T_scipy_model)
