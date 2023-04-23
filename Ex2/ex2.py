import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, log_loss


def normalization(x):
    for feature_i in range(x.shape[1]):
        x_feature_i = x[:, feature_i]
        x_mean = x_feature_i.mean()
        max_min = x_feature_i.max() - x_feature_i.min()
        x[:, feature_i] = (x_feature_i - x_mean) / max_min
    return x


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def get_data_by_classes_num(classes_num, samples_num):
    return make_classification(n_classes=classes_num, n_clusters_per_class=1, class_sep=7, weights=None,
                               n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_samples=samples_num,
                               random_state=42)


def cost_function_for_Logistic_regression_regularized(y_hat, y, thetas, lamda):
    m = y.shape[0]
    loss = (-1 / m) * (y @ np.log(y_hat) + (1 - y) @ np.log(1 - y_hat))
    theta_Reg = (lamda / (2 * m)) * np.sum(np.square(thetas))
    return loss + theta_Reg


def init_theta_by_glorot(y_train, n_thetas, classes_num, train_m):
    n_inputs = n_thetas  # Number of input features
    n_outputs = classes_num  # Number of output classes
    glorot_stddev = np.sqrt(2 / (n_inputs + n_outputs)) + classes_num
    thetas = np.ones(n_inputs * n_outputs).reshape((n_inputs, n_outputs))
    thetas = thetas * (train_m / (n_outputs * np.bincount(y_train))) * glorot_stddev
    return thetas


def divide_y_by_classes(y, classes_num):
    divided_y = np.zeros((y.shape[0], classes_num))
    for class_i in range(classes_num):
        divided_y[:, class_i] = np.where(y == class_i, 1, 0)
    return divided_y


def average_cost_train_test(x_train, x_test, y_train_divided_by_class, y_test_divided_by_class, T, classes_num, lamda):
    train_avg_cost = test_avg_cost = 0
    regularized_theta = T.copy()
    regularized_theta[0, :] = 0
    for class_i in range(classes_num):
        y_hat_train = x_train @ T[:, class_i]
        sig = sigmoid(y_hat_train)
        train_avg_cost += cost_function_for_Logistic_regression_regularized(sig,
                                                                            y_train_divided_by_class[:, class_i],
                                                                            regularized_theta[:, class_i], lamda)
        y_hat_test = x_test @ T[:, class_i]
        sig = sigmoid(y_hat_test)
        test_avg_cost += cost_function_for_Logistic_regression_regularized(sig, y_test_divided_by_class[:, class_i],
                                                                           regularized_theta[:, class_i], lamda)
    return train_avg_cost, test_avg_cost


def train(x_train, y_train, x_test, y_test, classes_num, lr=1e-4, max_iter_num=1e+5, epsilon=1e-6, lamda=1):
    n_thetas = x_train.shape[1] + 1
    train_m, test_m = x_train.shape[0], x_test.shape[0]
    x_train = np.c_[np.ones(train_m), x_train]
    x_test = np.c_[np.ones(test_m), x_test]
    T = init_theta_by_glorot(y_train, n_thetas, classes_num, train_m)
    regularized_theta = T.copy()
    regularized_theta[0, :] = 0
    y_train_divided_by_class = divide_y_by_classes(y_train, classes_num)
    y_test_divided_by_class = divide_y_by_classes(y_test, classes_num)
    i = 0
    d_cost = last_cost = epsilon + 1
    train_costs_list, test_costs_list = list(), list()
    while i < max_iter_num and d_cost > epsilon:
        for class_i in range(classes_num):
            y_hat = x_train @ T[:, class_i]
            sig = sigmoid(y_hat)
            loss = sig - y_train_divided_by_class[:, class_i]
            for theta_i in range(n_thetas):
                reg_theta_i = ((lr * lamda) / train_m) * regularized_theta[theta_i, class_i]
                gradient_descent = (lr / train_m) * (loss @ x_train[:, theta_i])
                gradient_descent_regularization = gradient_descent + reg_theta_i
                T[theta_i, class_i] = T[theta_i, class_i] - gradient_descent_regularization

        train_avg_cost, test_avg_cost = average_cost_train_test(x_train, x_test, y_train_divided_by_class,
                                                                y_test_divided_by_class, T, classes_num,
                                                                lamda)
        train_avg_cost /= classes_num
        test_avg_cost /= classes_num
        train_costs_list.append(train_avg_cost)
        test_costs_list.append(test_avg_cost)
        d_cost = abs(last_cost - train_costs_list[i])
        last_cost = train_costs_list[i]
        i += 1
    return T, train_costs_list, test_costs_list, i


def predict(x_train, T):
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    y_hat = x_train @ T
    return y_hat.argmax(axis=1)


def print_model_statistics(y_test, y_hat_test, classes_num):
    y_test = pd.Series(y_test, name='Actual')
    y_hat_test = pd.Series(y_hat_test, name='Predicted')
    y_hat_test_divided_by_class = divide_y_by_classes(y_hat_test, classes_num)
    ConfusionMatrix = pd.crosstab(y_hat_test, y_test)
    print("\nConfusion Matrix:\n", ConfusionMatrix)
    scores_statistics = pd.DataFrame(
        data=[accuracy_score(y_test, y_hat_test), recall_score(y_test, y_hat_test, average='weighted'),
              precision_score(y_test, y_hat_test, average='weighted'),
              f1_score(y_test, y_hat_test, average='weighted'),
              log_loss(y_test, y_hat_test_divided_by_class, eps=1e-6)],
        index=["accuracy", "recall", "precision", "f1_score", "loss"])
    scores_statistics.columns = scores_statistics.iloc[0]
    scores_statistics = scores_statistics[1:]
    print("\nOur Model Scores:")
    print(scores_statistics)
    print("---------------------------------------------\n")


def x2(x1, class_i, T):
    try:
        return (-(x1 * T[1, class_i]) - T[0, class_i]) / T[2, class_i]
    except Exception:
        return (-(x1 * T[1]) - T[0]) / T[2]


def lines_separation(x_train, y_train, T, classes_num, model):
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    min_x1, max_x1 = np.min(x_train[:, 1]), np.max(x_train[:, 1])
    for class_i, color, ls in zip(range(classes_num), "grby", ["-.", "--", ":", "dashdot"]):
        idx = np.where(y_train == class_i)
        plt.scatter(x_train[idx, 1], x_train[idx, 2], color=color, cmap=plt.cm.Paired, edgecolor="black",
                    s=20)
        if class_i != 0:
            plt.plot([min_x1, max_x1], [x2(min_x1, class_i, T), x2(max_x1, class_i, T)], ls=ls, color=color)
    plt.title(f'{model}: Points + Separation Lines for fold {fold} with number of classes {n_classes}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    all_classes = [2, 3, 4]
    for n_classes in all_classes:
        X, Y = get_data_by_classes_num(n_classes, 2000)
        X = normalization(X)
        for fold in range(10):
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
            our_T, train_costs, test_costs, i = train(X_train, Y_train, X_test, Y_test, n_classes)
            lines_separation(X_train, Y_train, our_T, n_classes, "Our Model")
            our_model_predictions = predict(X_test, our_T)
            sklearn_model = LogisticRegression(class_weight="balanced")
            sklearn_model.fit(X_train, Y_train)
            sklearn_model_predict = sklearn_model.predict(X_test)
            sklearn_thetas = sklearn_model.coef_
            sklearn_thetas = np.c_[np.transpose(sklearn_model.intercept_), sklearn_thetas]
            sklearn_thetas = np.transpose(sklearn_thetas)
            lines_separation(X_train, Y_train, sklearn_thetas, n_classes, "Sklearn Model")
            print(f"1) Our Model Statistics for fold {fold} with number of classes {n_classes}:")
            print_model_statistics(Y_test, our_model_predictions, n_classes)
            print(f"2) ScLearn Model Statistics for fold {fold} with number of classes {n_classes}:")
            print_model_statistics(Y_test, sklearn_model_predict, n_classes)
            plt.plot(np.arange(i), train_costs, label='Training Loss')
            plt.plot(np.arange(i), test_costs, label='Testing Loss')
            plt.title(f'Training and Testing Loss for fold {fold} with number of classes {n_classes}')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.xticks(np.arange(0, i, round(i / (math.log10(i) + 1))))
            plt.legend(loc='best')
            plt.show()
