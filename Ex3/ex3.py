# Load libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split, GridSearchCV  # Import train_test_split function
from sklearn import linear_model  # Import scikit-learn metrics module for accuracy calculation


def loadData(filename, sheet_name, list_of_column):
    """
    Loads Excel file into arrays
    :param list_of_column: the column's that we import
    :param filename:  Excel file to load
    :param sheet_name: sheet name to load
    :return: arrays of X and Y
    """
    x, y = list(), list()
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name, usecols=list_of_column)
        df = df.drop(index=0, axis=0)
        x, y = list(df[list_of_column[0]]), list(df[list_of_column[1]])
    except Exception as e:
        print(f'error : {e}')
        exit(1)
    return np.array(x), np.array(y)


################################################################LoF
def filter_data(X, Y, contamination=0.03):
    """It clears points classified as noise from the data"""
    if (contamination == 0):
        return X, Y, [], []
    # identify outliers in the training dataset
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=25)
    yhat = lof.fit_predict(X)
    # select all rows that are not outliers
    clean_mask = yhat != -1
    dirty_mask = yhat != 1
    X_clean, Y_clean = X[clean_mask, :], Y[clean_mask]
    X_dirty, Y_dirty = X[dirty_mask, :], Y[dirty_mask]
    return X_clean, Y_clean, X_dirty, Y_dirty


######################################################################

def activate_models_on_dataset(X, Y, X_dirty, Y_dirty, contamination):
    """Examine the different models to see which one gives better results"""
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=1)  # 80% training and 20% test
    # Fitting Polynomial Regression to the dataset
    lin1 = linear_model.LinearRegression()
    lin1.fit(X_train, y_train)

    poly_score_train = lin1.score(X_train, y_train)
    poly_score_test = lin1.score(X_test, y_test)

    #########################################
    rfc = RandomForestRegressor(n_estimators=700, max_features='auto', random_state=5)
    rfc.fit(X_train, y_train)
    rfc_score_train = rfc.score(X_train, y_train)
    rfc_score_test = rfc.score(X_test, y_test)
    #################################################################
    tree_regressor = DecisionTreeRegressor(max_depth=4, random_state=0)
    tree_regressor.fit(X_train, y_train)
    tree_regressor_score_train = tree_regressor.score(X_train, y_train)
    tree_regressor_score_test = tree_regressor.score(X_test, y_test)

    #################################################################
    xg_reg = xgb.XGBRegressor(max_depth=4)
    xg_reg.fit(X_train, y_train)

    xgb_score_train = xg_reg.score(X_train, y_train)
    xgb_score_test = xg_reg.score(X_test, y_test)
    ################################################################

    #################################################################
    poly_yhat_test = lin1.predict(X_test)
    rfc_yhat_test = rfc.predict(X_test)
    tree_yhat_test = tree_regressor.predict(X_test)
    xg_yhat_test = xg_reg.predict(X_test)

    poly_yhat_train = lin1.predict(X_train)
    rfc_yhat_train = rfc.predict(X_train)
    tree_yhat_train = tree_regressor.predict(X_train)
    xg_yhat_train = xg_reg.predict(X_train)

    def plot_model(X_plot, Y_plot, model, color='green', model_name=''):
        X_grid = np.arange(min(X_plot), max(X_plot), 0.01)
        X_grid = X_grid.reshape((len(X_grid), 1))
        plt.scatter(X_plot, Y_plot, linewidth=2)
        plt.scatter(X_dirty, Y_dirty, color='red', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{model_name} with contamination={contamination}')
        plt.plot(X_grid, model.predict(X_grid), color)
        plt.show()

    plot_model(X, Y, lin1, color='black', model_name='linear regression')
    plot_model(X, Y, tree_regressor, color='green', model_name='regression tree')
    plot_model(X, Y, rfc, color='orange', model_name='random forest')
    plot_model(X, Y, xg_reg, color='purple', model_name='XGBoost')

    def print_scores():
        print('Poly train score: ' + str(poly_score_train) + " | Poly test score: " + str(poly_score_test))
        print("rfc train score: " + str(rfc_score_train) + " | rfc test score: " + str(rfc_score_test))
        print("tree_reg train score: " + str(tree_regressor_score_train) + " | tree_reg test score: " + str(
            tree_regressor_score_test))
        print("xgb train score: " + str(xgb_score_train) + " | xgb test score: " + str(xgb_score_test))

    def print_MSE_train_results():
        print('*********************************************************************************')
        print('MSE train results:')
        print('Polynomial MSE: ' + str(mean_squared_error(poly_yhat_train, y_train)))
        print('Random Forest MSE: ' + str(mean_squared_error(rfc_yhat_train, y_train)))
        print('Regression Tree MSE: ' + str(mean_squared_error(tree_yhat_train, y_train)))
        print('XG MSE: ' + str(mean_squared_error(xg_yhat_train, y_train)))
        print('*********************************************************************************')

    def print_MSE_test_results():
        print('*********************************************************************************')
        print('MSE test results:')

        print('Polynomial MSE: ' + str(mean_squared_error(poly_yhat_test, y_test)))
        print('Random Forest MSE: ' + str(mean_squared_error(rfc_yhat_test, y_test)))
        print('Regression Tree MSE: ' + str(mean_squared_error(tree_yhat_test, y_test)))
        print('XG MSE: ' + str(mean_squared_error(xg_yhat_test, y_test)))
        print('*********************************************************************************')

    def print_MAE_train_results():
        print('*********************************************************************************')
        print('MAE train results:')

        print('Polynomial MAE: ' + str(mean_absolute_error(poly_yhat_train, y_train)))
        print('Random Forest MAE: ' + str(mean_absolute_error(rfc_yhat_train, y_train)))
        print('Regression Tree MAE: ' + str(mean_absolute_error(tree_yhat_train, y_train)))
        print('XG MAE: ' + str(mean_absolute_error(xg_yhat_train, y_train)))
        print('*********************************************************************************')

    def print_MAE_test_results():
        print('*********************************************************************************')
        print('MAE test results:')

        print('Polynomial MAE: ' + str(mean_absolute_error(poly_yhat_test, y_test)))
        print('Random Forest MAE: ' + str(mean_absolute_error(rfc_yhat_test, y_test)))
        print('Regression Tree MAE: ' + str(mean_absolute_error(tree_yhat_test, y_test)))
        print('XG MAE: ' + str(mean_absolute_error(xg_yhat_test, y_test)))
        print('*********************************************************************************')

    print_scores()
    print_MSE_train_results()
    print_MSE_test_results()
    print_MAE_train_results()
    print_MAE_test_results()


def func1(X, Y, contamination=0.03):
    """Performing the required tasks according to the contamination number"""
    print('with contamination value: ' + str(contamination))
    X, Y, X_dirty, Y_dirty = filter_data(X, Y, contamination)
    activate_models_on_dataset(X, Y, X_dirty, Y_dirty, contamination)
    plt.show()


if __name__ == '__main__':
    X, Y = loadData(filename='בדיקות_מעבדה.xlsx', sheet_name='בדיקות', list_of_column=['בדיקה', 'סיכוי לחלות במחלה '])
    X = X.reshape(-1, 1)
    func1(X, Y, 0)
    func1(X, Y, 0.01)
    func1(X, Y, 0.03)
    func1(X, Y, 0.05)
