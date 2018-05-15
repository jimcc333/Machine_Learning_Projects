""" Generalized linear models to be used for predictions. """

# Imports
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model


def bayesian_ridge(X, y, errors=False):
    """ Builds a Bayesian Ridge Regression model. """

    # Split data to training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build model
    regressor1 = linear_model.LinearRegression()
    regressor1.fit(X_train, y_train)
    y_predicted = regressor1.predict(X_test)

    percent_errors = 100 * abs(y_test - y_predicted) / y_predicted

    if errors:
        print("- Bayesian Ridge Model Performance -")
        print("Average error: ", round(percent_errors.mean(), 3), "%", sep='')
        print("Median error : ", round(np.median(percent_errors), 3), "%", sep='')
        print("Max error    : ", round(percent_errors.max(), 3), "%", sep='')

    return regressor1


def main_model(X, y, X_predict):
    """ Builds a linear regression model without splitting the data and returns predictions. """

    # Build model
    regressor1 = linear_model.LinearRegression()
    regressor1.fit(X, y)

    return regressor1.predict(X_predict)


def linear_regression(X, y, errors=False):
    """ Builds a linear regression model. """

    # Split data to training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build model
    regressor1 = linear_model.LinearRegression()
    regressor1.fit(X_train, y_train)
    y_predicted = regressor1.predict(X_test)

    percent_errors = 100 * abs(y_test - y_predicted) / y_predicted

    if errors:
        print("- Linear Regression Model Performance -")
        print("Average error: ", round(percent_errors.mean(), 3), "%", sep='')
        print("Median error : ", round(np.median(percent_errors), 3), "%", sep='')
        print("Max error    : ", round(percent_errors.max(), 3), "%", sep='')

    return regressor1


def lars_lasso(X, y, errors=False):
    """ Builds a LARS lasso model. """

    # Split data to training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build model
    regressor1 = linear_model.LassoLars()
    regressor1.fit(X_train, y_train)
    y_predicted = regressor1.predict(X_test)

    percent_errors = 100 * abs(y_test - y_predicted) / y_predicted

    if errors:
        print("- LARS Lasso Model Performance -")
        print("Average error: ", round(percent_errors.mean(), 3), "%", sep='')
        print("Median error : ", round(np.median(percent_errors), 3), "%", sep='')
        print("Max error    : ", round(percent_errors.max(), 3), "%", sep='')

    return regressor1


def lasso(X, y, errors=False):
    """ Builds a lasso model. """

    # Split data to training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build model
    regressor1 = linear_model.Lasso()
    regressor1.fit(X_train, y_train)
    y_predicted = regressor1.predict(X_test)

    percent_errors = 100 * abs(y_test - y_predicted) / y_predicted

    if errors:
        print("- Lasso Model Performance -")
        print("Average error: ", round(percent_errors.mean(), 3), "%", sep='')
        print("Median error : ", round(np.median(percent_errors), 3), "%", sep='')
        print("Max error    : ", round(percent_errors.max(), 3), "%", sep='')

    return regressor1


def ridge_regression(X, y, errors=False):
    """ Builds a ridge regression model. """

    # Split data to training/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build model
    regressor1 = linear_model.Ridge()
    regressor1.fit(X_train, y_train)
    y_predicted = regressor1.predict(X_test)

    percent_errors = 100 * abs(y_test - y_predicted) / y_predicted

    if errors:
        print("- Ridge Regression Model Performance -")
        print("Average error: ", round(percent_errors.mean(), 3), "%", sep='')
        print("Median error : ", round(np.median(percent_errors), 3), "%", sep='')
        print("Max error    : ", round(percent_errors.max(), 3), "%", sep='')

    return regressor1

