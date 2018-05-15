""" Used to select features of the model dataset for each product category. """

# Imports
import model

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np


def feature_fit(X, y):
    """ Fits the given data to the SelectKBest selector. """

    selector = SelectKBest(score_func=chi2)
    return selector.fit(X, y)


def feature_select(sales, category):
    """ Calls the correct feature_select_cat_* function given a category. """
    if category == 1:
        return feature_select_cat_1(sales)
    elif category == 2:
        return feature_select_cat_2(sales)
    elif category == 3:
        return feature_select_cat_3(sales)
    elif category == 4:
        return feature_select_cat_4(sales)
    elif category == 5:
        return feature_select_cat_5(sales)
    elif category == 6:
        return feature_select_cat_6(sales)
    else:
        print("feature_select function can't find category", category)
        return None, None


def feature_select_cat_1(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # Assign data
    X = sales[['on_promo', 'week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_41', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 20.139
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 65.126
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)
    '''
    --- Fitting Results ---
    -on_promo, score: 1675.0  0.0
    -week, score: 56298.986  0.0
    -week_52, score: 6708.261  0.0
    -item_id, score: 9479127415.533  0.0
    -is_g_11, score: 1479.64  0.0
    -is_g_12, score: 1559.848  0.0
    -is_g_13, score: 1581.356  0.0
    -is_g_21, score: 1512.588  0.0
    -is_g_22, score: 1580.466  0.0
    -is_g_23, score: 1554.566  0.0
    -is_g_24, score: 1554.928  0.0
    -is_g_31, score: 1530.252  0.0
    -is_g_32, score: 1505.486  0.0
    -is_g_33, score: 1564.521  0.0
    -is_g_34, score: 1606.281  0.0
    -is_g_41, score: 1677.0  0.0
    -is_g_42, score: 1490.318  0.0
    -is_g_43, score: 1481.679  0.0
    '''

    return X_new, y


def feature_select_cat_2(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # Assign data
    X = sales[['on_promo', 'week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_41', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']
    y = np.array(y).astype(int)

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 17.646
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 74.307
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    --- Fitting Results ---
    -on_promo, score: 1344.462  0.0
    -week, score: 65923.755  0.0
    -week_52, score: 6869.685  0.0
    -item_id, score: 10953665927.759  0.0
    -is_g_11, score: 1507.119  0.0
    -is_g_12, score: 1534.76  0.0
    -is_g_13, score: 1521.648  0.0
    -is_g_14, score: 1514.833  0.0
    -is_g_21, score: 1471.173  0.0
    -is_g_22, score: 1528.456  0.0
    -is_g_23, score: 1519.641  0.0
    -is_g_24, score: 1583.0  0.0
    -is_g_31, score: 1520.478  0.0
    -is_g_32, score: 1458.541  0.0
    -is_g_33, score: 1465.209  0.0
    -is_g_34, score: 1559.378  0.0
    -is_g_41, score: 1654.0  0.0
    -is_g_42, score: 1545.813  0.0
    -is_g_43, score: 1510.472  0.0
    '''

    return X_new, y


def feature_select_cat_3(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # Remove one negative row
    sales = sales.loc[sales['normalized_sales'] >= 0]

    # Assign data
    X = sales[['on_promo', 'week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_41', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']
    y = np.array(y).astype(int)

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 21.716
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 135.28
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    --- Fitting Results ---
    -on_promo, score: 1213.0  0.0
    -week, score: 53390.06  0.0
    -week_52, score: 6636.99  0.0
    -item_id, score: 7225666943.334  0.0
    -is_g_11, score: 1336.83  0.0
    -is_g_12, score: 1484.483  0.0
    -is_g_13, score: 1449.68  0.0
    -is_g_14, score: 1490.0  0.0
    -is_g_21, score: 1449.813  0.0
    -is_g_22, score: 1464.254  0.0
    -is_g_23, score: 1441.289  0.0
    -is_g_24, score: 1480.565  0.0
    -is_g_31, score: 1481.252  0.0
    -is_g_32, score: 1463.0  0.0
    -is_g_33, score: 1480.458  0.0
    -is_g_34, score: 1473.968  0.0
    -is_g_41, score: 1582.0  0.0
    -is_g_42, score: 1440.54  0.0
    -is_g_43, score: 1373.93  0.0
    '''

    return X_new, y


def feature_select_cat_4(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # Assign data
    X = sales[['on_promo', 'week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_41', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']
    y = np.array(y).astype(int)

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 11.93
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 60.758
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    --- Fitting Results ---
    -on_promo, score: 1510.24  0.0
    -week, score: 77972.183  0.0
    -week_52, score: 8783.779  0.0
    -item_id, score: 12367050738.41  0.0
    -is_g_11, score: 1916.438  0.0
    -is_g_12, score: 2042.521  0.0
    -is_g_13, score: 2054.591  0.0
    -is_g_14, score: 1939.218  0.0
    -is_g_21, score: 1860.753  0.0
    -is_g_22, score: 1861.604  0.0
    -is_g_23, score: 1938.343  0.0
    -is_g_24, score: 2065.056  0.0
    -is_g_31, score: 1966.725  0.0
    -is_g_32, score: 2000.407  0.0
    -is_g_33, score: 2012.369  0.0
    -is_g_34, score: 2060.03  0.0
    -is_g_41, score: 2185.0  0.0
    -is_g_42, score: 1819.773  0.0
    -is_g_43, score: 1886.093  0.0
    '''

    return X_new, y


def feature_select_cat_5(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # !
    # Category 5 has no promotional sales and does not include the on_promo column
    # !

    # Assign data
    X = sales[['week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_41', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']
    y = np.array(y).astype(int)

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 11.763
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 41.006
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    --- Fitting Results ---
    -week, score: 51747.576  0.0
    -week_52, score: 6040.551  0.0
    -item_id, score: 8630139482.853  0.0
    -is_g_11, score: 1439.252  0.0
    -is_g_12, score: 1428.224  0.0
    -is_g_13, score: 1478.693  0.0
    -is_g_14, score: 1456.582  0.0
    -is_g_21, score: 1417.292  0.0
    -is_g_22, score: 1451.449  0.0
    -is_g_23, score: 1486.87  0.0
    -is_g_24, score: 1527.348  0.0
    -is_g_31, score: 1457.755  0.0
    -is_g_32, score: 1489.264  0.0
    -is_g_33, score: 1490.495  0.0
    -is_g_34, score: 1446.092  0.0
    -is_g_41, score: 1601.0  0.0
    -is_g_42, score: 1362.492  0.0
    -is_g_43, score: 1421.182  0.0
    '''

    return X_new, y


def feature_select_cat_6(sales, talk=False):
    """ Used like a script to select the important features from the given sales dataframe. """

    # ! Ignores is_g_41 from the beginning (none are in this category)

    # Assign data
    X = sales[['on_promo', 'week', 'week_52', 'item_id', 'price_index', 'competitor_index', 'is_g_11', 'is_g_12',
               'is_g_13', 'is_g_14', 'is_g_21', 'is_g_22', 'is_g_23', 'is_g_24', 'is_g_31', 'is_g_32', 'is_g_33',
               'is_g_34', 'is_g_42', 'is_g_43']]

    y = sales['normalized_sales_int']
    y = np.array(y).astype(int)

    # Feature extraction
    fit = feature_fit(X, y)
    if talk:
        try_models(X, y, talk=talk)
        show_feature_scores(X, fit)

    '''
    price_index, score: 22.248
    '''
    X_new = X.drop(labels=['price_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    competitor_index, score: 88.757
    '''
    X_new = X_new.drop(labels=['competitor_index'], axis=1)

    fit = feature_fit(X_new, y)
    if talk:
        try_models(X_new, y, talk=talk)
        show_feature_scores(X_new, fit)

    '''
    --- Fitting Results ---
    -on_promo, score: 1693.0  0.0
    -week, score: 54339.811  0.0
    -week_52, score: 6902.42  0.0
    -item_id, score: 10009138890.633  0.0
    -is_g_11, score: 1403.989  0.0
    -is_g_12, score: 1566.112  0.0
    -is_g_13, score: 1577.841  0.0
    -is_g_14, score: 1660.0  0.0
    -is_g_21, score: 1567.0  0.0
    -is_g_22, score: 1523.323  0.0
    -is_g_23, score: 1577.613  0.0
    -is_g_24, score: 1596.281  0.0
    -is_g_31, score: 1545.824  0.0
    -is_g_32, score: 1459.019  0.0
    -is_g_33, score: 1545.027  0.0
    -is_g_34, score: 1569.991  0.0
    -is_g_42, score: 1634.0  0.0
    -is_g_43, score: 1511.965  0.0
    '''


    return X_new, y


def show_feature_scores(X, fit):
    """ Shows the fitting scores of the columns. """

    print("--- Fitting Results ---")
    np.set_printoptions(precision=3)

    scores = fit.scores_
    p_values = fit.pvalues_
    for i in range(len(scores)):
        print('-', X.columns.values[i], ', score: ', round(scores[i], 3), '  ', round(p_values[i], 3), sep='')


def try_models(X, y, talk=True):
    """ Runs all the available models on the given data and prints errors. """

    model.bayesian_ridge(X, y, errors=talk)
    model.linear_regression(X, y, errors=talk)
    model.lars_lasso(X, y, errors=talk)
    model.lasso(X, y, errors=talk)
    model.ridge_regression(X, y, errors=talk)
