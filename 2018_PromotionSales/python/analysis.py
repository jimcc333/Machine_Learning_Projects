""" Used to analyze the prepared weekly sales data and associated models. """

# Imports
import feature_select
import prep_data
import model

import pandas as pd

def estimate_lift(model_data, item_id, talk=True):
    """ Estimates the lift of the given item. """

    y_normalization = 10000  # Used to fix units of the y variables
    pd.options.mode.chained_assignment = None  # Stops printing a warning that is not relevant

    # Get the category
    category = 0
    for i in range(1, 7):
        if model_data.loc[((model_data['item_id'] == item_id) & (model_data['is_cat_' + str(i)] == 1))].empty:
            continue
        else:
            category = i
            break

    # Get the rows for this category (all items)
    X, y = feature_select.feature_select(prep_data.get_category(model_data, category), category)

    # Get the the promo period range
    start_week = X.loc[((X['on_promo'] == 1) & (X['item_id'] == item_id)), 'week'].min()
    end_week = X.loc[((X['on_promo'] == 1) & (X['item_id'] == item_id)), 'week'].max()

    # Get the total normalized sales during the promotion
    promotion_sales = model_data.loc[((model_data['item_id'] == item_id) & (model_data['week'] >= start_week) &
                                      (model_data['week'] <= end_week)), 'normalized_sales'].sum()

    # Estimate the sales during the same period if there was no promotion
    X_item = X.loc[((X['item_id'] == item_id) & (X['week'] >= start_week) & (X['week'] <= end_week))]
    X_item['on_promo'] = 0
    y_no_promo = model.main_model(X, y, X_item)/y_normalization

    if talk:
        print("Item", item_id)
        print("Promo period:", end_week-start_week, "weeks")
        print("Available data points were:", X_item.shape[0])
        print("Estimated lift per week: ", round(100*(promotion_sales-y_no_promo.sum()) / (end_week-start_week), 2),
              "%\n", sep='')





