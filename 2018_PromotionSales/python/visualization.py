""" Tools used to visualize the data for the 2018_BlackLocus project. """

# Imports
import prep_data
import model
import feature_select

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import spline


def explain_df(df):
    """ Prints info about the dataframe on screen. """
    print('\n-------------------------')
    print(df.head())
    print(df.describe())
    print("\nData types:\n", df.dtypes)


def make_overlay_plots(model_data):
    """ Function that calls the plot_predicted function on all promo items. """

    # Select important features using backwards feature elimination
    X_1, y_1 = feature_select.feature_select_cat_1(prep_data.get_category(model_data, 1), talk=False)
    X_2, y_2 = feature_select.feature_select_cat_2(prep_data.get_category(model_data, 2), talk=False)
    X_3, y_3 = feature_select.feature_select_cat_3(prep_data.get_category(model_data, 3), talk=False)
    X_4, y_4 = feature_select.feature_select_cat_4(prep_data.get_category(model_data, 4), talk=False)
    X_5, y_5 = feature_select.feature_select_cat_5(prep_data.get_category(model_data, 5), talk=False)
    X_6, y_6 = feature_select.feature_select_cat_6(prep_data.get_category(model_data, 6), talk=False)

    # Plot model estimates
    items = X_1.loc[X_1['on_promo'] == 1, 'item_id'].unique()
    plot_predicted(X_1, y_1, items[0], 1)

    items = X_2.loc[X_2['on_promo'] == 1, 'item_id'].unique()
    plot_predicted(X_2, y_2, items[0], 2)
    plot_predicted(X_2, y_2, items[1], 2)

    items = X_3.loc[X_3['on_promo'] == 1, 'item_id'].unique()
    plot_predicted(X_3, y_3, items[0], 3)
    plot_predicted(X_3, y_3, items[1], 3)
    plot_predicted(X_3, y_3, items[2], 3)

    items = X_4.loc[X_4['on_promo'] == 1, 'item_id'].unique()
    plot_predicted(X_4, y_4, items[0], 4)
    plot_predicted(X_4, y_4, items[1], 4)

    items = X_6.loc[X_6['on_promo'] == 1, 'item_id'].unique()
    plot_predicted(X_6, y_6, items[0], 6)

    return


def plot_predicted(X_observed, y_observed, item_id, category):
    """ Plots the predicted values vs observed values.

    Inputs:
    X_observed: pandas dataframe, all rows in a given category
    y_observed: pandas dataframe, the observations of X_observed
    item_id:    int,              the item ID
    category:   int,              the category of the item

    -> Create an overlay plot with the predicted and observed values for the model for each of the promotion products,
    be sure to highlight the promo periods with a different color.
    """

    y_normalization = 10000  # Used to fix units of the y variables

    # Create the values to predict: X_predict
    X_predict = X_observed.loc[X_observed['item_id'] == item_id]

    reference_row = X_predict.loc[X_predict['week'] == min(X_predict['week'])]

    for week in range(0, 206):
        week_row = X_predict.loc[X_predict['week'] == week]
        if week_row.empty:
            reference_row['week'] = week
            X_predict = X_predict.append(reference_row)
        else:
            reference_row = week_row

    plt.figure(item_id)

    X_observed_regular = X_observed.loc[((X_observed['on_promo'] == 0) & (X_observed['item_id'] == item_id))]
    y_observed_regular = y_observed[((X_observed['on_promo'] == 0) & (X_observed['item_id'] == item_id))]
    plt.scatter(X_observed_regular['week'], y_observed_regular/y_normalization,
                c='royalblue', zorder=8, label='Regular Sales')

    X_observed_promo = X_observed.loc[((X_observed['on_promo'] == 1) & (X_observed['item_id'] == item_id))]
    y_observed_promo = y_observed[((X_observed['on_promo'] == 1) & (X_observed['item_id'] == item_id))]
    plt.scatter(X_observed_promo['week'], y_observed_promo/y_normalization,
                c='firebrick', zorder=9, label='Promotional Sales', s=100)

    # The predicted data maintain the promo dates
    plt.scatter(X_predict['week'], model.main_model(X_observed, y_observed, X_predict)/y_normalization,
                c='g', zorder=10, label='Predicted Sales', marker='d')

    plt.xlabel('Week Number')
    plt.ylabel('Normalized Sales')
    plt.title('Sales of Item ' + str(item_id) + ' in Category ' + str(category))

    plt.legend()
    plt.show()


def plot_sales(sales):
    """ Creates plots from the raw sales data. """
    # Create a plot showing sales frequency by month
    binned_sales = prep_data.monthly_sales(sales)

    plt.figure(1)
    plt.plot(binned_sales)
    plt.title("Monthly revenue")
    # This is just labeling the x-axis better
    plt.xticks(binned_sales.index.values, [str(date)[2:7] for date in binned_sales.index.values])
    plt.xlabel("Month")
    plt.ylabel("Total sales ($)")
    [plt.axvline(str(year), color='g') for year in range(2014, 2019)]
    plt.show()


def plot_weekly_sales(weekly_sales):
    """ Makes the weekly sales plot described in the problem statement.

    -> Create a plot of the time series of the weekly product sales,
    use a different color for time periods when the products were on promotion.
    """
    promo_items = weekly_sales.loc[weekly_sales['on_promo'], 'item_id'].unique()
    print(promo_items)

    f, panel = plt.subplots(3, 3)
    plt.tight_layout()

    for i in range(len(promo_items)):
        item = promo_items[i]
        x0 = weekly_sales.loc[((weekly_sales['item_id'] == item) & ~weekly_sales['on_promo']), 'week']
        x1 = weekly_sales.loc[((weekly_sales['item_id'] == item) & weekly_sales['on_promo']), 'week']

        y0 = weekly_sales.loc[((weekly_sales['item_id'] == item) & ~weekly_sales['on_promo']), 'sales']
        y1 = weekly_sales.loc[((weekly_sales['item_id'] == item) & weekly_sales['on_promo']), 'sales']

        # Smooth the lines (if enough data exists)
        # if len(x0) > 4:
        #     x0_orig = x0
        #     x0 = np.linspace(min(x0), max(x0), 200)
        #     y0 = spline(x0_orig, y0, x0)
        # if len(x1) > 4:
        #     x1_orig = x1
        #     x1 = np.linspace(min(x1), max(x1), 200)
        #     y1 = spline(x1_orig, y1, x1)
        #
        # panel[int(np.floor(i / 3)), i % 3].plot(x0, y0, 'mediumblue')
        # panel[int(np.floor(i / 3)), i % 3].plot(x1, y1, 'blueviolet')

        panel[int(np.floor(i / 3)), i % 3].set_title('Weekly Sales of Item ' + str(item))
        panel[int(np.floor(i / 3)), i % 3].set_xlabel('Week Number')
        panel[int(np.floor(i / 3)), i % 3].set_ylabel('Weekly Total Sales, $')

        panel[int(np.floor(i / 3)), i % 3].axhline(y0.mean(), c='dodgerblue', alpha=0.9, zorder=5)
        # panel[int(np.floor(i / 3)), i % 3].text()
        panel[int(np.floor(i / 3)), i % 3].scatter(x0, y0, c='royalblue', zorder=9, label='Regular Sales')

        panel[int(np.floor(i / 3)), i % 3].axhline(y1.mean(), c='salmon', alpha=0.9, zorder=5)
        panel[int(np.floor(i / 3)), i % 3].scatter(x1, y1, c='firebrick', zorder=10, label='Promotional Sales')

    panel[2, 1].text(29, 244, "Average Sales (no promo)",
                     fontsize=9, fontdict=dict(color='dodgerblue', style='italic'))

    panel[0, 0].text(10, 330, "Average Promotional Sales",
                     fontsize=9, fontdict=dict(color='salmon', style='italic'))
    panel[2, 0].text(6, 336, "Average Promotional Sales",
                     fontsize=9, fontdict=dict(color='salmon', style='italic'))

    panel[1, 1].legend()    # Showing the legend at the center subplot should be enough

    plt.show()


def state_input_facts(sales, competitor_prices):
    """ Prints info about the sales and competitor data on screen. """

    print("Total number of unique items in store:", sales['item_id'].nunique())
    promotion_items = sales.dropna(subset=['promo_start_date'])['item_id'].unique()
    print("Items that were on promotion:", len(promotion_items))
    print(" The items are:", promotion_items)
    # Should get: [514002190 513103198 514002189 512319171 512317770 515375113
    # 514002186 512319983 512319146 512464637 512018493 512317789]

    print("\nTotal number of unique items of competitors:", competitor_prices['item_id'].nunique())
    competitor_items = competitor_prices.loc[competitor_prices['item_id'].isin(promotion_items)]['item_id'].unique()
    print("Number of matching promotional items in competitor data:", len(competitor_items))
    print(" The items are:", competitor_items)
    # print("Competitor data is missing item(s):", np.subtract(promotion_items, competitor_items))
    print("Competitor data is missing item(s):", list(set(promotion_items).symmetric_difference(competitor_items)))
    # Item 513103198 does not have competitor price data

