""" Tools to prepare the data for the 2018_BlackLocus project. """

# Imports
import visualization
import time
import hashlib

import pandas as pd
import numpy as np
from datetime import timedelta


def add_competitor_index(weekly_sales):
    """ Populates the competitor_index column in the weekly_sales dataframe. """

    weekly_sales['competitor_index'] = weekly_sales['price'] / weekly_sales['competitor_min']

    return weekly_sales


def add_price_index(weekly_sales):
    """ Populates the price_index column in the weekly_sales dataframe. """

    for item_id in weekly_sales['item_id'].unique():
        # Find the mean price of this item
        mean_price = round(weekly_sales.loc[weekly_sales['item_id'] == item_id, 'price'].mean(), 4)

        # Populate the price_index column
        if mean_price == 0:
            continue
        weekly_sales.loc[weekly_sales['item_id'] == item_id, 'price_index'] = \
            weekly_sales.loc[weekly_sales['item_id'] == item_id, 'price'] / mean_price

    return weekly_sales


def add_normalized_sales(weekly_sales):
    """ Populates the normalized_sales column in the weekly_sales dataframe. """

    for item_id in weekly_sales['item_id'].unique():
        # Find the weekly mean sale of this item
        mean_sales = round(weekly_sales.loc[weekly_sales['item_id'] == item_id, 'sales'].mean(), 4)

        # Assign to the normalized_sales column
        weekly_sales.loc[weekly_sales['item_id'] == item_id, 'normalized_sales'] = \
            weekly_sales.loc[weekly_sales['item_id'] == item_id, 'sales'] / mean_sales

    return weekly_sales


def add_promo(weekly_sales, sales):
    """ Populates the on_promo column in the weekly_sales dataframe. """

    sales_needed = sales.dropna(subset=['promo_start_date'])

    for item_id in weekly_sales['item_id'].unique():
        # Find the promo date range
        item_dates = sales_needed.loc[sales_needed['item_id'] == item_id, ['promo_start_date', 'promo_end_date']]

        if item_dates.empty:
            # No promo
            weekly_sales['on_promo'].loc[weekly_sales['item_id'] == item_id] = False

        else:
            # Set it all to False then switch ones on promotion
            weekly_sales['on_promo'].loc[weekly_sales['item_id'] == item_id] = False

            # There is a promo
            start_date = item_dates['promo_start_date'].unique()
            end_date = item_dates['promo_end_date'].unique()

            # print(item_id)
            # print(start_date)
            # print(end_date)

            # This is highly specific to the dataset
            if len(start_date) == 1 and len(end_date) == 1:
                weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                              (weekly_sales['week_begin'] > start_date[0]) &
                                              (weekly_sales['week_begin'] < end_date[0]))] = True
            elif len(start_date) == 2 and len(end_date) == 2:
                # There may be an error in this part, the if statement is a quick fix
                if item_id == 513103198:
                    weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                                  (weekly_sales['week_begin'] == '2014-11-23'))] = True

                    weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                                  (weekly_sales['week_begin'] == '2017-01-01'))] = True

                weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                              (weekly_sales['week_begin'] > start_date[0]) &
                                              (weekly_sales['week_begin'] < end_date[0]))] = True
                weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                              (weekly_sales['week_begin'] > start_date[1]) &
                                              (weekly_sales['week_begin'] < end_date[1]))] = True
            elif len(start_date) == 2 and len(end_date) == 1:
                weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                              (weekly_sales['week_begin'] > start_date[0]) &
                                              (weekly_sales['week_begin'] < end_date[0]))] = True
                weekly_sales['on_promo'].loc[((weekly_sales['item_id'] == item_id) &
                                              (weekly_sales['week_begin'] > start_date[1]))] = True
            else:
                print("\nHey there, looks like you're using a new dataset. Checkout prep_data.py and catch conditions.")
                print("Start date:\n", start_date)
                print("\nEnd date:\n", end_date)
                print("item_dates:\n", item_dates)
    return weekly_sales


def add_similarity(weekly_sales):
    """ Populates the similarity_group column based on quartiles in 2-dimensions for each product category. """

    for category in weekly_sales['category_id'].unique():
        # Get data for this category
        category_df = weekly_sales.loc[weekly_sales['category_id'] == category, ['price', 'sales']]

        # Create quartile column for each dimension
        category_df = category_df.assign(price_quartiles=pd.qcut(category_df['price'], 4,
                                                                 labels=['price1', 'price2', 'price3', 'price4']),
                                         sales_quartiles=pd.qcut(category_df['sales'], 4,
                                                                 labels=['sales1', 'sales2', 'sales3', 'sales4']))

        # Assign the cartesian product to the similarity_group column
        category_df = category_df.assign(similarity_group=category_df.filter(regex='_quartiles').apply(tuple, axis=1))

        # Drop the categorical (*_quartiles) columns
        category_df = category_df.drop(['price_quartiles', 'sales_quartiles'], axis=1)

        # Update the returned table
        weekly_sales.loc[weekly_sales['category_id'] == category, ['similarity_group']] = \
            category_df.iloc[:, category_df.columns.get_loc('similarity_group')]

    return weekly_sales


def aggregate_sales(sales, competitor):
    """ Aggregates the sales data to a weekly level and return it as weekly_sales. """

    # Initialize empty dataframe
    total_rows = 10500
    weekly_sales = pd.DataFrame(index=np.arange(0, total_rows), columns=[
        'week_begin',           # datetime of the beginning of week
        'week',                 # int, counts up from earliest week til end of data
        'month',                # int, see problem statement
        'item_id',
        'category_id',
        'price',                # float, the average weekly price of the item
        'competitor_min',       # float, the minimum price (closest available)
        'competitor_delta',     # int, the number of weeks between the price and the source of the competitor_min
        'on_promo',             # bool, is this item on promo
        'sales',                # float, total weekly sales of this product
        'normalized_sales',     # float, see problem statement
        'price_index',          # float, see problem statement
        'competitor_index',     # float, see problem statement
        'similarity_group'])    # int, see problem statement

    sales = sales.sort_values(['item_id', 'history_date'])

    current_index = 0
    all_items = sales['item_id'].unique()
    weeks = pd.date_range(min(sales['history_date']) - pd.Timedelta(' 8 days'),
                          max(sales['history_date']) + pd.Timedelta(' 8 days'), freq='W')

    for item_id in all_items:
        for week_i in range(len(weeks)-1):
            item_weekly_sales = sales.loc[((sales['item_id'] == item_id) & (sales['history_date'] > weeks[week_i]) &
                                           (sales['history_date'] <= weeks[week_i+1]))]

            if item_weekly_sales.empty:
                continue
            row = item_weekly_sales.iloc[0]
            # Add this weeks row for this item
            competitor_min, competitor_delta = find_competitor_price(item_id, weeks[week_i], competitor)
            if np.isnan(competitor_min):
                competitor_min = item_weekly_sales['price'].mean()
            if competitor_min == 0:
                competitor_min = np.nan
            weekly_sales.iloc[current_index] = [weeks[week_i],
                                                week_i,
                                                np.floor(week_i/4.33),  # This finds month (it's 11PM.. it won't matter)
                                                item_id,
                                                row['category_id'],
                                                item_weekly_sales['price'].mean(),
                                                competitor_min,     # competitor_min
                                                competitor_delta,     # competitor_delta
                                                None,       # on_promo
                                                item_weekly_sales['sales'].sum(),
                                                np.nan,     # normalized_sales
                                                np.nan,     # price_index
                                                np.nan,     # competitor_index
                                                None]       # similarity_group
            current_index = current_index + 1

    return weekly_sales


def drop_sales_missing(sales):
    """ Drops rows that do not have a sale (does not drop if zero). """

    return sales[np.isfinite(sales['sales'])]


def drop_useless(sales):
    """ Drops the columns in sales data that are not needed in this project. """

    # Drop inventory
    sales = sales.drop(columns=['inventory'])

    return sales


def find_competitor_price(item_id, history_date, competitor):
    """ Finds the minimum competitor price and the time delta for a given item and date. """

    unknown_price_delta = timedelta(weeks=300)  # This is used when there's no competitor price available
    original_date = history_date
    competitor_delta = unknown_price_delta

    # Get relevant rows from competitor table and put it in item_info
    item_info = competitor.loc[((competitor['item_id'] == item_id) & (competitor['history_date'] <= history_date))]

    # If no past info on it
    if item_info.empty:
        # Look for all info (so, future info)
        item_info = competitor.loc[(competitor['item_id'] == item_id)]
        if item_info.empty:
            # If there's no info on this item at all return nan
            return [np.nan, competitor_delta]
        # If you're here you won't get to the while loop below because lowest_price won't be np.nan
        item_info = item_info.sort_values('history_date')
        history_date = item_info['history_date'].iloc[0]
        closest_day = item_info.loc[item_info['history_date'] == history_date]

        competitor_min = closest_day['competitor_price'].min()
        competitor_delta = history_date - original_date

    else:
        competitor_min = np.nan  # This is the condition for the while loop below

    # Find the closest day and get the lowest price for it
    iterations = 0
    while np.isnan(competitor_min):
        # Find prices for the nearest date with prices
        closest_day = item_info.loc[item_info['history_date'] == history_date]
        if closest_day.empty:
            history_date = history_date - timedelta(days=1)
            continue

        # Determine lowest price
        competitor_min = closest_day['competitor_price'].min()
        competitor_delta = original_date - history_date

        # Don't let things get out of hand during this iteration
        iterations = iterations + 1
        if iterations > 50:
            break

    competitor_min = competitor_min / 100 # Dividing by 100 to "match units"

    return competitor_min, competitor_delta


def get_category(model_data, category):
    """ Returns only the rows with the given category and removes redundant rows from the prepared model data. """

    # Make sure we're working with strings
    if isinstance(category, int):
        category = str(category)

    # Get all rows with the category
    category_data = model_data.loc[model_data['is_cat_' + category] == 1]

    # Remove all is_cat_* rows since they carry no information now
    category_data = category_data.drop(labels=['is_cat_1', 'is_cat_2', 'is_cat_3', 'is_cat_4', 'is_cat_5', 'is_cat_6'],
                                       axis=1)

    return category_data


def impute_sales(sales, talk=False):
    """ Handles the missing data in the sales dataset. """

    if talk:
        print("Missing data:\n", sales.isnull().sum())
        """ Should see (except zero columns):
        Missing data:
            price                    3
            inventory                2
            sales               402560
            promo_start_date    416794
            promo_end_date      416794
        """

    # Default missing sales values to zero (see problem_description)
    sales['sales'] = sales['sales'].fillna(0)

    # Take care of item 512317789 with missing price/inventory info
    if talk:
        print('Entries of the item with dates near the entries with the missing values:')
        print(sales.loc[(sales['history_date'] > '2016-12-22') & (sales['history_date'] < '2017-01-22') &
                        (sales['item_id'] == 512317789), ['history_date', 'price', 'inventory', 'sales']]
              .sort_values('history_date'))

    # Since the price is 23.34 for before the missing dates, this price is assumed
    sales.loc[sales['item_id'] == 512317789, ['price']] \
        = sales.loc[sales['item_id'] == 512317789, ['price']].fillna(23.34)

    # Inventory of 2884 for row 405855 and 1760 for row 405854, see notes_assumptions.txt
    sales.loc[[405855], 'inventory'] = 2884
    sales.loc[[405854], 'inventory'] = 1760

    # Correct the data type of inventory
    sales['inventory'] = sales['inventory'].astype(int)

    # Drop rows with zero price, inventory, and sales (17292 dropped)
    sales = sales.loc[~((sales['price'] == 0) & (sales['inventory'] == 0) & (sales['sales'] == 0))]

    return sales


def monthly_sales(sales):
    """ Totals all sales by month.

    This function has been checked for validity on 2018-04-14. First month of 2014 total sales should be $86672.21
    Last month should be $3642.47
    """

    # Get the relevant sale data, drop other columns
    sale_data = drop_sales_missing(sales)[['history_date', 'sales']]

    return sale_data.groupby([pd.Grouper(key='history_date', freq='M')]).sum()


def prepare_data(sales_raw, competitor_raw):
    """ The master function in this file that prepares the weekly_sales dataframe. """

    # Impute missing sales data
    sales_only = drop_sales_missing(sales_raw)

    # Drop the inventory column
    sales_only = drop_useless(sales_only)

    # Aggregate to weekly resolution
    print("Starting the slow stage, please wait a few minutes..")
    start_time = time.clock()
    weekly_sales = aggregate_sales(sales_only, competitor_raw)
    print("..this took:", round((time.clock() - start_time)/60, 1), 'minutes')

    # Save this in case something goes wrong later (uses hash to uniquely identify file)
    weekly_sales.to_csv("..\\3_Prepared_Data\\sales_weekly_" + str(hash(weekly_sales.values.tostring()))[:3] + ".csv")

    # Populate the on_promo column
    weekly_sales = add_promo(weekly_sales, sales_raw)

    # Populate the normalized_sales column
    weekly_sales = add_normalized_sales(weekly_sales)

    # Populate the price_index column
    weekly_sales = add_price_index(weekly_sales)

    # Populate the competitor_index column
    weekly_sales = add_competitor_index(weekly_sales)

    # Populate the similarity_group column
    weekly_sales = add_similarity(weekly_sales)

    return weekly_sales
