""" Functions to read data for the 2018_BlackLocus project. """

# Imports
import pandas as pd
import numpy as np


def read_competitor(data_path="..\\2_Exploration\\raw_blacklocus_interview_competitor_prices.csv",
                    talk=False):
    """ Reads the raw competitor prices data and returns it as a pandas data frame. """

    competitor_raw = pd.read_csv(data_path, parse_dates=['history_date'])

    if talk:
        print(competitor_raw.describe(), '\n\n')
        print(competitor_raw.head())

    return competitor_raw


def read_model_data(data_path="..\\3_Prepared_Data\\model_data.csv", talk=False):
    """ Reads the data prepared for building models. """

    model_data = pd.read_csv(data_path)

    if talk:
        print(model_data.describe(), '\n\n')
        print(model_data.dtypes, '\n\n')
        print(model_data.head())

    return model_data


def read_sales(data_path="..\\2_Exploration\\raw_blacklocus_interview_sales.csv", talk=False):
    """ Reads the raw sales data and returns it as a pandas data frame. """

    sales_raw = pd.read_csv(data_path, parse_dates=['history_date', 'promo_start_date', 'promo_end_date'])

    if talk:
        print(sales_raw.describe(), '\n\n')
        print(sales_raw.dtypes, '\n\n')
        print(sales_raw.head())

    return sales_raw


def read_sales_weekly(data_path="..\\3_Prepared_Data\\sales_weekly_prepared_adjusted.csv", talk=False):
    """ Reads the weekly_sales data and returns it as a pandas data frame. """

    sales_weekly_raw = pd.read_csv(data_path, parse_dates=['week_begin', 'competitor_delta'])

    # Drop empty rows
    sales_weekly_raw = sales_weekly_raw[np.isfinite(sales_weekly_raw['week'])]

    # Set int columns
    sales_weekly_raw['week'] = sales_weekly_raw['week'].astype(int)
    sales_weekly_raw['month'] = sales_weekly_raw['month'].astype(int)
    sales_weekly_raw['item_id'] = sales_weekly_raw['item_id'].astype(int)
    sales_weekly_raw['category_id'] = sales_weekly_raw['category_id'].astype(int)

    if talk:
        print(sales_weekly_raw.describe(), '\n\n')
        print(sales_weekly_raw.dtypes, '\n\n')
        print(sales_weekly_raw.head())

    return sales_weekly_raw

