""" The main script for python tools in 2018_BlackLocus.

    Author:     Cem Bagdatlioglu
    Project:    2018_BlackLocus
    Date:       April 2018

    This script is used to read, explore, and manipulate the data.
    Visualization and exploration functions are commented out after being used.
    Functions may not work if called out of order.
"""

# Imports
import read_data
import prep_data
import visualization
import model
import feature_select
import analysis

import time
import hashlib

# - - - Step 1: Explore and transform the data - - -
#  - - - - - - - - - - - - - - - - - - - - - - - - -

# Read the data
# sales_raw = read_data.read_sales()
# competitor_raw = read_data.read_competitor()

# Prepare the weekly_sales table
# weekly_sales = prep_data.prepare_data(sales_raw, competitor_raw)

# Read the prepared weekly sales data (should have been prepared before)
# weekly_sales = read_data.read_sales_weekly()

# Plot weekly product sales
# visualization.plot_weekly_sales(weekly_sales)


# - - - Step 2: Model the normalized sales using a generalized linear model - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Read the prepared data for building models
model_data = read_data.read_model_data()

# Make the overlay plots requested in the problem statement
# visualization.make_overlay_plots(model_data)

# Estimate promotion lift on promo products
promo_items = [512018493, 512317770, 512317789, 512319146, 512319171, 513103198, 514002186, 514002190, 515375113]

for item in promo_items:
    analysis.estimate_lift(model_data, item)




