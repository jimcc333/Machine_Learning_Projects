""" The polynomial linear regression model. """
# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\"
dataset = pd.read_csv(data_path + "Position_Salaries.csv")
print("Original Data:\n", dataset, "\n")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Create and fit a linear regressor
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Create and fit a polynomial regressor
polynomial_regressor = PolynomialFeatures(degree=2)
X_polynomial = polynomial_regressor.fit_transform(X)
print(X_polynomial)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial, y)


