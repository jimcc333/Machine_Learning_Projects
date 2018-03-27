""" The polynomial linear regression model. """
# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\"
dataset = pd.read_csv(data_path + "Position_Salaries.csv")
print("Original Data:\n", dataset, "\n")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Create and fit a linear regressor
linear_regressor1 = LinearRegression()
linear_regressor1.fit(X, y)

# Create and fit a polynomial regressor
polynomial_regressor = PolynomialFeatures(degree=2)
X_polynomial = polynomial_regressor.fit_transform(X)
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial, y)

# Show linear regression results
plt.scatter(X, y, color="black")
plt.plot(X, linear_regressor1.predict(X), color="red", label="Linear Model")
plt.plot(X, linear_regressor2.predict(polynomial_regressor.fit_transform(X)),
         color="blue", label="Polynomial Model")
plt.xlabel("Level")
plt.ylabel("Salary ($)")
plt.title("Linear Regression Model")
plt.legend()
plt.show()
