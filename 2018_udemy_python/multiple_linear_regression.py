""" Multiple Linear Regression Model. """
# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\"
dataset = pd.read_csv(data_path + "50_Startups.csv")
print("Original Data:\n", dataset.head(10), "\n")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Encode categorical data
labeler_X = LabelEncoder()
X[:, 3] = labeler_X.fit_transform(X[:, 3])
encoder1 = OneHotEncoder(categorical_features=[3])
X = encoder1.fit_transform(X).toarray()
X = X[:, 1:]    # Avoiding the dummy variable trap

# Split data to training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Build model
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)
y_predicted = regressor1.predict(X_test)
print("Error of each prediction (%):")
for index in range(len(y_test)):
    print(round(100*abs(y_test[index] - y_predicted[index])/(y_predicted[index]), 2),
          end=', ')
print()

# --- Backwards feature elimination ---
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor1 = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor1.summary())
# Remove 2
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor1 = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor1.summary())
# Remove 1
X_opt = X[:, [0, 3, 4, 5]]
regressor1 = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor1.summary())
# Remove 3
X_opt = X[:, [0, 4, 5]]
regressor1 = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor1.summary())

# Re-build model
regressor1 = LinearRegression()
regressor1.fit(X_train[:, [0, 4, 5]], y_train)
y_predicted = regressor1.predict(X_test[:, [0, 4, 5]])
print("Error of each prediction (%):")
for index in range(len(y_test)):
    print(round(100*abs(y_test[index] - y_predicted[index])/(y_predicted[index]), 2),
          end=', ')
print()

print("\n\nEnd of script.\n")
