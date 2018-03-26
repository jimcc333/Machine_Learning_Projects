""" Multiple Linear Regression Model. """
# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\"
dataset = pd.read_csv(data_path + "50_Startups.csv")
print("Original Data:\n", dataset, "\n")

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

