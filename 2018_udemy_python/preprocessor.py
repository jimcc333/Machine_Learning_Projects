# Data preprocessing
# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\Data_Preprocessing\\"
dataset = pd.read_csv(data_path + "Data.csv")
print("Original Data:\n", dataset, "\n")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle missing data by replacing with average
imputer1 = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer1 = imputer1.fit(X[:, 1:3])
X[:, 1:3] = imputer1.transform(X[:, 1:3])

# Encode categorical data
labeler_X = LabelEncoder()
X[:, 0] = labeler_X.fit_transform(X[:, 0])
encoder1 = OneHotEncoder(categorical_features=[0])
X = encoder1.fit_transform(X).toarray()
labeler_y = LabelEncoder()
y = labeler_y.fit_transform(y)

# Scale data
scaler_X = StandardScaler()
X[:, 3:] = scaler_X.fit_transform(X[:, 3:])

print("Features before splitting:\n", X)
print("Observations before splitting:\n", y)

# Split data to training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
