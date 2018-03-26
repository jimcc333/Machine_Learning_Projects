"""Simple linear regression model. """

# Author: Cem Bagdatlioglu

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Import data
data_path = "D:\\Data Science\\2018_udemy_ml\\downloaded\\Simple_Linear_Regression\\"
dataset = pd.read_csv(data_path + "Salary_Data.csv")
print("Original Data:\n", dataset, "\n")

X = dataset.iloc[:, :1].values
y = dataset.iloc[:, -1].values

# Split data to training/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Model
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)
y_predicted = regressor1.predict(X_test)
print("% Error of each prediction:")
for index in range(len(y_test)):
    print(round(100*abs(y_test[index] - y_predicted[index])/(y_predicted[index]), 2),
          end=', ')
print()

# Plot model
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='red', marker='x', label='Test Data')
plt.plot(X_train, regressor1.predict(X_train), label='Fit Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary, $")
plt.legend()
plt.show()

print("\n\nEnd of script.\n")
