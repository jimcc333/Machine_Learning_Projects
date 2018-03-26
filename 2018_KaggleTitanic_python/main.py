""" Machine Learning Model for the Titanic Data Set.

Author: Cem Bagdatlioglu, Date: 2018-03-23
This script uses the inputs defined in the Model Inputs block. Creates several sets of data (by preparing the inputted
training data in various ways) for training and creates several models for each data frame. Reports the best model and
makes predictions on the test data. Also varies the data division seed for train-validation for each model.
"""

# Import libraries
from tools import prepare_csv
from tools import build_model

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ---- Model Inputs ----
# -------------------------------------------------------------------------------------------------------------------- #
training_path = "..\\2. Prepared Data\\train.csv"
"""str: Where the prepared training CSV data is located."""

test_path = "..\\2. Prepared Data\\test.csv"
"""str: Where the prepared testing CSV data is located."""

scoring = "accuracy"
"""str: The method used to score models during building."""

validation_size = 0.20
"""float: Fraction of data to use for validation."""

seed = 1
"""int: Train-validation split random seed."""

trials = 20
"""int: Number of different seed trails."""
# -------------------------------------------------------------------------------------------------------------------- #

# Clear screen
try:
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
except ImportError:
    os = None

# Get prepared data
training_data, test_data = prepare_csv(training_path=training_path, test_path=test_path)

# print(training_data[1].head(20))
# print(training_data[2].head(20))
# print(training_data[3].head(20))

# Create Models
models = list()
# models.append(("LR", LogisticRegression()))
# models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
# models.append(("NB", GaussianNB()))
# models.append(("SVM", SVC()))

# Evaluate each model in turn
results = []
names = []

print("\nBuilding models")
for name, model in models:
    for data_index in range(len(training_data)):
        accuracy = build_model(model, training_data[data_index], validation_size, trials=trials)
        results.append(accuracy)
        names.append(name)
        print(name, data_index, round(accuracy[0], 4), round(accuracy[1], 5))

print("\nBest model is", names[[x[0] for x in results].index(max([x[0] for x in results]))])

# Predict test data
best_model = DecisionTreeClassifier()
best_data_index = 1

x = training_data[best_data_index].iloc[:, 2:8]
y = training_data[best_data_index].iloc[:, 1]

best_model.fit(x, y)
predictions = best_model.predict(test_data[best_data_index].iloc[:, 1:8])
for value in predictions:
    print(value)

print('The end.')
