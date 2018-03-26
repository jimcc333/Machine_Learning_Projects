""" Machine Learning Model for the Titanic Data Set.

Author: Cem Bagdatlioglu, Date: 2018-03-22
This script uses the inputs defined in the Model Inputs block. Creates several sets of data (by preparing the inputted
training data in various ways) for training and creates several models for each data frame. Reports the best model and
makes predictions on the test data.
"""

# ---- Import ----
import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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
# -------------------------------------------------------------------------------------------------------------------- #

# ---- Preliminary Checks ----
# Clear screen
try:
    import os
    os.system('cls' if os.name == 'nt' else 'clear')
except ImportError:
    os = None

print("--Checking Required Libraries---")

# Python version
print('Python: {}'.format(sys.version))
# scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
print('sklearn: {}'.format(sklearn.__version__))

print("..all libraries are available in this environment\n")

# ---- Import Data ----
try:
    training_data = pandas.read_csv(training_path)
    print("Data at", training_path, "successfully loaded!")
    test_data = pandas.read_csv(test_path)
    print("Data at", test_path, "successfully loaded!")
except FileNotFoundError:
    print("\n\nERROR")
    print("Data could not be loaded, check input paths.")
    exit()

# ---- Data Check ----
print("Dimensions of the training data are:", training_data.shape)
print("Dimensions of the test data are:", test_data.shape)
# print("\nData Description:\n", training_data.describe())
# print("First six rows:")
# print(training_data.head(6))
if training_data.shape[1] != test_data.shape[1] + 1:
    print("\n\n\nUnexpected columns!\n")
input("Press Enter to continue...\n")

# ---- Prepare Data ----
print("Missing data per dimension:")
print(training_data.isnull().sum())

# Remove all rows with missing data
# print(training_data.isnull())
training_data = training_data.dropna()

# Divide data into training and validation sets
data_matrix = training_data.values
X = data_matrix[:, 3:9]  # Ignores ID
Y = data_matrix[:, 1]
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)

# ---- Create Models ----
models = list()
models.append(("LR", LogisticRegression()))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC()))

# Evaluate each model in turn
results = []
names = []

print("\nBuilding models")
for name, model in models:
    kfold = model_selection.KFold(n_splits=6, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, round(cv_results.mean(), 4), round(cv_results.std(), 5))

# Plot results
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# ---- Make predictions ----
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print("\nAccuracy:\n", round(accuracy_score(Y_validation, predictions), 4))
print("Confusion Matrix:\n", confusion_matrix(Y_validation, predictions))
print("Classification Report:\n", classification_report(Y_validation, predictions))

