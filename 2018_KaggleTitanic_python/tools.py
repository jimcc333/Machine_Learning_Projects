def fill_missing(input_df, method):
    """ Take a data frame with missing values in one column and fills it in using a user-defined method.

    Args:
        input_df: pandas data frame that has missing values in one column.
        method (str): The method to fill in the data.
    Returns:
        filled_df: pandas data frame with the filled missing values.
    """

    #TODO: fill missing doesn't work yet.

    # Import Libraries
    import pandas

    # Divide data frame into two groups based on empty values
    filled_df = input_df.fillna(0)

    return filled_df


def prepare_csv(training_path="..\\2. Prepared Data\\train.csv", test_path="..\\2. Prepared Data\\test.csv"):
    """ Read and prepare the two CSV files that will be used to train and test the machine learning models.

    Data preparation involves handling missing data. Each way of handling missing data will create a different data
    frame. The options to handle missing data in this function are (this corresponds to the order in the returned lists:
        1. Ignore the rows with missing data
        2. Ignore the columns that have rows with missing data (Age)
        3. Replace missing data with data average

    Args:
        training_path (str): Where the training CSV data is located.
        test_path (str): Where the prepared testing CSV data is located.
    Returns:
        training_data and test_data, lists containing pandas data frames.
    """

    # Import Libraries
    import pandas

    from sklearn import preprocessing

    # Import Data
    try:
        training_data_raw = pandas.read_csv(training_path)
        print("Data at", training_path, "successfully loaded!")
        test_data_raw = pandas.read_csv(test_path)
        print("Data at", test_path, "successfully loaded!")
    except FileNotFoundError:
        print("\n\nERROR")
        print("Data could not be loaded, check input paths.")
        exit()

    # Check Data
    print("Dimensions of the training data are:", training_data_raw.shape)
    print("Dimensions of the test data are:", test_data_raw.shape)
    # print("\nData Description:\n", training_data_raw.describe())
    # print("First six rows:")
    # print(training_data_raw.head(6))
    if training_data_raw.shape[1] != test_data_raw.shape[1] + 1:
        print("\n\n\nUnexpected columns!\n")

    # Prepare Data
    print("Missing data per dimension:")
    print(training_data_raw.isnull().sum())

    # Standardize
    training_data_scaled = (training_data_raw - training_data_raw.min()) / (training_data_raw.max() -
                                                                            training_data_raw.min())
    test_data_scaled = (test_data_raw - test_data_raw.min()) / (test_data_raw.max() - test_data_raw.min())

    # ---Option 1: Remove all rows with missing data.
    # ---Option 2: Remove the age column.
    # ---Option 3: Replace missing data with the median
    # ---Option 4: Replace missing data with value -10
    training_data = [training_data_scaled.dropna(), training_data_scaled.drop(['Age'], axis=1),
                     training_data_scaled.fillna(training_data_scaled.median(axis=0, skipna=True)['Age']),
                     training_data_scaled.fillna(-10)]
    test_data = [test_data_scaled.dropna(), test_data_scaled.drop(['Age'], axis=1),
                 test_data_scaled.fillna(training_data_scaled.median(axis=0, skipna=True)['Age']),
                 test_data_scaled.fillna(-10)]

    return training_data, test_data


def build_model(model, data_matrix, validation_size, trials=10, scoring='accuracy', n_splits=10):
    """ Test model a number of times and return the median accuracy. """

    # Import libraries
    from sklearn import model_selection
    import numpy as np

    model_accuracy = []
    accuracy_std = []

    for seed in range(trials):
        # Divide data to new random train-test set
        x = data_matrix.iloc[:, 2:8]
        y = data_matrix.iloc[:, 1]
        x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x, y, test_size=validation_size,
                                                                                        random_state=seed)
        kfold = model_selection.KFold(n_splits=n_splits, random_state=seed)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        model_accuracy.append(cv_results.mean())
        accuracy_std.append(cv_results.std())

    return [np.median(model_accuracy), np.median(accuracy_std)]
