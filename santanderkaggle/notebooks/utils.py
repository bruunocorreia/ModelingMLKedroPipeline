import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

def train_test_split_dist(x, y, test_size,seed):
    """
    Split the dataset into training and testing sets.

    Args:
        x (pandas.DataFrame): The feature data.
        y (pandas.Series): The target variable.
        test_size (float): The proportion of data to include in the test split.

    Returns:
        x_train (pandas.DataFrame): Training feature data.
        x_test (pandas.DataFrame): Testing feature data.
        y_train (pandas.Series): Training target variable.
        y_test (pandas.Series): Testing target variable.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)

    print("Number of rows in training set:", len(x_train))
    print('---------------------------')
    print('Checking target distribution in the training set:')
    print(y_train.value_counts() / len(y_train) * 100)
    print('---------------------------')
    print('Checking target distribution in the testing set:')
    print("Number of rows in testing set:", len(x_test))
    print('---------------------------')
    print(y_test.value_counts() / len(y_test) * 100)
    
    return x_train, x_test, y_train, y_test


def variance_threshold(df, threshold):
    """
    Remove low-variance features from the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        threshold (float): The threshold below which features will be removed.

    Returns:
        pandas.DataFrame: The DataFrame with low-variance features removed.
    """
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(df)
    mask = vt.get_support()
    num_vars_reduced = df.iloc[:, mask]
    return list(num_vars_reduced.columns)

def correlation(df, threshold):
    """
    Remove highly correlated features from the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        threshold (float): The correlation threshold above which features will be removed.

    Returns:
        pandas.DataFrame: The DataFrame with highly correlated features removed.
    """
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = df.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i]  # Getting the name of the column
                col_corr.add(colname)
                if colname in df.columns:
                    del df[colname]  # Deleting the column from the dataset

    return list(df.columns)
