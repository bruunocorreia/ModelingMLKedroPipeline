import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_split_dist(df_train, test_size,seed,id_columns,target_column):
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
    #Define x and y
    y = df_train[target_column]
    x = df_train.drop(columns=id_columns + target_column).fillna(0)
    
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