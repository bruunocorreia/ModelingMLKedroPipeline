import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


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
