import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from kedro.pipeline import node
from typing import Dict, Tuple
from kedro.extras.datasets.matplotlib import MatplotlibWriter
import io

def EvaluateModel(model, x_train, y_train, x_test, y_test, features):
    """
    Score the XGBoost model and calculate ROC AUC and KS statistics.

    Parameters:
    model : The trained classifier.
    x_train (pd.DataFrame): Training data features.
    y_train (pd.Series): Training data labels.
    x_test (pd.DataFrame): Test data features.
    y_test (pd.Series): Test data labels.
    feature_to_keep_s (list): List of selected features for scoring.

    Returns:
    pd.DataFrame: DataFrame with ROC AUC and KS statistics for training and test datasets.
    """
    
    # Convert the DataFrame to a list
    features = features['0'].values.tolist()

    # Use the boolean mask to select the desired columns
    x_train = x_train[features].fillna(0)
    x_test = x_test[features].fillna(0)
    
    #Filter feature in dataset
    x_train_scoring = x_train.copy()
    x_test_scoring = x_test.copy()

    # Scoring the model
    proba_train = model.predict_proba(x_train_scoring)[:, 1]
    proba_test = model.predict_proba(x_test_scoring)[:, 1]

    # Calculate ROC AUC scores
    auc_train = roc_auc_score(y_train, proba_train)
    auc_test = roc_auc_score(y_test, proba_test)

    # Calculate ROC curves
    fpr_train, tpr_train, _ = roc_curve(y_train, proba_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, proba_test)

    # Create a DataFrame with the statistics
    statistics_df = pd.DataFrame({
        'Dataset': ['Train', 'Test'],
        'ROC_AUC': [auc_train, auc_test]
    })
    
    # Plot the ROC curves
    plt.figure()
    plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='ROC curve (Train) - AUC = {:.2f}'.format(auc_train))
    plt.plot(fpr_test, tpr_test, color='navy', lw=2, label='ROC curve (Test) - AUC = {:.2f}'.format(auc_test))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    print('-----------------------------------------------------')
    # Summarize scores
    print('ROC AUC Train = {:.3f}'.format(auc_train))
    print('ROC AUC Test = {:.3f}'.format(auc_test))

    return statistics_df

