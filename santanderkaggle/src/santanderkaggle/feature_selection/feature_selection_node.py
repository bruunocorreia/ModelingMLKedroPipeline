from utils import variance_threshold
from utils import correlation
from utils import variance_threshold
from utils import correlation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

def UnsupervisedFeatureSelection(x_train, variance, corr):
    """
    Perform unsupervised feature selection based on variance and correlation.

    Args:
        x_train (pandas.DataFrame): The training feature data.
        id_columns (list): List of ID columns that should be included.
        variance (float): Variance threshold for feature selection (default: 0.01).
        corr (float): Correlation threshold for feature selection (default: 0.8).

    Returns:
        list: A list of selected feature columns after feature selection.
    """
    # Define type of columns
    num_vars = x_train.select_dtypes(include=['float64', 'int64']).columns
    cat_vars = x_train.select_dtypes(include=['object']).columns

    print('initial numerical vars =', len(num_vars))
    print('initial categorical vars =', len(cat_vars))

     #Aplicando variance e correlação
    num_vars_vt = variance_threshold(x_train.filter(num_vars),threshold =variance)
    selected_columns = correlation(x_train.filter(num_vars_vt), threshold = corr)
    
    print('Total feature to keep =', len(selected_columns))
    return selected_columns

def SupervisedFeatureSelection(x_train,
                               y_train, 
                               features,
                               metric="roc_auc",
                               cv=3,
                               model = RandomForestClassifier(),
                               step=0.1):
    """
    Perform supervised feature selection using Recursive Feature Elimination with Cross-Validation (RFECV).

    Args:
        x_train (pandas.DataFrame): The training feature data.
        y_train (pandas.Series): The training target variable.
        id_columns (list): List of ID columns to be dropped.
        features (list): List of feature columns to select from.
        metric (str): The scoring metric for feature selection (default: "roc_auc").
        cv (int): Number of cross-validation folds (default: 3).
        step (float): The step size for feature elimination (default: 0.1).

    Returns:
        list: A list of selected feature columns.
    """
    # Filter and drop ID columns
    x_train = x_train[features].copy()
    print('initial number of vars =', len(x_train.columns))

    # Instantiate RFECV visualizer with a linear Random Forest classifier
    visualizer = RFECV(model, scoring=metric, cv=cv, step=step)

    # Fit the data to the visualizer
    visualizer.fit(x_train, y_train)

    # Display the feature selection plot
    visualizer.show()

    # Get the optimal number of selected features and the list of best features
    optimal_num_features = visualizer.n_features_
    best_features = list(x_train.columns[visualizer.support_])

    print('Optimal number of features:', optimal_num_features)
    print('Selected features:', best_features)
    
    return best_features

