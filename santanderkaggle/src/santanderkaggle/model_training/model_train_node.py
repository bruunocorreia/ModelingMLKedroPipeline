from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import fmin, hp, tpe, space_eval
import joblib

def train_and_save_xgboost_model(x_train, y_train,features, seed=42, max_evals=30):
    """
    Train an XGBoost model with optimized hyperparameters and save it to a file.

    Parameters:
    - x_train (DataFrame): The training features.
    - y_train (Series): The target variable for training.
    - seed (int): Random seed for reproducibility. Default is 42.
    - max_evals (int): Maximum number of hyperparameter optimization evaluations. Default is 5.

    Returns:
    - model_xgb (XGBClassifier): Trained XGBoost model.

    This function trains an XGBoost model on the provided training data (x_train and y_train). It uses hyperparameter optimization
    with Bayesian optimization provided by Hyperopt to find the best set of hyperparameters for the model. The optimized model
    is then saved to a file using joblib.

    Example usage:
    model = train_and_save_xgboost_model(x_train, y_train, seed=42, max_evals=10)
    """
    
    x_train = x_train[features].copy()
    
    # Function for hyperparameter optimization
    def objective(params):
        """
        Objective function for hyperparameter optimization.

        Parameters:
        - params (dict): Dictionary of hyperparameters.

        Returns:
        - score (float): Negative ROC AUC score to be minimized.
        """
        params = {'max_depth': int(params['max_depth']),
                  'learning_rate': float(params['learning_rate']),
                  'n_estimators': int(params['n_estimators']),
                  'gamma': float(params['gamma']),
                  'min_child_weight': float(params['min_child_weight']),
                  'subsample': float(params['subsample']),
                  'colsample_bytree': float(params['colsample_bytree']),
                  'scale_pos_weight': int(params['scale_pos_weight'])
                 }

        clf = XGBClassifier(n_jobs=8,
                                objective='binary:logistic',
                                **params,
                                random_state=seed)

        score = cross_val_score(clf, x_train, y_train, scoring="roc_auc", cv=StratifiedKFold()).mean()

        print("Profit {:.3f} params {}".format(score, params))
        score = -1 * score
        return score

    space = {  # Hyperparameter search space
        'max_depth': hp.choice('max_depth', range(4, 8, 1)),
        'learning_rate': hp.quniform('learning_rate', 0.001, 0.01, 0.001),
        'n_estimators': hp.choice('n_estimators', range(50, 200, 20)),
        'gamma': hp.quniform('gamma', 5, 25, 5),
        'min_child_weight': hp.quniform('min_child_weight', 10, 50, 1),
        'subsample': hp.quniform('subsample', 0.4, 0.9, 0.01),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.4, 0.9, 0.01),
        'scale_pos_weight': hp.choice('scale_pos_weight', range(3, 15, 1)),
    }

    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals)

    # Best optimized hyperparameters
    best_params = space_eval(space, best)
    print("Best parameters:", best_params)

    print("Training the best model")

    # Training the model with the best hyperparameters
    model_xgb = XGBClassifier(n_jobs=-1, random_state=seed, objective='binary:logistic', **best_params)
    model_xgb.fit(x_train, y_train)

    # Saving the model to a file
    model_filename = 'best_xgboost_model.pkl'
    joblib.dump(model_xgb, model_filename)
    print(f"XGBoost model trained and saved as '{model_filename}'")

    return model_xgb