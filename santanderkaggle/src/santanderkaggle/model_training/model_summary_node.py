import joblib
import xgboost as xgb
import pandas as pd

def load_xgboost_model_and_get_feature_importance(model):
    """
    Load a saved XGBoost model and get feature importance.

    Parameters:
    - model_filename (str): The filename of the saved XGBoost model.
    - x_train (DataFrame): The training features used for calculating feature importance.

    Returns:
    - model (XGBClassifier): Loaded XGBoost model.
    - feature_importance (DataFrame): DataFrame containing feature importance scores.

    This function loads a previously saved XGBoost model, calculates feature importance,
    and returns a DataFrame with feature importance scores.
    """
    
    # Get feature importance scores
    feature_scores = model.get_booster().get_score(importance_type='weight')
    total = sum(feature_scores.values())
    feature_importance = pd.DataFrame(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True), columns=['Feature', 'Score'])
    feature_importance['Score'] = feature_importance['Score'] / total  # Normalize scores

    return feature_importance