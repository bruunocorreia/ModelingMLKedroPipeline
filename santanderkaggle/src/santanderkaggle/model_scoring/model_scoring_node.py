import pandas
from kedro.pipeline import node
from typing import Dict, Tuple
import pandas as pd

def scoring_dataframe(df, features, model,parameters: Dict) -> Tuple:
    """
    Create a DataFrame with deciles from a validation CSV file using a trained model.

    Parameters:
    - df: Path to the validation CSV file.
    - id_columns (str or list): The column(s) that act as unique identifiers.
    - features (list): List of feature columns to keep.
    - model: Trained machine learning model (e.g., XGBoost).

    Returns:
    - DataFrame: A DataFrame with 'score' and 'deciles' columns.

    """
    #get params
    id_columns = parameters["id_columns"]
    
    # Convert the DataFrame to a list
    features = features['0'].values.tolist()

    # Use the boolean mask to select the desired columns
    df_validation = df.set_index(id_columns)[features].fillna(0)
    
    # Use the model to predict the 'score' for each record
    df_validation['score'] = model.predict_proba(df_validation)[:, 1]
    
    #set df scoring
    scoring_df = df_validation[['score']]
    
    # Calculate the deciles
    scoring_df['deciles'] = pd.qcut(scoring_df['score'], q=10, labels=False)
    scoring_df['deciles'] = scoring_df['deciles'] + 1  # Start deciles from 1
    
    #print distribution
    result = scoring_df.groupby('deciles').size().reset_index(name='count')
    result = scoring_df.groupby('deciles').agg({'score': ['count', 'max', 'min']}).reset_index()
    print(result)

    return scoring_df