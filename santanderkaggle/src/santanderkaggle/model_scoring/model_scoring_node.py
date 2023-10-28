import pandas
def scoring_dataframe(df, id_columns, features, model):
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

    # Load the validation CSV file, filtering by the specified columns
    df_validation = df.set_index(id_columns)[features]

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