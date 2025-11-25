import pandas as pd

def add_hyperparameters(data: pd.DataFrame, hyperparameters_list: dict) -> pd.DataFrame:
    """
    Add new columns to a dataframe based on functions applied to each row.

    Parameters:
    - data (pd.DataFrame): input dataframe
    - hyperparameters_list (dict): {column_name: function(row)}. Each function
      receives a row (pd.Series) and returns a value.

    Returns:
    - pd.DataFrame: dataframe with the new hyperparameter columns added.
    """
    
    # Ensure we don't modify the original dataframe
    df = data.copy()
    
    for col_name, func in hyperparameters_list.items():
        df[col_name] = df.apply(func, axis=1)

    return df


def remove_columns(data: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Supprime les colonnes spécifiées d'un DataFrame.

    Parameters:
    - data: pd.DataFrame, le DataFrame initial
    - columns_to_remove: list, liste des noms de colonnes à supprimer

    Returns:
    - pd.DataFrame, le DataFrame après suppression des colonnes
    """
    return data.drop(columns=[col for col in columns_to_remove if col in data.columns])
