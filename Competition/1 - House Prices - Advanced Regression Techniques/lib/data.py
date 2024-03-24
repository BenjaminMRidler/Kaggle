import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.style import Style
from rich.color import Color
from scipy.stats import zscore
import numpy as np
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd


from .utility import yellow, green, red, cyan, white

################################################
# Data Handling
################################################

def analyze_data(df, feature=None):
    """
    Analyzes data in a DataFrame and prints a summary for each feature.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.
        feature (str, optional): The specific feature to analyze. Defaults to None.

    Returns:
        None
    """
    console = Console()
    panels = []

    column_width = 55  # Set the width of each column

    count = df.shape[0]
    for column in df.columns:
        unique_values = df[column].unique()
        unique_count = len(unique_values)

        # NaN or None percentages
        nan_count = df[column].isnull().sum()
        nan_percentage = df[column].isnull().mean() * 100

        text = f"{yellow('NaN/None values:')} {green(f'{nan_percentage:.2f}% | {nan_count} of {count}')}  {yellow('Type:')} {green(df[column].dtype.name)}\n"

        # If numeric, show distribution summary
        text += f"{yellow('Unique values:')} {green(unique_count)}\n"

        if len(unique_values) <= 50:
            text += white('"' + '", "'.join(unique_values.astype(str)) + '"\n')

        if pd.api.types.is_numeric_dtype(df[column]):
            text += yellow('Value distribution summary:\n')
            for stat, value in df[column].describe().to_frame().T.iloc[0].items():
                text += f"{yellow(f'{stat}:')} {green(f'{value:.2f}\t')}"

        panels.append(Panel(text, title=f'Feature: {column}', width=column_width, border_style="bright_cyan"))

    # Create a Columns object with the panels
    columns = Columns(panels, expand=True)
    console.print(columns)

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple

def analyze_missing_data(df) -> pd.Series:
    """
    Analyzes missing data in a DataFrame and prints a summary.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        None
    """
    if df is None:
        print(red("Error: DataFrame is None."))
        return

    if df.empty:
        print(red("Error: DataFrame is empty."))
        return

    if not isinstance(df, pd.DataFrame):
        print(red("Error: Invalid DataFrame."))
        return

    text = ''
    sorted_columns = df.isnull().mean().sort_values(ascending=False)
    sorted_columns = sorted_columns[sorted_columns > 0]
    for column in sorted_columns.index:
        percent_mising = df[column].isnull().mean()
        if percent_mising > 0.5:
            text += f"{white(column)}: {red(f'{percent_mising * 100:.2f}% of values are NaN\n')}"
        elif percent_mising > 0.25:
            text += f"{white(column)}: {yellow(f'{percent_mising * 100:.2f}% of values are NaN\n')}"
        elif percent_mising > 0:
            text += f"{white(column)}: {green(f'{percent_mising * 100:.2f}% of values are NaN\n')}"

    print(Panel(text, title=f'Features with Missing Data', border_style="bright_cyan"))
    return sorted_columns

def apply_zscore_bounds(df, z_threshold=2):
    """
    Apply z-score bounds to the DataFrame by clipping or filtering out outliers.

    Args:
        df (pd.DataFrame): The DataFrame to apply z-score bounds to.
        z_threshold (float, optional): The z-score threshold. Defaults to 2.

    Returns:
        pd.DataFrame: The DataFrame with z-score bounds applied.
    """
    try:
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()

            # Calculate upper and lower bounds
            upper_bound = mean + z_threshold * std
            lower_bound = mean - z_threshold * std

            # Apply clipping or filtering based on bounds
            # For clipping
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

            # Alternatively, for filtering out outliers:
            # df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df
    except Exception as e:
        print(red("Error: Failed to apply z-score bounds."))
        print(red(f"Details: {str(e)}"))
        return None

def impute_missing_data(impute_target: str, df: pd.DataFrame, temp_normalized_df: pd.DataFrame, impactful_features_list: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if len(impactful_features_list) == 0:
        print(f'{yellow("Warning: No impactful features found. Using normalized values for imputation.")}')
        df[impute_target] = temp_normalized_df[impute_target]
        return df, temp_normalized_df

    df[impute_target + "_original"] = df[impute_target]
    df[impute_target + "_imputed"] = 0
    
    # Make model
    training_feature = temp_normalized_df[impactful_features_list]
    training_feature[impute_target] = temp_normalized_df[impute_target]
    
    _ , model = make_impactful_feature_model(impute_target, df, training_feature)
    
    # Create predicted values
    features = temp_normalized_df[impactful_features_list]
    predicted_values = model.predict(features)
    
    # Find NA positions in the original DataFrame
    na_mask = df[impute_target].isna()  
    
    # Replace NA positions with predicted values in the normalized DataFrame
    temp_normalized_df.loc[na_mask, impute_target] = predicted_values[na_mask]  

    # Update the original DataFrame with the imputed values and normalized values 
    df[impute_target] = temp_normalized_df[impute_target]
    
    # Update the indicator that the value in this row is imputed
    df.loc[na_mask, impute_target + "_imputed"] = 1

    return df, temp_normalized_df

def load_data(file_name) -> pd.DataFrame:
    """
    Load data from a CSV file and return it as a pandas DataFrame.

    Parameters:
    file_name (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        print(red(f"Error: File '{file_name}' not found."))
        return None
    except Exception as e:
        print(red(f"Error: Failed to load data from '{file_name}'."))
        print(red(f"Details: {str(e)}"))
        return None

    df = pd.read_csv(file_name)
    return df

def make_impactful_feature_model(feature: str, df: pd.DataFrame, temp_normalized_df: pd.DataFrame) -> Tuple[pd.DataFrame, RandomForestRegressor]:
    """
    Analyzes the importance of features in predicting the specified feature's values using a Random Forest Regressor.
    This function first prepares the data by removing rows with NA values in the specified feature column.
    Then, it trains a Random Forest model to predict the feature's values based on other features in the dataset.

    Parameters:
    - feature (str): The name of the feature to analyze.
    - df (pd.DataFrame): The original DataFrame containing the feature and potential predictors.
    - temp_normalized_df (pd.DataFrame): A DataFrame where the specified feature might have been normalized or processed.

    Returns:
    - Tuple[pd.DataFrame, RandomForestRegressor]: A tuple containing a DataFrame of feature importances and the trained Random Forest model.
    """
    # Overwrite the feature column with the original containing 'NA' values
    temp_normalized_df = temp_normalized_df.copy()
    
    original_feature = feature + '_original'
    temp_normalized_df[original_feature] = df[feature]
    
    # Drop all of the rows in which the feature has NA values
    X = temp_normalized_df.dropna(subset=[original_feature])
    
    # Get our dependent variable
    y = X[feature]
    
    # Drop the feature row
    X.drop(columns=[feature, original_feature], inplace=True)
    
    # Splitting the data for model training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model (optional)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Model RMSE on test set: {rmse:.2f}")
    
    # Produce a DataFrame with columns and importance ranking
    df_feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
    return df_feature_importances, model

def remove_columns(df, columns) -> pd.DataFrame:
    """
    Remove specified columns from a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to remove columns from.
    columns (list): List of column names to remove.

    Returns:
    pd.DataFrame: The DataFrame with specified columns removed.
    """
    if len(columns) == 0:
        return
    
    if df is None:
        print(red("Error: DataFrame is None."))
        return None

    if df.empty:
        print(red("Error: DataFrame is empty."))
        return None

    if not isinstance(df, pd.DataFrame):
        print(red("Error: Invalid DataFrame."))
        return None

    non_existing_columns = [col for col in columns if col not in df.columns]
    if non_existing_columns:
        print(red(f"Error: The following columns do not exist in the DataFrame: {', '.join(non_existing_columns)}"))
        return None

    try:
        df = df.drop(columns, axis=1)
        return df
    except Exception as e:
        print(red(f"Error: Failed to remove columns."))
        print(red(f"Details: {str(e)}"))
        return None

def prepare_data(df) -> pd.DataFrame:
    """
    Prepare the data by mapping categorical data to numeric, imputing missing data to median,
    scaling data to 0-1, and mapping outliers to upper/lower.

    Args:
        df (pd.DataFrame): The DataFrame to prepare.

    Returns:
        pd.DataFrame: The prepared DataFrame.
    """
    if df is None:
        print(red("Error: DataFrame is None."))
        return None

    if df.empty:
        print(red("Error: DataFrame is empty."))
        return None

    if not isinstance(df, pd.DataFrame):
        print(red("Error: Invalid DataFrame."))
        return None
    
    transformed_df = df.copy()

    # Map categorical data to numeric
    categorical_columns = transformed_df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        transformed_df[column] = transformed_df[column].astype('category').cat.codes

    # Impute missing data to mean
    numeric_columns = transformed_df.select_dtypes(include=['float', 'int']).columns
    for column in numeric_columns:
        transformed_df[column].fillna(transformed_df[column].median(), inplace=True)

    # Map outliers to upper/lower
    transformed_df = apply_zscore_bounds(transformed_df)

    # Scale data to 0-1
    scaler = MinMaxScaler()
    transformed_df[numeric_columns] = scaler.fit_transform(transformed_df[numeric_columns])

    return transformed_df