import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from rich import print
from rich.console import Console
from rich.panel import Panel

def analyze_impact_features(X, target, exclude_cols=None):
    """
    Analyzes the impact of features on 'LotFrontage' using a Random Forest regressor.
    
    Parameters:
    - df: Pandas DataFrame containing the dataset including 'LotFrontage'
    - target: the name of the target variable (default is 'LotFrontage')
    - exclude_cols: a list of column names to exclude from the analysis (e.g., ID columns)
    
    Returns:
    - Plots feature importances and returns a sorted DataFrame of feature importances.
    """
    cat = X.select_dtypes(include=['object']).columns

    discrete_cols = X.select_dtypes(include='object').columns
    print(f"Converting discrete columns to numeric: {discrete_cols}")
    for column in discrete_cols:
        for i, value in enumerate(X[column].unique()):
            X[column] = X[column].replace(value, i)

    X_encoded = pd.get_dummies(X[discrete_cols], drop_first=True)
    X = pd.concat([X.drop(columns=discrete_cols), X_encoded], axis=1)

    # Drop rows with missing data
    X.dropna(inplace=True)

    # Make Y
    y = X[target]
    X = X.drop(columns=[target])

    # Impute missing values in predictors
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)

    # Initialize and train the Random Forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.dropna())

    # Predict on the test set
    y_pred = rf.predict(X_test)

    # Calculate and print the RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE on the test set: {rmse:.2f}")

    # Feature importance
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                        index=X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)

    print(feature_importances)    # Plotting feature importances
    plt.figure(figsize=(10, 6))
    feature_importances['importance'].head(10).plot(kind='bar')
    plt.title('Top 10 Feature Importances for Predicting LotFrontage')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.show()
    
    return feature_importances

def analysis_features_missing_data(df):
    
    text = ''
    sorted_columns = df.isnull().mean().sort_values(ascending=False)
    for column in sorted_columns.columns:
        if df[column].isnull().mean() > 0.5:
            text += f"{white(column)}: {red(f'{df[column].isnull().mean() * 100:.2f}% of values are NaN\n')}"
        elif df[column].isnull().mean() > 0.25:
            text += f"{white(column)}: {yellow(f'{df[column].isnull().mean() * 100:.2f}% of values are NaN\n')}"
        elif df[column].isnull().mean() > 0:
            text += f"{white(column)}: {green(f'{df[column].isnull().mean() * 100:.2f}% of values are NaN\n')}"
    
    print(Panel(text, title=f'Features with Missing Data', border_style="bright_cyan"))
    
    
def analyze_features(df, feature=None):
    """
    Analyzes features of a pandas dataframe.

    Parameters:
    - df: pandas.DataFrame

    Prints:
    - Feature name
    - Percentage of NaN or None values
    - Value distribution summary (for numeric features)
    - Unique values (for categorical features with fewer than 10 unique values)
    """
    
    from rich.columns import Columns
    from rich.console import Console
    from textwrap import fill

    console = Console()
    panels = []

    column_width = 55  # Set the width of each column

    count = df.shape[0]
    text = ''
    if feature in df.columns:
        df = df[feature]
    
    for column in df.columns:
        unique_values = df[column].unique()
        unique_count = len(unique_values)

        # NaN or None percentages
        nan_count = df[column].isnull().sum()
        nan_percentage = df[column].isnull().mean() * 100
        
        text = f"{yellow('NaN/None values:')} {green(f'{nan_percentage:.2f}% | {nan_count} of {count}')}"
        text += f"  {yellow('Type:')} {green(df[column].dtype.name)}\n"

        # If numeric, show distribution summary
        text += f"{yellow('Unique values:')} {green(unique_count)}\n"

        if len(unique_values) <= 50:
            text += white('"' + '", "'.join(unique_values.astype(str)) + '"\n')
        
        if pd.api.types.is_numeric_dtype(df[column]):
            text += yellow('Value distribution summary:\n')
            for stat, value in df[column].describe().to_frame().T.iloc[0].items():
                text += f"{yellow(f'{stat}:')} {green(f'{value:.2f}\t')}"
        
        panels.append(Panel(text, title=f'Feature: {column}', width=column_width, border_style="bright_cyan"))
    
    # Create a Columns object with the panels, specifying 2 columns
    columns = Columns(panels, expand=True)
    console.print(columns)

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

def impute_feature(df, target_feature, features_to_use):
    """
    Imputes missing values in the 'LotFrontage' column using multivariate imputation.

    Parameters:
    - df: DataFrame containing the dataset.
    - features_to_use: List of column names to be used for imputation.
    
    Returns:
    - DataFrame with 'LotFrontage' imputed.
    """
    # Ensure 'LotFrontage' is included in the features to use
    if target_feature not in features_to_use:
        features_to_use.append(target_feature)
    
    # Filter the DataFrame to include only the specified features
    df_filtered = df[features_to_use]
    
    # Identify categorical features for encoding
    categorical_features = df_filtered.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Define a column transformer for OneHotEncoding categorical variables
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)], remainder='passthrough')
    
    # Define the imputer model using RandomForestRegressor
    imputer = IterativeImputer(estimator=RandomForestRegressor(n_estimators=10, random_state=42),
                               max_iter=10, random_state=42)
    
    # Create a pipeline with preprocessing and imputation
    impute_pipeline = make_pipeline(preprocessor, imputer)
    
    # Perform imputation
    df_imputed = impute_pipeline.fit_transform(df_filtered)
    df_imputed = pd.DataFrame(df_imputed, columns=preprocessor.transformers_[0][1].get_feature_names_out().tolist() + features_to_use)
    
    # The imputed 'LotFrontage' will be among the last columns after OneHotEncoding. Locate and return it.
    # Note: Adjust the column name extraction based on actual DataFrame structure post-imputation
    target_imputed = df_imputed.filter(like=target_feature).iloc[:, 0]  # Adjust based on actual column positioning
    
    # Insert the imputed 'LotFrontage' values back into the original DataFrame
    df[target_feature] = target_imputed.values
    
    return df

# Example usage:
# df_imputed = impute_lot_frontage(df, ['LotArea', 'Condition1', 'RoofMatl', 'LotConfig', 'GarageArea'])
# print(df_imputed[['LotFrontage', 'LotFrontage_Imputed']])



def add_one_hot_features(source: pd.DataFrame, target: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature in source.columns:
        encoded_features = OneHotEncoder(handle_unknown='ignore',sparse_output=False)\
            .set_output(transform='pandas')\
            .fit_transform(source[[feature]])
        target = pd.concat([target, encoded_features], axis=1)
    return target

def normalize_feature_MinMax(source: pd.DataFrame, target: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature in source.columns:
        scaler = MinMaxScaler( ).set_output(transform='pandas')
        encoded_feature = scaler.fit_transform(source[[feature]])
        target = pd.concat([target, encoded_feature], axis=1)
    return target

def normalize_feature_Standard(source: pd.DataFrame, target: pd.DataFrame, feature: str) -> pd.DataFrame:
    if feature in source.columns:
        scaler = StandardScaler( ).set_output(transform='pandas')
        encoded_feature = scaler.fit_transform(source[[feature]])
        target = pd.concat([target, encoded_feature], axis=1)
    return target

def add_age_features(source: pd.DataFrame, target: pd.DataFrame,  feature: str, range: int) -> pd.DataFrame:
    
    if feature not in source.columns:
        return target
    
    current_year = datetime.datetime.now().year
    target[feature + '_age'] = current_year - source[feature]
    
    new_feature_name = f'{feature}_{range}_category'

    target[new_feature_name ] = target[feature + '_age'] // range
    target = Features.add_one_hot_features(target, target, new_feature_name)
    target.drop(new_feature_name , axis=1, inplace=True)
    
    target.drop(feature + '_age', axis=1, inplace=True)
    
    return target

def color(text, color):
    return f"[{color}]{text}[/]"

def white(text):
    return color(text, "bright_white")

def green(text):
    return color(text, "bright_green")

def yellow(text):
    return color(text, "bright_yellow")

def red(text):
    return color(text, "bright_red")


def plot_data(X: pd.DataFrame, Y: pd.DataFrame) -> None:
    
    for column in X.columns:

        if X[column].std() != 0:
            # Calculate Pearson correlation
            r = X[column].corr(Y)
            print(f"Pearson Correlation {yellow(column)}: {green(f'{r:.2f}')}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Scatter plot
        ax1.scatter(X[column], Y)
        ax1.set_xlabel(column)
        ax1.set_ylabel(Y.name)
        ax1.set_title(f"Scatter plot of {column} vs {Y.name}")

        # Histogram
        ax2.hist(X[column])
        ax2.set_xlabel(column)
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"Histogram of {column}")

        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    df = pd.read_csv('./data/train.csv')
    analyze_features(df)
    #analysis_features_missing_data(df)
    df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType'], axis=1, inplace=True)   
    #analysis_features_missing_data(df)
    analyze_impact_features(df,'LotFrontage')
    
    
            
    # Drop columns with high percentage of missing values


    

    #analyze_impact_features(df, 'LotFrontage')

