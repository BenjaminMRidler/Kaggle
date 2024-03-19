# Load the CSV file into a pandas DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.source = self.load_data()    
        
    def analyze_features(df):
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
            for column in df.columns:
                # NaN or None percentages
                nan_percentage = df[column].isnull().mean() * 100
                
                print(f"Feature: {column}")
                print(f"Percentage of NaN/None values: {nan_percentage:.2f}%")
                
                # If numeric, show distribution summary
                if pd.api.types.is_numeric_dtype(df[column]):
                    print("Value distribution summary:")
                    print(df[column].describe())
                else:
                    # For categorical data, if fewer than 10 unique values, show them
                    unique_values = df[column].unique()
                    if len(unique_values) <= 10:
                        print("Unique values:")
                        print(unique_values)
                print("-" * 40)
        
    def compare_sets(df_1: pd.DataFrame, df_2: pd.DataFrame) -> None:
        # Compare the features in the training and verification sets
        df_1_features = set(df_1.columns)
        df_2_features = set(df_2.columns)
        features_only_in_df_1 = df_1_features - df_2_features
        features_only_in_df_2 = df_2_features - df_1_features
        print("Features only in DataFrame 1:")
        print(features_only_in_df_1)
        print("Features only in DataFrame 2:")
        print(features_only_in_df_2)
        
    def load_data(self) -> pd.DataFrame:
        # Load the data from the 'train.csv' file
        df = pd.read_csv('train.csv')
        return df
    
    def make_dependent_datasets(self) -> None:
        self.train_Y = self.train_X[['SalePrice']]
        self.verification_Y = self.verification_X[['SalePrice']]
        self.train_X.drop('SalePrice', axis=1, inplace=True)  # Remove the dependent column from the training features
        self.verification_X.drop('SalePrice', axis=1, inplace=True)  # Remove the dependent column from the verification features

    def split_data(self, test_size: int = 0.25) -> None:
        # Split the data into training and verification sets
        self.train_X, self.verification_X = train_test_split(self.source, test_size=0.25)
        self.make_dependent_datasets(self.train_X, self.verification_X)

    def print_summary(self) -> None:
        
        if(self.train_X is None or self.verification_X is None):
            print("The data has not been split yet. Please run the split_data method first.")
            return  
        
        # Print the shapes of the training and verification sets
        print("Training set shape:", self.train_X.shape)
        print("Verification set shape:", self.verification_X.shape)

