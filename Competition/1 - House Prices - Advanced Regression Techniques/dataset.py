# Load the CSV file into a pandas DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.source = self.load_data()    
        
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
    
    def plot()
    
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

