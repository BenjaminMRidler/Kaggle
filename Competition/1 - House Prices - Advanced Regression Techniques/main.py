import lib.utility as util
import lib.data as data
from rich import print
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from lib.utility import yellow, green, red, cyan, white

def main():
    # Load the data
    df = data.load_data('data/train.csv')
    Console().clear()
    
    ### Remove columns with significant missing data
    print(yellow('Remove columns with significant missing data'))
    columns_to_fix_series = data.analyze_missing_data(df)
    
    remove_columns = []
    threshold = .5
    for column in columns_to_fix_series.index:
        if columns_to_fix_series[column] > threshold:
            remove_columns.append(column)
    
    df = data.remove_columns(df, remove_columns)
    
    ### Verify that the columns have been removed. 
    print(yellow('Verify removal'))
    columns_to_fix_series = data.analyze_missing_data(df)

    ### Create a dataset with columns can be used for analysis and imputing
    temp_normalized_df = data.prepare_data(df)
    columns_to_fix_series = data.analyze_missing_data(df)
    ### Fix the remaining columns that are missing data
    columns_to_fix_df = df[columns_to_fix_series.index]
    
    ### iterate through each column-to-be-fixed and analyze the impactful features
    for impute_target in columns_to_fix_df.columns:
        
        impactful_features_df, _ = data.make_impactful_feature_model(impute_target, df, temp_normalized_df)
        text = f"{red(f'Impact Ranking: {impactful_features_df}\n')}"
        
        impactful_features_df = impactful_features_df[impactful_features_df['Importance'] > 0.05]
        impactful_features_list = impactful_features_df['Feature'].tolist()
        
        text += f"{red(f'Selected columns: {impactful_features_df}\n')}"
        text += f"Before Imputation: {df[impute_target].head().transpose()}\n"
        df, temp_normalized_df = data.impute_missing_data(impute_target, df, temp_normalized_df, impactful_features_list)
        text += f"After Imputation: {df[impute_target].head().transpose()}\n"
        
        print(Panel(text, title=f'Column: {impute_target}', border_style="bright_cyan"))
        
    pass
        
    
    
if __name__ == "__main__":
    main()  
