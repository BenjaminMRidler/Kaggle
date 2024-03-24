1. load data : load_data(file_name) -> df

2. analyze missing data () : analyze_missing_data(df) -> None
    - display columns with missing data

3. remove columns with large percentage of missing data : remove_columns(df, columns) -> df

4. analyze data : analyze_data(df) -> None
    - data type
    - data distribution
    - missing data
    - discrete values

5. prepare_data : prepare_data(df) -> prepared_df
    - map categorical data to numeric
    - impute missing data to mean
    - scale data to 0-1
    - map outliers to upper/lower

loop (df.columns):
    6. analyse Impactful Features using "Feature Importance using Tree-based Models" : analyze_impactful_features(df, prepared_df, feature) -> impactful_feature_df, impactful_features_list
        - replace feature in prepared_df with original feature from df
        - analyze feature impact and report 
        * return impactful_feature_df

    7. impute missing data : impute_missing_data(impactful_feature_df, dependent_var, impactful_features_list) -> impactful_feature_df
        - impute data using Random Forest Imputation
        * return impactful_feature_df with feature populated with imputed data

7. Repeat for all columns with missing data

8. analyse Impactful Features on dependent var using "Feature Importance using Tree-based Models"
    - Scale dependent variable to 0-1
    - rank impactful features and produce a DataFrame
    - run DataFram through models
    - report results

