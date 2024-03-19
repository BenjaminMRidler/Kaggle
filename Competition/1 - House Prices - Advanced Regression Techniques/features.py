import datetime
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List

class Features():

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

if __name__ == "__main__":
    ...


