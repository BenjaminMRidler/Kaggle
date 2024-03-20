import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam, Ftrl

import dataset as ds
from dataset import DataSet
from features import Features

class HousingData(DataSet):
    def __init__(self, file_name: str) -> None:
        super().__init__(file_name)
        self.build_features()
        self.split_data()
        self.make_dependent_datasets()

    def build_features(self) -> None:
        
        f = Features()

        features = pd.DataFrame()
        features = pd.concat([features, self.source['SalePrice']], axis=1)
            
        #features = add_one_hot_features(source, features, 'MSSubClass')
        features = f.add_one_hot_features(self.source, features, 'MSZoning')
        #features = normalize_feature_Standard(source, features, 'LotFrontage')
        features = f.normalize_feature_Standard(self.source, features, 'LotArea')
        #features = add_one_hot_features(source, features, 'Street')
        #features = add_one_hot_features(source, features, 'Alley')
        #features = add_one_hot_features(source, features, 'LotShape')
        #features = add_one_hot_features(source, features, 'LandContour')
        #features = add_one_hot_features(source, features, 'Utilities')
        #features = add_one_hot_features(source, features, 'LotConfig')
        #features = add_one_hot_features(source, features, 'LandSlope')
        features = f.add_one_hot_features(self.source, features, 'Neighborhood')
        #features = add_one_hot_features(source, features, 'Condition1')
        #features = add_one_hot_features(source, features, 'Condition2')
        features = f.add_one_hot_features(self.source, features, 'BldgType')
        #features = add_one_hot_features(source, features, 'HouseStyle')
        #features = normalize_feature(source, features, 'OverallQual')
        
        features = f.add_one_hot_features(self.source, features, 'OverallCond')
        #features = add_one_hot_features(source, features, 'OverallCond')
        
        #features = add_age_features(source, features, 'YearBuilt')
        features = f.add_age_features(self.source, features, 'YearRemodAdd', 10)
        #features = add_one_hot_features(source, features, 'RoofStyle')
        #features = add_one_hot_features(source, features, 'RoofMatl')
        #features = add_one_hot_features(source, features, 'Exterior1st')
        #features = add_one_hot_features(source, features, 'Exterior2nd')
        # features = add_one_hot_features(source, features, 'MasVnrType')
        # features = normalize_feature(source, features, 'MasVnrArea')
        #features = ordinal_encode_normalize(source, features, 'ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

        features = f.add_one_hot_features(self.source, features, 'ExterCond')
        #features = ordinal_encode_normalize(source, features, 'ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

        features = f.add_one_hot_features(self.source, features, 'Foundation')
        #features = ordinal_encode_normalize(source, features, 'BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
        #features = ordinal_encode_normalize(source, features, 'BsmtCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
        #features = ordinal_encode_normalize(source, features, 'BsmtExposure', ['No', 'Mn', 'Av', 'Gd'])
        #features = ordinal_encode_normalize(source, features, 'BsmtFinType1', ['No', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
        #features = normalize_feature(source, features, 'BsmtFinSF1')
        features = f.add_one_hot_features(self.source, features, 'BsmtFinType2')
        #features = normalize_feature(source, features, 'BsmtFinSF2')
        #features = normalize_feature(source, features, 'BsmtUnfSF')
        features = f.normalize_feature_Standard(self.source, features, 'TotalBsmtSF')
        #features = add_one_hot_features(source, features, 'Heating')
        #features = ordinal_encode_normalize(source, features, 'HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
        
        features = f.add_one_hot_features(self.source, features, 'CentralAir')
        #features = ordinal_encode_normalize(source, features, 'CentralAir', ['N', 'Y'])
        
        #features = add_one_hot_features(source, features, 'Electrical')
        #features = normalize_feature(source, features, '1stFlrSF')
        #features = normalize_feature(source, features, '2ndFlrSF')
        #features = normalize_feature(source, features, 'LowQualFinSF')
        features = f.normalize_feature_Standard(self.source, features, 'GrLivArea')
        #features = normalize_feature(source, features, 'BsmtFullBath')
        #features = normalize_feature(source, features, 'BsmtHalfBath')
        features = f.add_one_hot_features(self.source, features, 'FullBath')
        #features = add_one_hot_features(source, features, 'HalfBath')
        features = f.add_one_hot_features(self.source, features, 'BedroomsAbvGr')
        #features = add_one_hot_features(source, features, 'KitchenAbvGr')
        
        features = f.add_one_hot_features(self.source, features, 'KitchenQual')
        #features = ordinal_encode_normalize(source, features, 'KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
        
        features = f.add_one_hot_features(self.source, features, 'TotRmsAbvGrd')
        #features = ordinal_encode_normalize(source, features, 'Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'])
        #features = add_one_hot_features(source, features, 'Fireplaces')
        #features = ordinal_encode_normalize(source, features, 'FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])    
        #features = add_one_hot_features(source, features, 'GarageType')
        #features = add_age_features(source, features, 'GarageYrBlt')
        #features = ordinal_encode_normalize(source, features, 'GarageFinish', ['Unf', 'RFn', 'Fin'])   
        
        features = f.add_one_hot_features(self.source, features, 'GarageCars')
        
        #features = normalize_feature(source, features, 'GarageArea')
        #features = ordinal_encode_normalize(source, features, 'GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])

        #features = add_one_hot_features(source, features, 'GarageCond')
        
        #features = add_one_hot_features(source, features, 'PavedDrive')
        #features = ordinal_encode_normalize(source, features, 'PavedDrive', ['N', 'P', 'Y'])
        
        #features = normalize_feature(source, features, 'WoodDeckSF')
        #features = normalize_feature(source, features, 'OpenPorchSF')
        #features = normalize_feature(source, features, 'EnclosedPorch')
        #features = normalize_feature(source, features, '3SsnPorch')
        #features = normalize_feature(source, features, 'ScreenPorch')
        #features = normalize_feature(source, features, 'PoolArea')
        
        #features = add_one_hot_features(source, features, 'PoolQC')
        #features = ordinal_encode_normalize(source, features, 'PoolQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex'])
        
        #features = ordinal_encode_normalize(source, features, 'Fence', ['MnWw', 'GdWo', 'MnPrv', 'GdPrv'])
        #features = add_one_hot_features(source, features, 'MiscFeature')
        #features = normalize_feature(source, features, 'MiscVal')
        features = f.add_one_hot_features(self.source, features, 'MoSold')
        
        #features = add_age_features(source, features, 'YrSold', 5)
        
        #features = add_one_hot_features(source, features, 'SaleType')
        #features = add_one_hot_features(source, features, 'SaleCondition')

        return features

class Model:
    def __init__(self, train_x: pd.DataFrame, train_y: pd.DataFrame, validation_x: pd.DataFrame, validation_y: pd.DataFrame):
        self.train_x = train_x
        self.train_y = train_y
        self.validation_x = validation_x
        self.validation_y = validation_y
        self.input_shape = (train_x.shape[1],)
        self.build_regression_model()

    def build_regression_model(self):
        model = keras.Sequential([
            layers.InputLayer(input_shape=self.input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=.025), loss='mean_squared_error',metrics=['mae'])

        self.model = model

    def fit(self):
        self.model.fit(self.train_features, self.train_dependent, epochs=50, batch_size=100, verbose=1,\
            validation_data=(self.verification_features, self.verification_dependent))

    def summary(self):
        if self.model is not None:
            self.model.summary()


# Verify the shapes of the training and verification sets
if __name__ == "__main__":
    housingData = HousingData('train.csv')  # Load the data
    model = Model(housingData.train_x, housingData.train_y, housingData.validation_x, housingData.validation_y)
    plt.    
    
    #model.fit()
    

