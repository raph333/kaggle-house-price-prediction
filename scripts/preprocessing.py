#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:01:44 2018

@author: raph
"""


import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def impute_LotFrontage(df):
    fitting_data = df[['1stFlrSF', 'LotFrontage']].dropna()
    X = fitting_data[['1stFlrSF']].values
    y = fitting_data[['LotFrontage']].values
    
    lm = linear_model.LinearRegression()
    lm.fit(X, y)
    
    prediction = lm.predict(df[['1stFlrSF']].values)
    prediction= [x for sub in prediction for x in sub]  # flatten nested list
    
    new_df = df.copy()
    new_df['LF_prediction'] = prediction
    new_df.LotFrontage = new_df.apply(
            lambda x: (x.LF_prediction if np.isnan(x.LotFrontage) else
                       x.LotFrontage),
            axis=1
            )
    return new_df.drop('LF_prediction', axis=1)


def impute_GarageYrBlt(df):
    new_df = df.copy()
    new_df.GarageYrBlt = new_df.apply(
            lambda x: x.YearBuilt if np.isnan(x.GarageYrBlt) else x.GarageYrBlt,
            axis=1
            )
    return new_df


def impute_MasVnrArea(df):
    new_df = df.copy()
    new_df.MasVnrArea = new_df.MasVnrArea.map(
            lambda x: np.mean(df.MasVnrArea) if np.isnan(x) else x
            )
    return new_df


def preprocess_data_frame(df):
    numeric_f = [x for x in df.columns if np.issubdtype(df[x], np.number) and
                 x not in ('Id', 'SalePrice')
                 ]
    new_df = df.copy()
    new_df[numeric_f] = new_df[numeric_f].apply(scale_feature, axis=0)
    
    num_missing = df.isnull().sum()
    too_many_missing = list(num_missing[num_missing > (len(df) * 0.3)].index)
    new_df = new_df.drop(too_many_missing, axis=1)
    
    # impute missing values in numeric features:
    new_df = impute_GarageYrBlt(new_df)
    new_df = impute_MasVnrArea(new_df)
    new_df = impute_LotFrontage(new_df)
    
    new_df = new_df.dropna()  # drop data points with missing values
    
    non_numeric_f = [x for x in new_df.columns if x not in
                     [*numeric_f, 'Id', 'SalePrice']
                     ]
    new_df, dummy_var_features = make_dummy_variables(new_df, non_numeric_f)
        
    #X = np.array(new_df[(numeric_f + dummy_var_features)])
    #if test_set:
    #    return new_df, X

    #y = np.array(new_df[['SalePrice']])
    return new_df


def train_test_set_preprocessing(train_df, test_df, targe_var='SalePrice'):
    not_in_train = [x for x in test_df.columns if x not in train_df.columns]
    not_in_test = [x for x in train_df.columns if x not in test_df.columns and
                   x != targe_var]

    add_to_train = dict([(x, [0] * len(train_df)) for x in not_in_train])
    add_to_test = dict([(x, [0] * len(test_df)) for x in not_in_test])
    
    train_df = train_df.assign(**add_to_train)
    test_df = test_df.assign(**add_to_test)
    
    column_diff = set(train_df.columns).symmetric_difference(test_df.columns)
    assert len(column_diff) == 1 and column_diff.pop() == targe_var
    
    feature_order = sorted(test_df.columns)
    X = np.array(train_df[feature_order])
    y = np.array(train_df[[targe_var]])
    X_test = np.array(test_df[feature_order])
    
    return X, y, X_test
    
    



    




def scale_feature(feature: pd.Series, scale_by='std') -> pd.Series:
    """
    Scale numeric feature into smaller range (approximately [-1, 1])
    and center around 0
    """
    assert np.issubdtype(feature.dtype, np.number), 'need numeric feature'
    
    if scale_by == 'std':
        return (feature - np.mean(feature)) / np.std(feature)
    elif scale_by == 'range':
        return (feature - np.mean(feature)) / max(feature) - min(feature)


def make_dummy_variables(df: pd.DataFrame, feature_names: list) ->pd.DataFrame:
    """
    input:
    feat
    'A'
    'B'
    'C'
    
    return:
    'A'  'B'  'C'
    1     0    0
    0     1    0
    0     0    1
    """
    new_df = df.copy()
    dummy_variables = []  # just to keep track of the newly created features
    for feature in feature_names:
        for level in set(df[feature].dropna()):  # for all not-NULL levels
            dummy_var_name = f'{feature}_{level}'
            new_df[dummy_var_name] = df[feature].map(
                    lambda x: {level: 1, np.nan:np.nan}.get(x, 0)
                    )
            dummy_variables.append(dummy_var_name)
        new_df = new_df.drop(feature, axis=1)  # drop old feature
    return new_df, dummy_variables


