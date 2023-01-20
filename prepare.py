import pandas as pd
import numpy as np
import os
from env import get_connection
from sklearn.model_selection import train_test_split
import functions as f
from sklearn.impute import SimpleImputer

def prep_zillow(df):
    ''' Prepare zillow data for exploration by taking in a dataframe and returns train, validate, test'''
     
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqr_feet',
                          'taxvaluedollarcnt':'property_value', 
                          'taxamount': 'tax_amount',
                          'yearbuilt':'year_built'})

    # removing outliers
    df = f.remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'sqr_feet', 'property_value',])
    

    
    # drop null values that are left in the lot size data since there are so few
    df=df.dropna()
    
    # converting column datatypes 
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using median from train data set and then applied to the validate and test set as well
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test    

