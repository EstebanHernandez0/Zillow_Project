import pandas as pd
import numpy as np
import os
from env import get_connection
from sklearn.model_selection import train_test_split


def get_zillow():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT yearbuilt, taxvaluedollarcnt, taxamount, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet ,fips FROM properties_2017 JOIN propertylandusetype USING(propertylandusetypeid) WHERE propertylandusetypeid = 261; ', get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 
    
def prep_zillow(df):
    '''
    This function takes in the zillow df
    then the data is cleaned and returned
    '''
    #change column names to be more readable
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqr_feet',
                          'taxvaluedollarcnt':'property_value', 
                          'taxamount': 'tax_amount',
                          'yearbuilt':'year_built'})

    #drop null values- at most there were 9000 nulls (this is only 0.5% of 2.1M)
    df = df.dropna()

    #drop duplicates
    df.drop_duplicates(inplace=True)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


def wrangle_zillow():
    '''
    Uses both acquire, and prepare at the same time
    and returns the split data
    '''
    
    train, val, test = prep_zillow(get_zillow())
    
    return train, val, test