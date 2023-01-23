 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def prep_zillow(df):
    ''' Prepare zillow data for exploration by taking in a dataframe and returns train, validate, test'''
     
    df = df.rename(columns={'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'sqr_feet',
                          'taxvaluedollarcnt':'property_value', 
                          'taxamount': 'tax_amount',
                          'yearbuilt':'year_built'})

    # removing outliers to make the data easier to use and more trustworthy
    df = drop_outers(df, 1.5, ['bedrooms', 'bathrooms', 'sqr_feet', 'property_value',])
    
    
    # drop null values that are left in the lot size data since there are so few
    df=df.dropna()
    
    df.drop(columns= ['Unnamed: 0'], inplace= True)
    
    # converting column datatypes 
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # train/validate/test split and is reproducible due to random_state = 123
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, val = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using median from train data set and then applied to the validate and test set as well
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    val[['year_built']] = imputer.transform(val[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])  
    
    
    
    print('Train shape:' ), print(train.shape)
    
    print('Validate shape:' ), print(val.shape)
    
    print('Test shape:' ), print(test.shape)
    
    return train, val, test    



def drop_outers(df, o, col_list):
    ''' 
    drops the outliers and returns new cleaned dataframe
    '''
    
    for col in col_list:
        # here we get the quartiles 
        quant1, quant3= df[col].quantile([.25, .75])  
        
        # find the IQR
        IQR= quant3 - quant1   
        
        # find the opposite of lower bound
        upper= quant3 + o * IQR   
        
        # find the opposite of upper bound
        lower= quant1 - o * IQR

        # return new dataframe 
        
        df= df[(df[col] > lower) & (df[col] < upper)]
        
    return df


