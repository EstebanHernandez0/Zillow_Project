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
        df = pd.read_sql('SELECT yearbuilt, taxvaluedollarcnt, taxamount, bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, fips, regionidcounty FROM properties_2017 JOIN propertylandusetype USING(propertylandusetypeid) WHERE propertylandusetypeid =261' , get_connection('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df 
    
