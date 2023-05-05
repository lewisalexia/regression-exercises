# Wrangle module to pull in Zillow db from MySQL

# My Modules
import explore as ex
import stats_conclude as sc

# Imports
import env
import os

# Numbers
import pandas as pd 
import numpy as np
from scipy import stats

# Vizzes
import matplotlib.pyplot as plt
import seaborn as sns

# Splits
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------------------

# ACQUIRE

def get_connection(db):
    """This functions grants access to server with eny credentials"""
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('CSV file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('Creating df and exporting CSV...')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df 
print(f'Load in successful, awaiting commands...')

def get_zillow_261():
    url = env.get_connection('zillow')
    query = ''' SELECT bedroomcnt,
		bathroomcnt,
		calculatedfinishedsquarefeet,
        taxvaluedollarcnt,
        yearbuilt,
        taxamount,
        fips,
        propertylandusetypeid,
        parcelid
        FROM properties_2017
        WHERE propertylandusetypeid like 261;
        '''
    filename = 'zillow_261.csv'
    df = check_file_exists(filename, query, url)

    return df

# -------------------------------------------------------------------------

# PREPARE WAS DONE MANUALLY THEN INPUTTED INTO THE FUNCTION BELOW

# TOTAL WRANGLE FUNCTION

def wrangle_zillow(df):
    """This function is meant to clean and return the prepared df.
    """
    # read in or create the CSV
    # df = get_zillow_261()
    # print(f"Returning Zillow's Single Family Residential Homes from 2017")
    
    # rename columns
    df.rename(columns = {'bedroomcnt':'bed', 'bathroomcnt':'bath', 'calculatedfinishedsquarefeet':\
    'sqft', 'taxvaluedollarcnt': 'assessed_worth', 'yearbuilt':'year', 'taxamount':'property_taxes',\
    'propertylandusetypeid':'use', 'fips':'county'}, inplace = True)
    print(f"--------------------------------------------")
    print(f"Renamed columns for ease of use")

    # drop all nulls
    df = df.dropna()
    print(f"NaN's removed - 99% of data remains")

    # change data types for fips and use id
    df.year = df.year.astype(int)
    df.use = df.use.astype(int)
    print(f"Year and Use data types changed from float to integer\n")

    # outliers
    df = df [df.sqft < 25_000]
    df = df [df.assessed_worth < df.assessed_worth.quantile(.95)].copy()
    print(f"Outliers removed from Sqft - <25,000 and Assessed Worth > 95th quantile")

    return df

# -------------------------------------------------------------------------

# TRAIN, VALIDATE, TEST SPLIT

def split_zillow(df, target_variable): ## B R O K E N!!
    '''
    Takes in a dataframe and return train, validate, test subset dataframes
    '''
    train, test = train_test_split(df, #first split
                                   test_size=.2, 
                                   random_state=123, 
                                   stratify=df[target_variable])
    train, validate = train_test_split(train, #second split
                                       test_size=.25, 
                                       random_state=123, 
                                       stratify=train[target_variable])
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')

    return train, validate, test



