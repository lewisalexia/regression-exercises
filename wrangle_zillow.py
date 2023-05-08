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

# Scaling
import sklearn.preprocessing

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
    """This function is meant to clean and return the prepared df with
    encoded variables - ready for scaling/modeling.
    """
    print(f"Returning Zillow's Single Family Residential Homes from 2017")
    
    # rename columns
    df = df.rename(columns = {'bedroomcnt':'bed', 'bathroomcnt':'bath', 'calculatedfinishedsquarefeet':\
    'sqft', 'taxvaluedollarcnt': 'assessed_worth', 'yearbuilt':'year', 'taxamount':'property_taxes',\
    'propertylandusetypeid':'use', 'fips':'county'})
    print(f"--------------------------------------------")
    print(f"Renamed columns for ease of use")

    # drop all nulls
    df_clean = df.dropna()
    print(f"NaN's removed - Percent Original Data Remaining: {round(df_clean.shape[0]/df.shape[0]*100,0)}")

    # drop parcelid and use (used for initial exploration only)
    df_clean = df_clean.drop(columns=['parcelid', 'use'])

    # move target column to index position 0
    df_clean.insert(0, 'assessed_worth', df_clean.pop('assessed_worth'))
    print(f"Moved target column to index 0 for ease of assignment")

    # change data types and map FIPS code
    df_clean.county = df_clean.county.map({6037:"LA", 6059:"Orange", 6111:"Ventura"})
    df_clean.bed = df_clean.bed.astype(int)
    df_clean.year = df_clean.year.astype(int)
    print(f"Bed and year data types changed from float to integer\nChanged FIPS code to actual county name")

    # outliers
    df_clean = df_clean [df_clean.sqft < 25_000]
    df_clean = df_clean [df_clean.assessed_worth < df_clean.assessed_worth.quantile(.95)].copy()
    print(f"Outliers removed from Sqft < 25,000 and Assessed Worth > 95th quantile")

    # encode / get dummies
    dummy_df = pd.get_dummies(df_clean[['county']], dummy_na=False, drop_first=[True])

    # clean up and return final product
    df_clean = pd.concat([df_clean, dummy_df], axis=1).drop(columns=['county'])

    return df_clean

# -------------------------------------------------------------------------

# TRAIN, VALIDATE, TEST SPLIT

def split_zillow(df):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.

    (train, validate, test = split_zillow() to assign variable and return shape of df.)
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate,
                                       test_size=.25,
                                       random_state=123)
    
    print(f'Prepared DF: {df.shape}')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test

# -------------------------------------------------------------------------

# X_train, y_train, X_validate, y_validate, X_test, y_test

def x_y_train_validate_test(train, validate, test, target):
    """This function takes in the train, validate, and test dataframes and assigns 
    the chosen features to X_train, X_validate, X_test, and y_train, y_validate, 
    and y_test.
    """

    # X_train, validate, and test to be used for modeling
    X_train = train[1:]
    X_validate = validate[1:]
    X_test = test[1:]
    y_train = train[{target}]
    y_validate = validate[{target}]
    y_test = test[{target}]
    
    print(f"Verifying number of features and target:")
    print(f'Train: {X_train.shape[1], y_train.shape[1]}')
    print(f'Validate: {X_validate.shape[1], y_validate.shape[1]}')
    print(f'Test: {X_test.shape[1], y_test.shape[1]}')

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# -------------------------------------------------------------------------

# SCALING

def scale_zillow(X_train, X_validate, X_test):
    """This function is built to take in train, validate, and test dataframes
    and scale them returning a visual of the before and after.
    """

    # make the scaler
    robustscaler = sklearn.preprocessing.RobustScaler()

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.

    # fit the scaler on train ONLY
    robustscaler.fit(X_train)

    # use the scaler
    X_train_scaled_ro = robustscaler.transform(X_train)
    X_validate_scaled_ro = robustscaler.transform(X_validate)
    X_test_scaled_ro = robustscaler.transform(X_test)

    # visualize
    plt.figure(figsize=(13, 8))

    ax = plt.subplot(321)
    plt.hist(X_train, bins=25, ec='black')
    plt.title('Original Train')
    ax = plt.subplot(322)
    plt.hist(X_train_scaled_ro, bins=25, ec='black')
    plt.title('Scaled Train')

    ax = plt.subplot(323)
    plt.hist(X_validate, bins=25, ec='black')
    plt.title('Original Validate')
    ax = plt.subplot(324)
    plt.hist(X_validate_scaled_ro, bins=25, ec='black')
    plt.title('Scaled Validate')

    ax = plt.subplot(325)
    plt.hist(X_test, bins=25, ec='black')
    plt.title('Original Test')
    ax = plt.subplot(326)
    plt.hist(X_test_scaled_ro, bins=25, ec= 'black')
    plt.title('Scaled Test')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.show()