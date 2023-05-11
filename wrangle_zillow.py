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

def outlier(df, feature, m):
    '''
    outlier will take in a dataframe's feature:
    - calculate it's 1st & 3rd quartiles,
    - use their difference to calculate the IQR
    - then apply to calculate upper and lower bounds
    - using the `m` multiplier
    '''
    q1 = df[feature].quantile(.25)
    q3 = df[feature].quantile(.75)
    
    iqr = q3 - q1
    
    multiplier = m
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    
    return upper_bound, lower_bound

def outliers_zillow(df,m):
    """This function uses a built-in outlier function to scientifically identify
    all outliers in the zillow dataset and then print them out for each column.
    """
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numerical varibles

    for col in df.columns:
        if col in df.select_dtypes(include=['int64', 'float64']):
            col_num.append(col)
        else:
            col_cat.append(col)

    for col in col_num:
        q1 = df[col].quantile(.25)
        q3 = df[col].quantile(.75)
        iqr = q3 - q1
        upper_bound = q3 + (m * iqr)
        lower_bound = q1 - (m * iqr)
        print(f"{col.capitalize().replace('_',' ')}: upper,lower ({upper_bound}, {lower_bound})")
    print(f"---")

    for col in col_cat:
        print(f"{col.capitalize().replace('_',' ')} is a categorical column.")

# SECOND RUN THROUGH OF EXPLORE

def wrangle_zillow(df):
    """This function is meant to clean and return the prepared df with
    encoded variables.
    ---
    This function is the second iteration of exploration on this dataset.
    ---
    Further shrinking the outliers to prevent skewing of the data for the 
    target audience.
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

    # outliers ACTUAL
    df_clean = df_clean [(df_clean.assessed_worth <= 1_054_062) & (df_clean.assessed_worth >= 0)]
    df_clean = df_clean [(df_clean.bed <= 6) & (df_clean.bed > 0)]
    df_clean = df_clean [(df_clean.bath <= 5) & (df_clean.bath > 0)]    
    df_clean = df_clean [(df_clean.sqft <= 5_000) & (df_clean.sqft > 0)]
    df_clean = df_clean [df_clean.year >= 1908]
    df_clean = df_clean [(df_clean.property_taxes <= 12_000) & (df_clean.property_taxes > 0)]
    print(f"Outliers removed: Percent Original Data Remaining: {round(df_clean.shape[0]/df.shape[0]*100,0)}\n Sqft <= 5,000 and > 0\n Property Taxes <= $12,233 and > 0\n Bathrooms <= 5 and > 0\n Bedrooms <= 6 and > 0\n Built after 1908\n Assessed Worth <= $1,054,062 and > 0")

    # encode / get dummies
    dummy_df = pd.get_dummies(df_clean[['county']], dummy_na=False, drop_first=True)
    print(f"Encoded County column and renamed encoded columns for readability")

    # clean up and return final product
    df_clean = pd.concat([df_clean, dummy_df], axis=1)
    df_clean = df_clean.rename(columns={'county_Orange':'orange','county_Ventura':'ventura'})
    df_clean = df_clean.drop(columns=['county'])
    
    print(f"DataFrame is clean and ready for exploration :)")

    return df_clean

# -------------------------------------------------------------------------

# TRAIN, VALIDATE, TEST SPLIT

def split_zillow(df):
    '''
    This function takes in a DataFrame and returns train, validate, and test DataFrames.

    (train, validate, test = split_zillow() to assign variable and return shape of df.)
    '''
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3,
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
    ---
    Format: X_train, y_train, X_validate, y_validate, X_test, y_test = function()
    """ 
    # X_train, validate, and test to be used for modeling
    X_train = train.drop(columns=target)
    y_train = train[{target}]

    X_validate = validate.drop(columns=target)
    y_validate = validate[{target}]
   
    X_test = test.drop(columns=target)
    y_test = test[{target}]

    print(f"Variable assignment successful...")

    # verifying number of features and target
    print(f"Verifying number of features and target:")
    print(f'Train: {X_train.shape, y_train.shape}')
    print(f'Validate: {X_validate.shape, y_validate.shape}')
    print(f'Test: {X_test.shape, y_test.shape}')

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# -------------------------------------------------------------------------

# SCALING

# def scale_zillow(X_train, X_validate, X_test):
#     """This function is built to take in train, validate, and test dataframes
#     and scale them returning a visual of the before and after. You do not need to
#     scale the target variable.

#     Returns scaled df's for train, validate, and test.

#     format = (X_train_scaled_ro, X_validate_scaled_ro, X_test_scaled_ro)
#     """

#     # make the scaler
#     robustscaler = sklearn.preprocessing.RobustScaler()

#     # Note that we only call .fit with the training data,
#     # but we use .transform to apply the scaling to all the data splits.

#     # fit the scaler on train ONLY
#     robustscaler.fit(X_train)

#     # use the scaler
#     X_train_scaled_ro = pd.DataFrame(robustscaler.transform(X_train))
#     X_validate_scaled_ro = pd.DataFrame(robustscaler.transform(X_validate))
#     X_test_scaled_ro = pd.DataFrame(robustscaler.transform(X_test))

#     # visualize
#     plt.figure(figsize=(13, 8))

#     ax = plt.subplot(321)
#     plt.hist(X_train, bins=50, ec='black')
#     plt.title('Original Train')
#     ax = plt.subplot(322)
#     plt.hist(X_train_scaled_ro, bins=50, ec='black')
#     plt.title('Scaled Train')

#     ax = plt.subplot(323)
#     plt.hist(X_validate, bins=50, ec='black')
#     plt.title('Original Validate')
#     ax = plt.subplot(324)
#     plt.hist(X_validate_scaled_ro, bins=50, ec='black')
#     plt.title('Scaled Validate')

#     ax = plt.subplot(325)
#     plt.hist(X_test, bins=50, ec='black')
#     plt.title('Original Test')
#     ax = plt.subplot(326)
#     plt.hist(X_test_scaled_ro, bins=50, ec= 'black')
#     plt.title('Scaled Test')
    
#     plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.9, 
#                     wspace=0.4, 
#                     hspace=0.4)
    
#     plt.show()

#     return X_train_scaled_ro, X_validate_scaled_ro, X_test_scaled_ro



# def inverse_robust(scaled_df):
#     """This function takes in the robustscaler object and returns the inverse
    
#     format to return original df = robustscaler_back = function()
#     """
#     robustscaler_back = pd.DataFrame(robustscaler.inverse_transform(scaled_df))

#     # visualize if you want it too
#     # plt.figure(figsize=(13, 6))
#     # plt.subplot(121)
#     # plt.hist(X_train_scaled_ro, bins=50, ec='black')
#     # plt.title('Scaled')
#     # plt.subplot(122)
#     # plt.hist(robustscaler_back, bins=50, ec='black')
#     # plt.title('Inverse')
#     # plt.show()

#     return robustscaler_back

def scale_zillow_2(X_train, X_validate, X_test):
    """This function is built to take in train, validate, and test dataframes
    and scale them returning a visual of the before and after.

    Returns scaled df's for train, validate, and test.

    Format: X_train_scaled_mm, X_validate_scaled_mm, X_test_scaled_mm = function()
    """
    # make the scaler
    minmaxscaler = sklearn.preprocessing.MinMaxScaler()

    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.

    # fit the scaler on train ONLY
    minmaxscaler.fit(X_train)

    # use the scaler
    X_train_scaled_mm = pd.DataFrame(minmaxscaler.transform(X_train))
    X_validate_scaled_mm = pd.DataFrame(minmaxscaler.transform(X_validate))
    X_test_scaled_mm = pd.DataFrame(minmaxscaler.transform(X_test))

    # visualize
    plt.figure(figsize=(13, 8))

    plt.subplot(321)
    plt.hist(X_train, bins=50, ec='black')
    plt.title('Original Train')
    plt.subplot(322)
    plt.hist(X_train_scaled_mm, bins=50, ec='black')
    plt.title('Scaled Train')

    plt.subplot(323)
    plt.hist(X_validate, bins=50, ec='black')
    plt.title('Original Validate')
    plt.subplot(324)
    plt.hist(X_validate_scaled_mm, bins=50, ec='black')
    plt.title('Scaled Validate')

    plt.subplot(325)
    plt.hist(X_test, bins=50, ec='black')
    plt.title('Original Test')
    plt.subplot(326)
    plt.hist(X_test_scaled_mm, bins=50, ec= 'black')
    plt.title('Scaled Test')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.show()

    return X_train_scaled_mm, X_validate_scaled_mm, X_test_scaled_mm



def inverse_minmax(scaled_df): # Need to connect with funtion above to work
    """This function takes in the MinMaxScaler object and returns the inverse
    
    format to return original df = minmaxscaler_back = function()
    """

    minmaxscaler_back = pd.DataFrame(minmaxscaler.inverse_transform(scaled_df))

    # visualize if you want it too
    # plt.figure(figsize=(13, 6))
    # plt.subplot(121)
    # plt.hist(X_train_scaled_ro, bins=50, ec='black')
    # plt.title('Scaled')
    # plt.subplot(122)
    # plt.hist(robustscaler_back, bins=50, ec='black')
    # plt.title('Inverse')
    # plt.show()

    return minmaxscaler_back