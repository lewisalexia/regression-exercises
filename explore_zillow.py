
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

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------

# Visualizations

def hist_zillow(df):
    """This function display histograms for all columns"""
    plt.figure(figsize=(12,6))
    for i, col in enumerate(df.columns):
        plot_number = i + 1
        plt.subplot(2, 5, plot_number)
        plt.hist(df[col])
        plt.title(f"{col}")
    plt.show()


def visual_explore_univariate(df):
    """This function takes in a DF and explores each variable visually
    as well as the value_counts to identify outliers.
    Works for numerical."""
    for col in df.columns[:-3]:
        print(col)
        sns.boxplot(data=df, x=col)
        plt.show()
        print(df[col].value_counts().sort_index())
        print() 

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
