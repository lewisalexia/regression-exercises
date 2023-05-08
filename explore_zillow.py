
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

