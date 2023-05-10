
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
        plt.tight_layout(pad=3.0)
        plot_number = i + 1
        plt.subplot(2, 4, plot_number)
        plt.hist(df[col])
        plt.title(f"{col}")   

    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.4, 
                hspace=0.4)
    
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

def plot_variable_pairs_120K(df):
    """This function takes in a df with a sample size of 120_000
    and returns a pairplot with regression line.
    """
    sns.pairplot(data=df.sample(120_000),\
                 kind='reg',corner=True, plot_kws={'line_kws':{'color':'red'}}\
                 , palette="Accent")
    plt.show()
    
def plot_variable_pairs(df):
    """This function takes in a df and returns a pairplot with regression line."""
    sns.pairplot(data=df,\
             kind='reg',corner=True, plot_kws={'line_kws':{'color':'red'}}\
             , palette="Accent")
    plt.show()

def plot_categorical_and_continuous_vars(df):
    """This function takes in a df and a hardcoded target variable to explore.
    
    This function is meant to assign the df columns to categorical and numerical 
    columns. The default for numerical is to be continuous (col_num). 
    
    Object types indicate "buckets" which indicates a categorical variable (col_cat).

    The function will then print for each col in col_cat:
        * Value Counts
        * Proportional size of data
        * Hypotheses (null + alternate)
        * Analysis and summary using CHI^2 test function (chi2_test from stats_conclude.py)
        * A conclusion statement
        * A graph representing findings

    The function will then print for each col in col_num:
        * A graph with two means compared to the target variable.
    """
    col_cat = [] #this is for my categorical varibles
    col_num = [] #this is for my numerical varibles
    target = 'assessed_worth' # assigning target variable
    
    # assign
    for col in df.columns:
        if col in df.select_dtypes(include=['number']):
            col_num.append(col)
        else:
            col_cat.append(col)
            
    # iterate through categorical
    for col in col_cat:
        print(f"Categorical Columns\n**{col.upper()}**")
        print(df[col].value_counts())
        print(round(df[col].value_counts(normalize=True)*100),2)
        print()
        print(f'HYPOTHESIZE')
        print(f"H_0: {col.lower().replace('_',' ')} does not affect {target}")
        print(f"H_a: {col.lower().replace('_',' ')} affects {target}")
        print()
        print('ANALYZE and SUMMARIZE')
        observed = pd.crosstab(df[col], df[target])
        α = 0.05
        chi2, pval, degf, expected = stats.chi2_contingency(observed)
        print(f'chi^2 = {chi2:.4f}')
        print(f'p-value = {pval} < {α}')
        print('----')
        if pval < α:
            print ('We reject the null hypothesis.')
        else:
            print ("We fail to reject the null hypothesis.")
            
        # visualize 1
        print()
        print(f'VISUALIZE')
        sns.barplot(x=df[col], y=df[target])
        plt.title(f"{col.lower().replace('_',' ')} vs {target}")
        plt.axhline(df[target].mean(), color='black')
        plt.show()
        print(f'\n')
        
        # visualize 2
        sns.histplot(data=df, x=col, y=target, bins=50)
        plt.title(f"distribution of {col.lower().replace('_',' ')} vs {target}")
        plt.axhline(df[target].mean(), color='black')
        plt.show()
        
        # visualize 3
        sns.boxenplot(data=df, x=col, y=target, palette='Accent')
        plt.title(f"boxenplot of {col.lower().replace('_',' ')} vs {target}")
        plt.axhline(df[target].mean(), color='black')
        plt.show()
        
    # looking at numericals
    print(f"Numerical Columns")
    
    # visualize 1
    # We already determined that all of the columns were normally distributed.
    # create the correlation matrix using pandas .corr() using pearson's method
    worth_corr = df.corr(method='pearson')
    sns.heatmap(worth_corr, cmap='PRGn', annot=True, mask=np.triu(worth_corr))
    plt.title(f"Assessed Worth Correlation Heatmap")
    plt.show()
    
    # iterate through numericals
    for col in col_num:
        
        # visualize 2
        sns.relplot(data=df, x=col, y=target, kind='scatter')
        plt.title(f"Is {target} independent of {col.lower().replace('_',' ')}?")
        pop_mn = df[col].mean()
        plt.axvline(pop_mn, label=(f"{col.lower().replace('_',' ')} mean"), color='red',\
                   linestyle='--')
        plt.axhline(df[target].mean(), label=(f"{target.lower().replace('_',' ')} mean"), color='black',\
           linestyle='--')
        plt.legend()
        plt.show()
        print()

        # visualize 3
        sns.lmplot(data=df, x=col, y=target, scatter=True, hue='county', col=None)
        plt.title(f"{col.lower().replace('_',' ')} vs {target} for county type")
        plt.axvline(pop_mn, label=(f"{col.lower().replace('_',' ')} mean"), color='red',\
           linestyle='--')
        plt.axhline(df[target].mean(), label=(f"{target.lower().replace('_',' ')} mean"), \
                    color='black', linestyle='--')
        plt.legend()
        plt.show()
        print()

# KEILA'S BATHROOM FULL AND HALF CODE
def categorize_bathrooms(bath):
    if bath.is_integer():
        return 'full'
    else:
        return 'half'
    
# create a new column with the categorized bathrooms
# train['bathroom_type'] = train['bathroom'].apply(categorize_bathrooms)
