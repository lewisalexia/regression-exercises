
# Regression Metrics

def plot_residuals(y, yhat):
    """This function calculates and plots residuals (actual - predicted)
    ---
    Format: residuals = function()
    ---
    Then add to df as a new column.
    """
    import matplotlib.pyplot as plt

    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title(' Residual vs Home Value Plot')
    plt.show()

    return residuals

def regression_errors(y, yhat):
    """This function returns regression error metrics.
    ---
    Format: SSE, ESS, TSS, MSE, RMSE = function()
    ---
    MSE: SSE / len(df) \\ Average Squared Errors
    SSE: (yhat - target) ^2 \\ Sum of Squared Errors
    RMSE: sqrt(MSE) \\ Root Mean Squared Error
    ESS: sum((yhat - y.mean())**2) \\ Explained Sum of Squares (for r2 value)
    TSS: ESS + SSE \\ Total Sum of Squares (for r2 value)
    """

    from sklearn.metrics import mean_squared_error

    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE ** .5
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE

    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    """This function calculates baseline mean errors from regression errors function
    ---
    Format: MSE, SSE, RMSE = function()    
    """
    import numpy as np 
    from sklearn.metrics import mean_squared_error

    baseline = np.repeat(y.mean(),len(y))

    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE ** .5

    return MSE, SSE, RMSE

def better_than_baseline(SSE, SSE_baseline):
    """This function returns true if model performed better than baseline, otherwise false"""
    if SSE < SSE_baseline:
        return True
    else:
        return False
    
# -------------------------------------------------------------------------

# Feature Modeling

def select_kbest(encoded_train_scaled_mm, encoded_train, y_train, n_features):
    """This function returns the top features to model based on K Best selection."""
    import pandas as pd
    from sklearn.feature_selection import SelectKBest, f_regression

    # MAKE the thing
    kbest = SelectKBest(f_regression, k=n_features)

    # FIT the thing
    kbest.fit(encoded_train_scaled_mm, y_train)
    
    # USE the thing and put into DF
    encoded_train_KBtransformed = pd.DataFrame(kbest.transform(encoded_train_scaled_mm),\
                                     columns = encoded_train.columns[kbest.get_support()],\
                                     index=encoded_train.index)
    
    # return column names
    print(f"These are the top {n_features} columns selected from Select K Best model:")
    return encoded_train_KBtransformed.columns.tolist()

def rfe(encoded_train_scaled_mm, encoded_train, y_train, n_features):
    """This function returns the top features to model based on RFE selection."""
    import pandas as pd    
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    lr = LinearRegression()

    # make the thing
    rfe = RFE(lr, n_features_to_select = n_features)

    # fit the thing
    rfe.fit(encoded_train_scaled_mm, y_train)
    
    # use the thing
    encoded_train_RFEtransformed = pd.DataFrame(rfe.transform(encoded_train_scaled_mm),index=encoded_train.index,
    columns = encoded_train.columns[rfe.support_])
    
     # return column names
    print(f"These are the top {n_features} columns selected from RFE model:")
    return encoded_train_RFEtransformed.columns.tolist()
