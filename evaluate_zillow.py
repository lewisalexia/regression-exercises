
def plot_residuals(y, yhat):
    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title(' Residual vs Home Value Plot')
    plt.show()

def regression_errors(y, yhat):
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = MSE ** .5
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE

    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    baseline = np.repeat(y.mean(),len(y))

    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE ** .5

    return MSE, SSE, RMSE

def better_than_baseline(y, yhat):
    """returns true if your model performed better than baseline, otherwise false"""