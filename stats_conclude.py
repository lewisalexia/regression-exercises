from scipy import stats

# STATS CONCLUSIONS FUNCTIONS 

def chi2_test(table):
    """Goal: ((compare two categorical variables))
    Compare categorical variables testing for correlation.
    ---
    Sets alpha to 0.05, runs a chi^2 test, and evaluates the pvalue 
    against alpha, printing a conclusion statement.
    ---
    This function takes in a pd.crosstab table to analyze.
    """
    α = 0.05
    chi2, p, degf, expected = stats.chi2_contingency(table)
    print('Observed')
    print(table.values).head()
    print('\nExpected')
    print(expected.astype(int)).head()
    print('\n----')
    print(f'chi^2 = {chi2:.10f}')
    print(f'p-value = {p:.10f} < {α}')
    print('----')
    if p < α:
        print ('We reject the null hypothesis.')
    else:
        print ("We fail to reject the null hypothesis.")

def conclude_1samp_tt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Non-parametric = Wilcoxon)
    Bubble in the bubble...
    ---
    This function is a one-sample, two-tailed, t-test reliant on parametric data.
    ---
    Sets alpha = 0.05, runs a one-sample, two-tailed test, evaluates the 
    pvalue and tstat against alpha, printing a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f"Assumptions are met: One-Sample, Two-Tailed T-Test successful...")
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p:.10f} < {α}?')
    print('\n----')
    if ((p < α) & (tstat > 0)):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_1samp_gt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Non-parametric = Wilcoxon)
    Bubble in the bubble...
    ---
    This function is a one-sample, one-tailed, t-test reliant on parametric data.
    ---
    Sets alpha = 0.05, runs a one-sample, one-tailed test (greater than), evaluates the 
    pvalue and tstat against alpha, and prints a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f"Assumptions are met: One-Sample, Right-Tailed T-Test successful...")
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat > 0):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_1samp_lt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Non-parametric = Wilcoxon)
    Bubble in the bubble...
    ---
    This function is a one-sample, one-tailed, t-test reliant on parametric data.
    ---
    Sets alpha = 0.05, runs a one-sample, one-tailed test (less than), evaluates the 
    pvalue and tstat against alpha, and prints a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.ttest_1samp(group1, group_mean)
    print(f"Assumptions are met: One-Sample, Left-Tailed T-Test successful...")
    print(f't-stat: {tstat} < 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat < 0):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_2samp_tt(sample1, sample2):
    """Goal: ((compare one categorical and one continuous variable))
    Compare two observed means (independent samples). (Non-parametric = Mann-Whitney's)
    ---
    This function is a two-sample, two-tailed, t-test reliant on
    parametric data, independence, and equal variances.
    ---
    Sets alpha to 0.05, performs Levene Test, runs a two-sample, 
    two-tailed test, evaluates the pvalue against alpha, and prints 
    a conclusion statementusing the outcome of the Levene test.
    """
    from scipy import stats
    α = 0.05
    
    # Check Variance
    stat, pval = stats.levene(sample1, sample2)
    print(f"Running Levene Test...")
    if pval > α:
        print(f'p-value: {pval:.10f} > {α}?')
        print(f"Variance is True")
        
        # Perform test for true variance
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
        print(f"Assumptions are met: Two-Sample, Two-Tailed, 'equal_Var=True', T-Test successful...")    
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if p < α:
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')
        
    else:
        print(f'p-value: {pval:.10f} < {α}?')
        print(f"Variance is False")
        
        # Perform test for False variance
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=False)
        print(f"Assumptions are met: Two-Sample, Two-Tailed, 'equal_Var=False', T-Test successful...")    
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if p < α:
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')

def conclude_2samp_gt(sample1, sample2):
    """Goal: ((compare one categorical and one continuous variable))
    Compare two observed means (independent samples). (Non-parametric = Mann-Whitney's)
    ---
    This function is a two-sample, right-tailed, t-test reliant on
    parametric data, independence, and equal variances.
    ---
    Sets alpha to 0.05, performs Levene Test, runs a two-sample, 
    right-tailed test, evaluates the pvalue against alpha, and prints 
    a conclusion statementusing the outcome of the Levene test.
    """
    from scipy import stats
    α = 0.05

    # Check Variance
    stat, pval = stats.levene(sample1, sample2)
    print(f"Running Levene Test...")
    if pval > α:
        print(f'p-value: {pval:.10f} > {α}?')
        print(f"Variance is True")

        # Perform test for True variance
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
        print(f"Assumptions are met: Two-Sample, Right-Tailed, 'equal_Var=True', T-Test successful...")
        print(f't-stat: {tstat} > 0?')
        print(f'p-value: {p/2} < {α}?')
        print('----')
        if (((p/2) < α) and (tstat > 0)):
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')

    else:
        print(f'p-value: {pval:.10f} < {α}?')
        print(f"Variance is False")
        
        # Perform test for False variance
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=False)
        print(f"Assumptions are met: Two-Sample, Right-Tailed, 'equal_Var=False',T-Test successful...")    
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if (((p/2) < α) and (tstat > 0)):
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')

def conclude_2samp_lt(sample1, sample2):
    """Goal: ((compare one categorical and one continuous variable))
    Compare two observed means (independent samples). (Non-parametric = Mann-Whitney's)
    ---
    This function is a two-sample, left-tailed, t-test reliant on
    parametric data, independence, and equal variances.
    ---
    Sets alpha to 0.05, runs a Levene test, runs a two-sample, 
    left-tailed (less than) test, evaluates the pvalue against alpha, and 
    prints a conclusion statement using the outcome of the Levene test.
    """
    from scipy import stats
    α = 0.05

    # Check Variance
    stat, pval = stats.levene(sample1, sample2)
    print(f"Running Levene Test...")
    if pval > α:
        print(f'p-value: {pval:.10f} > {α}?')
        print(f"Variance is True")

        # Perform test for True variance
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=True)
        print(f"Assumptions are met: Two-Sample, Left-Tailed, 'equal_Var=True', T-Test successful...")
        print(f't-stat: {tstat} < 0?')
        print(f'p-value: {p/2} < {α}?')
        print('\n----')
        if (((p/2) < α) and (tstat < 0)):
            print("we can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')
    
    else:
        print(f'p-value: {pval:.10f} < {α}?')
        print(f"Variance is False")

        # Perform test for False variance        
        tstat, p = stats.ttest_ind(sample1, sample2, equal_var=False)
        print(f"Assumptions are met: Two-Sample, Right-Tailed, 'equal_Var=False', T-Test successful...")    
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if (((p/2) < α) and (tstat < 0)):
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')



def conclude_anova(theoretical_mean, group1, group2):
    """Goal: ((compare one continuous variable and more than one categorical))
    Compare several observed means (independent samples). (Non-parametric = Kruskal-Wallis) 
    ---
    This function is reliant on parametric data, independence, 
    and equal variances. 
    ---
    Sets alpha to 0.05, runs a Levene test, evaluates the pvalue and tstat 
    against alpha, and prints a conclusion statement based on the outcome of the
    Levene test.
    """
    from scipy import stats
    α = 0.05

    # Check Variance
    stat, pval = stats.levene(theoretical_mean, group1, group2)
    print(f"Running Levene Test...")
    if pval > α:
        print(f'p-value: {pval:.10f} > {α}?')
        print(f"Variance is True, Proceed with ANOVA test...")  

        # Perform test for true variance
        tstat, p = stats.f_oneway(theoretical_mean, group1, group2)
        print(f"Assumptions are met: ANOVA successful...")
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if p < α:
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')
    
    else:
        print(f'p-value: {pval:.10f} < {α}?')
        print(f"Variance is False, Proceed with Kruskal-Willis test...")

        # Run alternative test for false variance
        tsat, p = stats.kruskal(theoretical_mean, group1, group2)
        print(f"Assumptions are not met: Kruskal successful...")
        print(f't-stat: {tstat}')
        print(f'p-value: {p:.10f} < {α}?')
        print('----')
        if p < α:
            print("We can reject the null hypothesis.")
        else:
            print('We fail to reject the null hypothesis.')


def conclude_pearsonr(floats1, floats2):
    """Goal: ((compare two continuous variables))
    Compare two continuous variables for linear correlation. (Non-parametric = Spearman's R)
    ---
    This function is a correlation test reliant on parametric data.
    ---
    Sets alpha to 0.05, runs a Pearson's Correlation test on parametric data, 
    evaluates the pvalue against alpha, and prints a conclusion statement.
    ---
    This function takes in two seperate floats and is used when the relationship is 
    linear, both variables are quantitative, normally distributed, and have no outliers.
    """
    from scipy import stats
    α = 0.05
    r, p = stats.pearsonr(floats1, floats2)
    print(f"Parametric data: Pearson's R test successful...")
    print(f'r (correlation value): {r}')
    print(f'p-value: {p:.10f} < {α}?')
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")



def iterate_columns(df):
    """This function takes in a dataframe, iterates through all column
    combinations, runs a pearson's correlation test and appends the data 
    back to the df, and then renames the columns.
    """
    import pandas as pd
    from itertools import combinations
    from scipy.stats import pearsonr
    α = 0.05
    columns = df.columns
    if columns.is_integer():
        for i in range(2, len(columns) + 1):
            for combo in combinations(columns, i):
                corr, p_value = pearsonr(df[combo[0]], df[combo[1]])
                df[f"{combo[0]}_{combo[1]}_corr"] = corr
                df[f"{combo[0]}_{combo[1]}_p_value"] = p_value
    df.rename(columns={'assessed_worth_bed_corr':'aw_bed_corr',
     'assessed_worth_bed_p_value':'aw_bed_p',
     'assessed_worth_bath_corr':'aw_bath_corr',
     'assessed_worth_bath_p_value':'aw_bath_p',
     'assessed_worth_sqft_corr':'aw_sqft_corr',
     'assessed_worth_sqft_p_value': 'aw_sqft_p',
     'assessed_worth_year_corr':'aw_year_corr',
     'assessed_worth_year_p_value':'aw_year_p',
     'assessed_worth_property_taxes_corr':'aw_taxes_corr',
     'assessed_worth_property_taxes_p_value':'aw_taxes_p'}, inplace=True)

    return df

def conclude_spearmanr(floats1, floats2):
    """Goal: ((compare two continuous variables))
    Compare two continuous variables for linear correlation. (Parametric = Pearson's R)
    ---
    Sets alpha to 0.05, runs a Spearman's Correlation test on non-parametric
    data, evaluates the pvalue against alpha, and prints a conclusion statement.
    ---
    Takes in two seperate floats and is used when the relationship is rank-ordered, 
    both variables are quantitative, NOT normally distributed, and presents 
    as potentially monotonic.
    """
    from scipy import stats
    α = 0.05
    r, p = stats.spearmanr(floats1, floats2)
    print(f"Non-Parametric data: Spearman's R test successful...")
    print(f'r (correlation value): {r}')
    print(f'p-value: {p:.10f} < {α}?')
    print('----')
    if p < α:
        print("We can reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")

def conclude_wilcoxon_tt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Parametric = One-Sample, Two-Tailed, T-Test)
    ---
    This function is an alternative two-tailed test reliant on 
    NON-parametric data.
    ---
    Sets alpha = 0.05, runs a Wilcoxon Two-Tailed test, evaluates the 
    pvalue and tstat against alpha, and prints a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.wilcoxon(group1, group_mean)
    print(f"Non-parametric data: Wilcoxon Two-Tailed Test successful...")
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p:.10f} < {α}?')
    print('\n----')
    if ((p < α) & (tstat > 0)):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_wilcoxon_lt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Parametric = One-Sample, Left-Tailed, T-Test)
    ---
    This function is an alternative left-tailed test reliant on 
    NON-parametric data.
    ---
    Sets alpha = 0.05, runs a Wilcoxon Left-Tailed test, evaluates the 
    pvalue and tstat against alpha, and prints a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.wilcoxon(group1, group_mean)
    print(f"Non-parametric data: Wilcoxon Left-Tailed Test successful...")
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p:.10f} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat < 0):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_wilcoxon_gt(group1, group_mean):
    """Goal: ((compare one categorical and one continuous variable))
    Compare observed mean to theoretical mean. (Parametric = One-Sample, Right-Tailed T-Test)
    ---
    This function is an alternative right-tailed test reliant on 
    NON-parametric data.
    ---
    Sets alpha = 0.05, runs a Wilcoxon Right-Tailed test, evaluates the 
    pvalue and tstat against alpha, and prints a conclusion statement.
    """
    from scipy import stats
    α = 0.05
    tstat, p = stats.wilcoxon(group1, group_mean)
    print(f"Non-Parametric data: Wilcoxon Right-Tailed Test successful...")
    print(f't-stat: {tstat} > 0?')
    print(f'p-value: {p/2} < {α}?')
    print('\n----')
    if ((p / 2) < α) and (tstat > 0):
        print("We can reject the null hypothesis.")
    else:
        print('We fail to reject the null hypothesis.')

def conclude_mannwhitneyu_tt(subpop1, subpop2):
    """Goal: ((compare one categorical and one continuous variable))
    ---
    Sets alpha to 0.05, runs a Mann-Whitney U test on non-parametric, ordinal
    data, evaluates the pvalue against alpha, and prints a conclusion statement.
    ---
    This function takes in two sub-populations and is usually used to compare
    sample means: use when the data is ordinal (non-numeric) and t-test assumptions
    are not met.
    """
    from scipy import stats
    α = 0.05
    t, p = stats.mannwhitneyu(subpop1, subpop2)
    print(f"Non-Parametric/Ordinal data: Mann-Whitney test successful...")
    print(f"t-stat: {t}")
    print(f'p-value: {p:.10f} < {α}?')
    if p < α:
        print("We reject the null hypothesis.")
    else:
        print("We fail to reject the null hypothesis.")