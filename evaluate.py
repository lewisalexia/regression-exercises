# for loop to try MANY depths
# for x in range(1,20):
#     # print(x)
#     tree = DecisionTreeClassifier(max_depth=x)
#     tree.fit(X_train, y_train)
#     acc = tree.score(X_train, y_train)
#     print(f"For depth of {x:2}, the accuracy is {round(acc,2)}")

# to calculcate the best model
# make a list of lists, then turn to df, then turn to new column to give difference
# then it visulaizes the differences

def classifier_tree_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best classifier decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    import warnings
    warnings.filterwarnings("ignore")
    for x in range(1,11):
        tree = DecisionTreeClassifier(max_depth=x, random_state=123)
        tree.fit(X_train, y_train)
        train_acc = tree.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # can plot to visulaize
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()

# select a model before the split of the two graphs. A large split indicates overfitting
# when selecing the depth to run with select the point where the difference between
# the train and validate set is the smallest before they seperate.



def random_forest_eval(X_train, y_train, X_validate, y_validate):
    ''' This function is to calculate the best random forest decision tree model by running 
    a for loop to explore the max depth per default range (1,20).

    The loop then makes a list of lists of all max depth calculations, compares the
    accuracy between train and validate sets, turns to df, and adds a new column named
    difference. The function then calculates the baseline accuracy and plots the
    baseline, and the train and validate sets to identify where overfitting occurs.
    '''
    scores_all=[]

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    import warnings
    warnings.filterwarnings("ignore")

    for x in range(1,11):
        rf = RandomForestClassifier(random_state = 123,max_depth = x)
        rf.fit(X_train, y_train)
        train_acc = rf.score(X_train, y_train)
        print(f"For depth of {x:2}, the accuracy is {round(train_acc,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc = rf.score(X_validate, y_validate)

        # append to df scores_all
        scores_all.append([x, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')

        # establish baseline accuracy
    baseline_accuracy = (y_train == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(12,12))
    plt.bar(X_train.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()   

# increasing leaf samples by one and decreasing depth by 1
# for x in range(1,11):
#     print(x, 11-x)

# rf = RandomForestClassifier(random_state=123, min_samples_leaf=x, max_depth=11-x)
# rf.fit(X_train, y_train)
# train_acc = rf.score(X_train, y_train)


def knn_titanic_acq_prep_split_evaluate():
    import pandas as pd
    import numpy as np

    import acquire as acq
    import prepare as prep
    import stats_conclude as sc
    import evaluate as ev

    import matplotlib.pyplot as plt
    import seaborn as sns

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    """
    This function acquires, prepares(base level), splits, and tests for the best fit
    number of neighbors for KNN modeling. 

    This model works on the train and validate sets. 

    The output is the number of features inputted and their column titles, baseline
    accuracy, and a plot visualizing the entire test range while identifying the best
    fit model graphically and explicitly.
    """
    # acquire
    df = acq.get_titanic_data()

    # prepare
    dft = prep.clean_titanic(df)
    dft.drop(columns=['passenger_id', 'sex', 'embark_town', 'embark_town_Queenstown','sibsp','parch', 'embark_town_Southampton'], inplace=True)
    
    # split
    train, validate, test = prep.split_titanic(dft)
    
    # assign variables
    target = 'survived'
    X_train = train.iloc[:,1:]
    X_validate = validate.iloc[:,1:]
    # X_test = test.iloc[:,1:]
    y_train = train[target]
    y_validate = validate[target]
    # y_test = test[target]
    print(f"The number of features sent in : {len(X_train.columns)} and are {X_train.columns.tolist()}.")

    # run for loop and plot
    metrics = []
    for k in range(1,21):
        
        # make the model
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the model
        knn.fit(X_train, y_train)
        
        # calculate accuracy
        train_score = knn.score(X_train, y_train)
        validate_score = knn.score(X_validate, y_validate)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']
    print(f"{n} is the number of neighbors that produces the best fit model.")
    print(f"The accuracy score for the train model is {round(train_score,2)}.")
    print(f"The accuracy score for the validate model is {round(validate_score,2)}.")
    
    
    # plot the data
    metrics_df.set_index('k').plot(figsize = (14,12))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.axhline(y=train_score, color='blue', linestyle='--', linewidth=1, label='train accuracy')
    plt.axhline(y=validate_score, color='orange', linestyle='--', linewidth=1, label='validate accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,21,1))
    plt.legend()
    plt.grid()
    
# This code calculates the value of k that results in the minimum absolute 
# difference between the train and validation accuracy. Here's a step-by-step 
# breakdown of what's happening:

# results['diff_score'] retrieves the column of the DataFrame that contains the 
# difference between the train and validation accuracy for each value of k.

# np.abs(results['diff_score']) takes the absolute value of each difference score, 
# since we're interested in the magnitude of the difference regardless of its sign.

# np.abs(results['diff_score']).argmin() finds the index of the minimum value in 
# the absolute difference score column. This corresponds to the value of k that 
# results in the smallest absolute difference between the train and validation accuracy.

# results.loc[min_diff_idx, 'k'] retrieves the value of k corresponding to the 
# minimum absolute difference score.

# results.loc[min_diff_idx, 'diff_score'] retrieves the minimum absolute difference 
# score itself.


def logit_evaluate(x_df, y_s):
    import pandas as pd
    import numpy as np

    import warnings
    warnings.filterwarnings("ignore")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    """
    This function takes in a DataFrame (train, validate, test) and
    applies a plain Logistic Regression model with no hyperparameters set 
    outside of default.
    """
    #create it
    logit = LogisticRegression(random_state=123)

    #fit it
    logit.fit(x_df, y_s)

    #use it
    score = logit.score(x_df, y_s)
    print(f"The model's accuracy is {round(score,2)}")
    
    #establish series from array of coefficients to print
    coef = logit.coef_
    
    #baseline
    baseline_accuracy = (y_s == 0).mean()
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.")

    #classification report
    print(classification_report(y_s, logit.predict(x_df)))

    #coef & corresponding columns
    print(f"The coefficents for features are: {coef.round(2)}.\nThe corresponding columns are {x_df.columns.tolist()}.")
   



def all_4_classifiers(X_tr, y_tr, X_va, y_va, nn):
    """
    This function takes in the train and validate datasets, a KNN number to go 
    to (exclusive) and returns models/visuals/explicit statments for decision tree, 
    random forest, knn, and logistic regression.

    Decision Tree:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
    
    Random Forest:
        * runs for loop to discover best fit "max depth". Default to 10
        * random_state = 123
        * returns visual representing models ran and where overfitting occurs
        * explicitly identifies the baseline and best fit "max depth"
        * visually presents feature importance

    KNN:
        * runs for loop to discover best fit "number of neighbors". Default to 30.
        * explicitly identifes the number of features sent in with column names
        * explicitly identifies the best fit number of neighbors
        * explicitly states accuracy scores for train, validate, and baseline
        * visually represents findings and identifies best fit neighbor size

    Logistic Regression:
        * random_seed = 123
        * runs logit on train and vaidate set
        * prints model, baseline accuracy, and a classification report
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    import warnings
    warnings.filterwarnings("ignore")

    # DECISION TREE
    print(f"DECISION TREE")
    scores_all=[]
    
    for i in range(1,11):
        tree = DecisionTreeClassifier(max_depth=i, random_state=123)
        tree.fit(X_tr, y_tr)
        train_acc = tree.score(X_tr, y_tr)
        print(f"For depth of {i:2}, the accuracy is {round(train_acc,2)}")
        
        # evaludate on validate set
        validate_acc = tree.score(X_va, y_va)

        # append to df scores_all
        scores_all.append([i, train_acc, validate_acc])

        # turn to df
        scores_df = pd.DataFrame(scores_all, columns=['max_depth', 'train_acc', 'validate_acc'])

        # make new column
        scores_df['difference'] = scores_df.train_acc - scores_df.validate_acc

        # sort on difference
        scores_df.sort_values('difference')
        
        # establish baseline accuracy
    baseline_accuracy = (y_tr == 0).mean()
    print()
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # can plot to visulaize
    plt.figure(figsize=(12,8))
    plt.plot(scores_df.max_depth, scores_df.train_acc, label='train', marker='o')
    plt.plot(scores_df.max_depth, scores_df.validate_acc, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Decision Tree')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
    # RANDOM FOREST
    print(f"RANDOM FOREST")
    scores_rf=[]

    for i in range(1,11):
        rf = RandomForestClassifier(random_state = 123,max_depth = i)
        rf.fit(X_tr, y_tr)
        train_acc_rf = rf.score(X_tr, y_tr)
        print(f"For depth of {i:2}, the accuracy is {round(train_acc_rf,2)}")
        
        # establish feature importance variable
        important_features = rf.feature_importances_
        
        # evaluate on validate set
        validate_acc_rf = rf.score(X_va, y_va)

        # append to rf scores_all
        scores_rf.append([i, train_acc_rf, validate_acc_rf])

        # turn to df
        scores_df2 = pd.DataFrame(scores_rf, columns=['max_depth', 'train_acc_rf', 'validate_acc_rf'])

        # make new column
        scores_df2['difference'] = scores_df2.train_acc_rf - scores_df2.validate_acc_rf

        # sort on difference
        scores_df2.sort_values('difference')

        # print baseline
    print(f'The baseline accuracy is {round(baseline_accuracy,2)}')
          
        # plot to visulaize train and validate accuracies for best fit
    plt.figure(figsize=(12,8))
    plt.plot(scores_df2.max_depth, scores_df2.train_acc_rf, label='train', marker='o')
    plt.plot(scores_df2.max_depth, scores_df2.validate_acc_rf, label='validate', marker='o')
    plt.axhline(baseline_accuracy, linewidth=2, color='black', label='baseline')
    plt.xlabel('Max Depth for Random Forest')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(1,11, step=1))
    plt.title('Where do I begin to overfit?')
    plt.legend()
    plt.show()
    
        # plot feature importance
    plt.figure(figsize=(12,12))
    plt.bar(X_tr.columns, important_features)
    plt.title(f"Feature Importance")
    plt.xlabel(f"Features")
    plt.ylabel(f"Importance")
    plt.xticks(rotation = 60)
    plt.show()
    
    # KNN
    print(f"KNN")
    print(f"The number of features sent in : {len(X_tr.columns)} and are {X_tr.columns.tolist()}.")

    # run for loop and plot
    metrics = []
    for k in range(1, nn):
        
        # make the model
        knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        
        # fit the model
        knn.fit(X_tr, y_tr)
        
        # calculate accuracy
        train_score = knn.score(X_tr, y_tr)
        validate_score = knn.score(X_va, y_va)
        
        # append to df metrics
        metrics.append([k, train_score, validate_score])

        # turn to df
        metrics_df = pd.DataFrame(metrics, columns=['k', 'train score', 'validate score'])
      
        # make new column
        metrics_df['difference'] = metrics_df['train score'] - metrics_df['validate score']
    min_diff_idx = np.abs(metrics_df['difference']).argmin()
    n = metrics_df.loc[min_diff_idx, 'k']
    print(f"{n} is the number of neighbors that produces the best fit model.")
    print(f"The accuracy score for the train model is {round(train_score,2)}.")
    print(f"The accuracy score for the validate model is {round(validate_score,2)}.")
    
    
    # plot the data
    metrics_df.set_index('k').plot(figsize = (14,12))
    plt.axvline(x=n, color='black', linestyle='--', linewidth=1, label='best fit neighbor size')
    plt.axhline(y=train_score, color='blue', linestyle='--', linewidth=1, label='train accuracy')
    plt.axhline(y=validate_score, color='orange', linestyle='--', linewidth=1, label='validate accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,nn,1))
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # LOGISTIC REGRESSION TRAIN
    print(f"LOGISTIC REGRESSION")
    print(f"Train Dataset")
    #create it
    logit = LogisticRegression(random_state=123)

    #fit it
    logit.fit(X_tr, y_tr)

    #use it
    lt_score = logit.score(X_tr, y_tr)
    print(f"The train model's accuracy is {round(lt_score,2)}")
    
    #baseline
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.") 
    
    #classification report
    print(classification_report(y_tr, logit.predict(X_tr)))

# LOGISTIC REGRESSION VALIDATE
    print(f"LOGISTIC REGRESSION")
    print(f"Validate Dataset")
    #create it
    logit2 = LogisticRegression(random_state=123)

    #fit it
    logit2.fit(X_va, y_va)

    #use it
    lt_score2 = logit2.score(X_va, y_va)
    print(f"The validate model's accuracy is {round(lt_score2,2)}")

    #baseline
    print(f"The baseline accuracy is {round(baseline_accuracy,2)}.")
    
    #classification report
    print(classification_report(y_va, logit2.predict(X_va)))