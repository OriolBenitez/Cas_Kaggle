import sys
sys.path.insert(1, 'helpers')

from generate_features import *
from utils import *
from imports import *

def Logistic_regression(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train scikit-learn logistic model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = LogisticRegression()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)

def Random_Forest(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train scikit-learn Random Forest model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = RandomForestClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)

def Cat_Boost(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train CatBoost model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model=CatBoostClassifier(verbose=0)
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)

def xg_Boost(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train XGBoost model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = xgb.XGBClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train,eval_metric='logloss',verbose=0)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)
def lgbm(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train LGBM model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = LGBMClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)

def decission_tree(X_train,X_test,Y_train,Y_test,params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False):
    """
   Train scikit-learn Decission Tree model , and return accuracy at test
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   show_confussion_matrix(bool): true if you want to show the confussion matrix
                   advanced_stats(bool): true if you want to see advanced stats
                  
           Returns:
                   accuracy(float):accuracy of the model
                   
   """
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = DecisionTreeClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    predictions=model.predict(X_test)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return np.mean(scores)