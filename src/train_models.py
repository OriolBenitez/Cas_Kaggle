import sys
sys.path.insert(1, 'helpers')

from generate_features import *
from utils import *
from imports import *

def Logistic_regression(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = LogisticRegression()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)

def Random_Forest(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = RandomForestClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)

def Cat_Boost(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model=CatBoostClassifier(verbose=0)
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)

def xg_Boost(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = xgb.XGBClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train,eval_metric='logloss',verbose=0)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)
def lgbm(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = LGBMClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1,verbose=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)

def decission_tree(X_train,X_test,Y_train,Y_test,params,n_slpits=10,n_repeats=3,show_confussion_matrix=False):
    cv =RepeatedStratifiedKFold(n_splits=n_slpits,n_repeats=n_repeats)
    model = DecisionTreeClassifier()
    model.set_params(**params)
    model.fit(X_train,Y_train)
    scores = cross_val_score(model, X_test, Y_test, scoring='accuracy', cv=cv, n_jobs=-1)
    if(show_confussion_matrix):
        visualize_confusion_matrix(model.predict(X_test),Y_test)
    return np.mean(scores)