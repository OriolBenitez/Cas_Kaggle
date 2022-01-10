import sys
sys.path.insert(1, 'helpers')

from generate_features import *
from train_models import *
from utils import *
from imports import *

def hyperparam_search_logistic(X_train,Y_train, n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "log.sav",n_iter=1):
    """
   Search scikit-learn logistic model hyperparameters by doing RandomizedSearchCV, given a train data.
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   n_slpits (int): Number of splits of the cross-validation.
                   save_model(bool): true if you want to save the model
                   save_params(bool): true if you want to save the params
                   output_name(str): name to save the model and params
                   n_iter=1 (int): number of iterations of the search
           Returns:
                   params(dictionary): best params founded
                   
   """
    space = dict()
    model = LogisticRegression()
    cv = RepeatedStratifiedKFold(n_splits=n_slpits, n_repeats=n_repeats)
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] =   ['l2']
    space['C'] = loguniform.rvs(1e-5, 3,size=10000)
    start_time = time.time()
    search = RandomizedSearchCV(model, space, n_iter=n_iter, n_jobs=-1, cv=cv)
    # execute search
    result = search.fit(X_train, Y_train)
    best_params=result.best_params_
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        pickle.dump(result, open('../models/model_' + output_name, 'wb'))
    return best_params

def hyperparam_search_RF(X_train,Y_train, n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "rf.sav",n_iter=1):
    """
   Search scikit-learn Random-Forest model hyperparameters by doing RandomizedSearchCV, given a train data.
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   n_slpits (int): Number of splits of the cross-validation.
                   save_model(bool): true if you want to save the model
                   save_params(bool): true if you want to save the params
                   output_name(str): name to save the model and params
                   n_iter=1 (int): number of iterations of the search
           Returns:
                   params(dictionary): best params founded
                   
   """
    n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 150)]
    R_f=RandomForestClassifier()
    # Number of features to consider at every split
    cv = RepeatedStratifiedKFold(n_splits=n_slpits, n_repeats=n_repeats)
    criterion=['entropy','gini']
    
    # Minimum number of samples required to split a node
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'bootstrap': bootstrap}
    clf =  RandomizedSearchCV(RandomForestClassifier(), random_grid, n_iter=n_iter, n_jobs=-1, cv=cv)
    #Fit the model
    best_model = clf.fit(X_train,Y_train)
    best_params=best_model.best_params_
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        pickle.dump(best_model, open('../models/model_' + output_name, 'wb'))
    return best_params

def hyperparam_search_xgboost(X_train,X_test,y_train,y_test, n_slpits=10,n_repeats=3,save_model = False,save_params=False,  output_name = "xgboost.sav",n_iter=1):
    """
   Search Xgboost model hyperparameters by doing RandomizedSearch usin Hyperopt library, given a train data.
           Parameters:
                   X_train(pandas dataframe):features of the variable that we want to predict
                   Y_train(vector):variable to predict
                   X_test(pandas dataframe):features of the variable that we want to predict at test
                   Y_est(vector):variable to predict at test
                   n_slpits (int): Number of splits of the cross-validation.
                   save_model(bool): true if you want to save the model
                   save_params(bool): true if you want to save the params
                   output_name(str): name to save the model and params
                   n_iter=1 (int): number of iterations of the search
           Returns:
                   params(dictionary): best params founded
                   
   """
    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }
    def objective(space):
        clf=xgb.XGBClassifier(
                        n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                        reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                        colsample_bytree=int(space['colsample_bytree']))
        
        evaluation = [( X_train, y_train), ( X_test, y_test)]
        
        clf.fit(X_train, y_train,
                eval_set=evaluation, eval_metric="auc",
                early_stopping_rounds=10,verbose=0)
        

        pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, pred>0.5)
        return {'loss': -accuracy, 'status': STATUS_OK }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = n_iter,
                            trials = trials,verbose=0)
    best_hyperparams["max_depth"]=int(best_hyperparams["max_depth"])
    best_params=best_hyperparams
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        clf=xgb.XGBClassifier()
        clf.set_params(**best_params)
        pickle.dump(clf, open('../models/model_' + output_name, 'wb'))
    return best_hyperparams

def hyperparam_search_Catboost(X_train,Y_train, n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "Catboost.sav",n_iter=1):
    """
    Search hyperopt Catboost model hyperparameters by doing RandomizedSearchCV, given a train data.
            Parameters:
                    X_train(pandas dataframe):features of the variable that we want to predict
                    Y_train(vector):variable to predict
                    n_slpits (int): Number of splits of the cross-validation.
                    save_model(bool): true if you want to save the model
                    save_params(bool): true if you want to save the params
                    output_name(str): name to save the model and params
                    n_iter=1 (int): number of iterations of the search
            Returns:
                    params(dictionary): best params founded
                    
    """
    cv = RepeatedStratifiedKFold(n_splits=n_slpits, n_repeats=n_repeats)
    parameters = { "learning_rate": np.linspace(0,5,1000),'max_depth': randint(3, 10)}#
    clf =  RandomizedSearchCV(CatBoostClassifier(), parameters, n_iter=1, n_jobs=-1, cv=cv)
    #Fit the model
    best_model = clf.fit(X_train,Y_train)
    best_params=best_model.best_params_
    best_params=best_model.best_params_
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        pickle.dump(best_model, open('../models/model_' + output_name, 'wb'))
    return best_params
def hyperparam_search_lgbm(X_train,Y_train, n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "lgbm.sav",n_iter=1):
    """
    Search hyperopt LGBM model hyperparameters by doing RandomizedSearchCV, given a train data.
            Parameters:
                    X_train(pandas dataframe):features of the variable that we want to predict
                    Y_train(vector):variable to predict
                    n_slpits (int): Number of splits of the cross-validation.
                    save_model(bool): true if you want to save the model
                    save_params(bool): true if you want to save the params
                    output_name(str): name to save the model and params
                    n_iter=1 (int): number of iterations of the search
            Returns:
                    params(dictionary): best params founded
                    
    """
    cv = RepeatedStratifiedKFold(n_splits=n_slpits, n_repeats=n_repeats)
    parameters = {'num_leaves':[1000,1250,1500], 'n_estimators':[50,100,150],'max_depth':[-1],
             'learning_rate':[random.uniform(0, 3) for i in range(1000)],'reg_alpha':[random.uniform(0, 1) for i in range(1000)]}
    clf =  RandomizedSearchCV(LGBMClassifier(), parameters, n_iter=n_iter, n_jobs=-1, cv=cv)
    #Fit the model
    best_model = clf.fit(X_train,Y_train)
    best_params=best_model.best_params_
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        pickle.dump(best_model, open('../models/model_' + output_name, 'wb'))
    return best_params
def hyperparam_search_DecisionTree(X_train,Y_train, n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "decission_tree.sav",n_iter=1):
    """
    Search scikit-learn Decision Tree model hyperparameters by doing RandomizedSearchCV, given a train data.
            Parameters:
                    X_train(pandas dataframe):features of the variable that we want to predict
                    Y_train(vector):variable to predict
                    n_slpits (int): Number of splits of the cross-validation.
                    save_model(bool): true if you want to save the model
                    save_params(bool): true if you want to save the params
                    output_name(str): name to save the model and params
                    n_iter=1 (int): number of iterations of the search
            Returns:
                    params(dictionary): best params founded
                    
    """
    cv = RepeatedStratifiedKFold(n_splits=n_slpits, n_repeats=n_repeats)
    Estimator = DecisionTreeClassifier()
    parameters = {'max_depth':[None,1,2,3], 'min_samples_leaf' :[2,3,4] }
    clf =  RandomizedSearchCV(DecisionTreeClassifier(), parameters, n_iter=n_iter, n_jobs=-1, cv=cv)
    best_model = clf.fit(X_train,Y_train)
    best_params=best_model.best_params_
    best_params=best_model.best_params_
    if (save_params):
        #Saving model to pickle file
        pickle.dump(best_params, open('../models/params_' + output_name, 'wb'))
    if (save_model):
        #Saving model to pickle file
        pickle.dump(best_model, open('../models/model_' + output_name, 'wb'))
    return best_params
   