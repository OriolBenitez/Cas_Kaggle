import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from loading_data import *
from generate_features import *
from hyperparam_search import *
from scoring_model import *

params = pickle.load(open("../models/params_log.sav", 'rb'))
print("Accuracy of saved model(Logistic):",Logistic_regression(X_train,X_test,np.ravel(Y_train),np.ravel(Y_test),params={},n_slpits=10,n_repeats=3,show_confussion_matrix=False))

#example of how to do hyperparameter search
params=hyperparam_search_DecisionTree(X_train,np.array(Y_train), n_slpits=10,n_repeats=3, save_model = False,save_params=False,  output_name = "log.sav",n_iter=3)

#training the model with the parameters found
print("Accuracy of hyperparameter search model(Decission Tree):",decission_tree(X_train,X_test,np.ravel(Y_train.values),np.ravel(Y_test.values),params=params,n_slpits=10,n_repeats=3,show_confussion_matrix=False,advanced_stats=False))

