import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from generate_features import *
from hyperparam_search import *
from hyperparam_search import *



dataset=load_dataset('..\data\csgo_round_snapshots.csv')
dataset=remove_none_important_columns(dataset)
dataset=convert_categorical_strings_to_int(dataset)
dataset=create_new_features(dataset)
X_train, X_test,Y_train, Y_test=feature_selection_and_dataset_split(dataset)
params=hyperparam_search_lgbm(X_train,np.ravel(Y_train), n_slpits=10,n_repeats=3, save = True,  output_name = "lgbm.sav")
print("Accuracy:", xg_Boost(X_train,X_test,np.ravel(Y_train),np.ravel(Y_test),params,n_slpits=10,n_repeats=3,show_confussion_matrix=True))
