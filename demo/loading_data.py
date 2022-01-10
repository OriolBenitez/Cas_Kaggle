import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from generate_features import *

dataset=load_dataset('..\data\csgo_round_snapshots.csv')#load dataset

dataset=remove_none_important_columns(dataset)#remove useless features

dataset=convert_categorical_strings_to_int(dataset)#Encode the labels

dataset=create_new_features(dataset)#adding the new features

X_train, X_test,Y_train, Y_test=feature_selection_and_dataset_split(dataset,feature_selection=True)#splitting into train and test