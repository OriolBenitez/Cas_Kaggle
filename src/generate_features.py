import sys
sys.path.insert(1, 'helpers')
from imports import *

def load_dataset(file):
    """
   Returns a datasest given a file.
           Parameters:
                   file (string): csv file that contains de data.
           Returns:
                   dataset (pandas dataframe): Dataset 
   """
    dataset = pd.read_csv(file)
    return dataset

def remove_none_important_columns(dataset):
    """
    Removes none important features given a dataset.
           Parameters:
                  dataset (pandas dataframe): Dataset 
           Returns:
                   dataset (pandas dataframe): Dataset 
   """
    t=[]
    for i in dataset.columns:
        t.append(dataset[i].nunique())
    temp =[]
    for i in range(len(t)):
        if t[i]==1:
            temp.append(i)
            #print(dataset.columns[i])
    for i in temp[::-1]:
        dataset=dataset.drop([dataset.columns[i]],axis=1)
    dataset=dataset.drop(columns=["ct_weapon_m249"])
    #print(dataset.columns.shape)
    return dataset

def convert_categorical_strings_to_int(dataset):
    """
    Converts categorical strings to int
           Parameters:
                  dataset (pandas dataframe): Dataset 
           Returns:
                   dataset (pandas dataframe): Dataset 
   """
    label_encoder = sklearn.preprocessing.LabelEncoder() 
    dataset['map'] = label_encoder.fit_transform(dataset['map'])
    dataset['bomb_planted'] = label_encoder.fit_transform(dataset['bomb_planted'])
    dataset['round_winner'] = label_encoder.fit_transform(dataset['round_winner'])
    return dataset

def create_new_features(dataset):
    """
    Creates new features
           Parameters:
                  dataset (pandas dataframe): Dataset 
           Returns:
                   dataset (pandas dataframe): Dataset 
   """
    dataset['difference_between_players_alive']=dataset['t_players_alive']-dataset['ct_players_alive']
    dataset['difference_between_players_health']=dataset['t_health']-dataset['ct_health']
    dataset['difference_between_players_armor']=dataset['t_armor']-dataset['ct_armor']
    return dataset

def feature_selection_and_dataset_split(dataset,feature_selection=True):
    """
   Splits the dataset, does feature selection 
           Parameters:
                  dataset (pandas dataframe): Dataset 
                  feature_selection(bool): True if you want it to do Feature selection
           Returns:
                   X_train_selected (pandas dataframe): Dataset with the independent variables at train
                   X_test_selected (pandas dataframe): Dataset with the independent variables at test
                   Y_train(pandas dataframe): Dataset with the objective variable at train
                   Y_test(pandas dataframe): Dataset with the objective variable at test
   """
    y = dataset.filter(['round_winner'])
    X =dataset.drop(['round_winner'],axis=1)
    X_train,X_test,Y_train,Y_test = train_test_split(X, y, test_size=0.2,random_state=42)
    X_test=(X_test-X_train.mean())/X_train.std()
    X_train=(X_train-X_train.mean())/X_train.std()
    if (not feature_selection):
        return (X_train, X_test,Y_train, Y_test)
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))
    sel_.fit(X_train, np.ravel(Y_train,order='C'))
    sel_.get_support()
    X_train = pd.DataFrame(X_train)
    selected_feat = X_train.columns[(sel_.get_support())]
    print('total features: {}'.format((X_train.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))
    
    removed_feats = X_train.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
    removed_feats
    X_train = sel_.transform(X_train)
    X_test = sel_.transform(X_test)
    X_train.shape, X_test.shape
    X_train_selected= pd.DataFrame(data=X_train,columns=selected_feat)
    X_test_selected=pd.DataFrame(data=X_test,columns=selected_feat)
    sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l2', solver='liblinear'))
    sel_.fit(X_train_selected, np.ravel(Y_train,order='C'))
    sel_.get_support()
    selected_feat = X_train_selected.columns[(sel_.get_support())]
    print('total features: {}'.format((X_train_selected.shape[1])))
    print('selected features: {}'.format(len(selected_feat)))
    print(selected_feat)
    print('features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_ == 0)))
    removed_feats = X_train_selected.columns[(sel_.estimator_.coef_ == 0).ravel().tolist()]
    print(removed_feats)
    X_train_selected = sel_.transform(X_train_selected)
    X_test_selected = sel_.transform(X_test_selected)
    X_train_selected= pd.DataFrame(data=X_train_selected,columns=selected_feat)
    X_test_selected=pd.DataFrame(data=X_test_selected,columns=selected_feat)
    X_train_selected.shape, X_test_selected.shape
    return(X_train_selected, X_test_selected,Y_train, Y_test)