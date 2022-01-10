import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from utils import *

def scoring_model(X_test,Y_test,filename,show_confussion_matrix=False,advanced_stats=False):
    """
    Scores data given a model.
            Parameters:
                    X_test(pandas dataframe):features of the variable that we want to predict at test
                    Y_est(vector):variable to predict at test
                    show_confussion_matrix(bool):True if you want to see the confussion matrix
                    advanced_stat(bool): True if you want to see advances stats
            Returns:
                   0: ended the scoring
                    
    """
    model = pickle.load(open(filename, 'rb'))
    predictions=model.predict(X_test)
    result = accuracy_score(Y_test, predictions)
    print("Accuracy:",result)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    if(advanced_stats):
        print(classification_report(Y_test,predictions))
    return 0
