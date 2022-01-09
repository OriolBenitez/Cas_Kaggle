import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from utils import *

def scoring_model(X_test,Y_test,filename,show_confussion_matrix=False):
    model = pickle.load(open(filename, 'rb'))
    predictions=model.predict(X_test)
    result = accuracy_score(Y_test, predictions)
    print("Accuracy:",result)
    if(show_confussion_matrix):
        visualize_confusion_matrix(predictions,Y_test)
    return 0
