import sys
sys.path.insert(1, '../src/helpers')
sys.path.insert(1, '../src')
from imports import *
from loading_data import *
from scoring_model import *

print("LBGM: \n")
scoring_model(X_test,Y_test,"..\models\model_lgbm.sav",show_confussion_matrix=True,advanced_stats=True)
print("XGBoost:\n")
scoring_model(X_test,Y_test,"..\models\model_xgboost.sav",show_confussion_matrix=True,advanced_stats=True)
print("Decission Tree:\n")
scoring_model(X_test,Y_test,"..\models\model_dt.sav",show_confussion_matrix=True,advanced_stats=True)
print("CatBoost:\n")
scoring_model(X_test,Y_test,"..\models\model_Catboost.sav",show_confussion_matrix=True,advanced_stats=True)
print("Logistic :\n")
scoring_model(X_test,Y_test,"..\models\model_log.sav",show_confussion_matrix=True,advanced_stats=True)
print("Random Forest :\n")
scoring_model(X_test,Y_test,"..\models\model_RandomForest.sav",show_confussion_matrix=True,advanced_stats=True)
