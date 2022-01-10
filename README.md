# Pràctica CasKaggle APC UAB 2021-2022
### Author: Oriol Benítez Bravo
### NIU: 1566931
### Dataset: [csgo_round_snapshots.csv](https://www.kaggle.com/christianlillelund/csgo-round-winner-classification)
## Context of the data
Counter-Strike: Global Offensive (CS:GO) is a multiplayer first-person shooter developed by Valve and Hidden Path Entertainment.The game pits two teams, Terrorists and Counter-Terrorists, against each other in different objective-based game modes. The most common game modes involve the Terrorists planting a bomb while Counter-Terrorists attempt to stop them, or Counter-Terrorists attempting to rescue hostages that the Terrorists have captured.

The dataset consists of round snapshots from about 700 demos from high level tournament play in 2019 and 2020. Warmup rounds and restarts have been filtered, and for the remaining live rounds a round snapshot has been recorded every 20 seconds until the round is decided. Following its initial publication, It has been pre-processed and flattened to improve readability and make it easier for algorithms to process. The total number of snapshots is 122411. Snapshots are i.i.d and should be treated as individual data points, not as part of a match.

The information that has been substracted from this 700 demos every 20 seconds are 97 attributes; 3 are categorical strings, 94 are numerical data.

### Objectives:

The main objective is to train some supervised models in order to predict the team(Terrorist or Counter-Terrorist) who is going to win the round.

### Exploring the Data

In order to have a better performance, there have been created some new features using the ones given. After having added those new features, we can have a look at the correlation of the round winner with the other variables.

![correlation](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/correlations.png)

As the round winner is the variable that has to be predicted, a problem would be that the variable was unbalanced, so we can analyze that.

![countplot](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/histogram.png)

A game of CS:GO can be played at differents maps, the map to play is commonly decided(picking and banning) by the both teams, at the past there were maps called ct-sided, that means that there were so many more rounds won by the Counter-Terrorist, at the pre-processing stage we will know if the map is an important variable to predict who wins the round.

![countplot_per_map](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/histogram_per_map.png)

(Cache has so few rounds played because it was only played during a showmatch)

### Preprocessing

Some features have been removed, all of them were guns that have not been used. 

*  Feature scaling: Standardization

The given data is not standarized, once applied different models with both types of data (standarized and non-standarized), the results have showed that the models with standarized data get a better accuracy and are also faster.

*  Feature-selection

After delating the features that didn't give any information the dataset is conformed by 90 features. As I did not wanted to select a specific number of features to use, after doing a simple model with each feature and getting its accuracy, the features that are going to be selected are the ones that have the accuracy above the mean of accuracies. 


(PCA's have not shown good results, so it will not be applied)

### Models
The models that will be applied are:

* Logistic

* Random Forest

* LGBM

* CatBoost

* XGBoost

* Decission Tree

### Results
The hyperparameter search has been done applying a RandomSearch using parallelism(8 threads)

| Method         | Data used | Accuracy           | Hyperparameters                                                                                                                                                       | Time(seconds) | Number of iterations |
|----------------|-----------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|----------------------|
| Logistic       | 33%       | 0.7523895106608938 | {'solver': 'newton-cg', 'penalty': 'l2', 'C': 0.0003138337860695424}                                                                                                  | 1140          | 100                  |
| Logistic       | 100%      | 0.7505782060794011 | {'solver': 'newton-cg', 'penalty': 'l2', 'C': 0.0023463152069802705}                                                                                                  | 5283          | 25                   |
| Random Forest  | 33%       | 0.8654113226043624 | {'bootstrap': False, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None,  'min_samples_leaf': 1, 'min_samples_split': 2,  'n_estimators': 188}           | 10395         | 100                  |
| Random Forest  | 100%      | 0.858768074503717   | {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': None,  'min_samples_leaf': 1, 'min_samples_split': 2,  'n_estimators': 120}            | 11460         | 20                   |
| CatBoost       | 33%       | 0.8417612940119271 | {'learning_rate': 0.1005, 'max_depth': 9}                                                                                                                             | 11346         | 50                   |
| CatBoost       | 100%      | 0.8319542763421571 | {'learning_rate': 0.1205, 'max_depth': 4}                                                                                                                             | 11196         | 15                   |
| LGBM           | 33%       | 0.8598153745609018 | {'reg_alpha': 0.70305655040732, 'num_leaves': 1250, 'num_iterations': 150, 'max_depth': -1, 'learning_rate': 0.4103147752802211}                                      | 2846          | 100                  |
| LGBM           | 100%      | 0.8236087941225211 | {'reg_alpha': 0.01, 'num_leaves': 1250, 'num_iterations': 150, 'max_depth': -1, 'learning_rate': 0.15}                                                                | 4532          | 100                  |
| XGBoost        | 33%       | 0.7980965607385018 | {'colsample_bytree': 0.7012362120853994, 'gamma': 3.543969952820899, 'max_depth': 15, 'min_child_weight': 7.0, 'reg_alpha': 109.0, 'reg_lambda': 0.35977544396941163} | 740           | 100                  |
| XGBoost        | 100%      | 0.8095123657852987 | {'colsample_bytree': 0.7012362120853994, 'gamma': 3.543969952820899, 'max_depth': 15, 'min_child_weight': 7.0, 'reg_alpha': 109.0, 'reg_lambda': 0.35977544396941163} | 2003          | 100                  |
| Decission Tree | 33%       | 0.7991994118127604 | {'min_samples_leaf': 2, 'max_depth': None}                                                                                                                            | 304           | 100                  |
| Decission Tree | 100%      | 0.7362688830352703 | {'min_samples_leaf': 3, 'max_depth': None}                                                                                                                            | 1800          | 100                  |

We can see some confussion matrices

* Random Forest

![Image 1](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/cm_RandomForest.png?raw=true)

* CatBoost

![Image 2](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/cm_catboost.png?raw=true)

* Decision Tree

![Image 3](https://github.com/OriolBenitez/Cas_Kaggle/blob/main/pictures/cm_dt.png?raw=true)

### Demo
First, download the data from [Kaggle](https://www.kaggle.com/christianlillelund/csgo-round-winner-classification) and copy it to the data folder

If you want to train a model or look-up for Hyperparameters, at the console(being at the demo directory), run:
```
python training_models.py
```
If you want to load a model or parameters that have been saved, at the console(being at the demo directory), run:
```
python scoring_models.py
```
### Conclusions

After all the analysis and the results obtained, the following conclusions can be exposed:
* The map is an usefull variable whenever to decide which team is going to win the round.
* Random Forest, LGBM and Catboost are good models to make this predictions. Random Forest has shown the best results, so in case that we wanted to choose a predictor that guesses the round-winner it would be the chosen one. Thinking about real aplications of a predictor, a live-time predictor would be usefull for those who bet, at this case, LGBM would be the best algorithm to choose because Random Forest takes to much time to make a prediction.
* The original data contained lots of features that resulted not to be usefull.

### Ideas to work in the future

* Learn how to take data from a CS:GO match using Skybox software.

* Implement some kind of neural network to make predictions.

* Apply classes to use the models instead of using the similar functions I use.
