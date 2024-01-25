import imblearn.ensemble
import numpy as np
import sklearn.linear_model
import sklearn.svm
import sklearn.dummy
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import xgboost

MODELS = [
    (
        sklearn.linear_model.LogisticRegression(),
        {
            "C":np.logspace(-5,5,11),
            "penalty":["l1","l2"],
            "class_weight":["balanced"]
        }
    ),
    (
        sklearn.ensemble.RandomForestClassifier(),
        {
            'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            "class_weight":["balanced"]
        }
    ),
    (
        sklearn.neural_network.MLPClassifier(),
        {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
    ),
    (
        sklearn.svm.SVC(),
        {
            'C':[1,10,100,1000],
            'gamma':[1,0.1,0.001,0.0001], 
            'kernel':['linear','rbf'],
            "class_weight":["balanced"],
            'probability':[True]
        }
    ),
    (
        imblearn.ensemble.RUSBoostClassifier(),
        {
            'n_estimators': [50, 100, 400, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            "replacement":[True,False]
        }
    ),
    (
        imblearn.ensemble.BalancedRandomForestClassifier(),
        {
            'bootstrap': [True, False],
            'max_depth': [10, 30, 50, 80, 100, None],
            'max_features': ['auto', 'log2', None],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [100, 200, 500, 1000, 1200, 1400, 1600, 1800],
            "class_weight":["balanced", None]
        }
    ),
    (
        imblearn.ensemble.BalancedBaggingClassifier(),
        {
            'bootstrap': [True, False],
            'bootstrap_features': [True, False],
            'warm_start':[False],
            'replacement':[True, False],
            'n_estimators': [10,50, 100, 200, 500, 1000, 1200, 1400, 1600, 1800]
        }
    ),
    (
        imblearn.ensemble.EasyEnsembleClassifier(),
        {
            'n_estimators': [10,50, 100, 200, 500, 1000, 1200, 1400, 1600, 1800],
            'warm_start':[False],
            'replacement':[True, False]
        }
    ),
    (
        sklearn.ensemble.AdaBoostClassifier(),
        {
            'learning_rate': [0.01, 0.05, 0.1, 0.2, None],
            'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
        }
    ),
    (
        xgboost.XGBClassifier(),
        {
            'booster' : ['gblinear', 'gbtree'],
            'learning_rate': [0.01, 0.05, 0.1, 0.2, None],
            'n_estimators': [50, 100, 200, 300, 500, 750, 1000],
            'max_depth': [2, 4, 8, 10, 30, 50, 80, None],
            'subsample': [0.3, 0.5, 0.75, None],
            'scale_pos_weight ':[3,4,5]
        }
    ),

]
