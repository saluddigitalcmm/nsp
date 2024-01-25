import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.svm
import sklearn.dummy
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import imblearn.metrics
import imblearn.ensemble
import joblib
import json
import logging
import sklearn.tree

from utils.settings import MODELS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(11)

def cost_effectiveness_score(y_true,y_pred):
    confusion_matrix=sklearn.metrics.confusion_matrix(y_true,y_pred)
    FP = confusion_matrix[0,1]
    FN = confusion_matrix[1,0]
    TP = confusion_matrix[1,1]
    TN = confusion_matrix[0,0]

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    total = FP + FN + TP + TN
    calls = FP + TP
    nsp_i = FN + TP

    nsp_i_p=nsp_i / total
    calls_p=calls / total
    nsp_f_p=FN/total
    reduction_p=1-nsp_f_p/nsp_i_p
    cost_effectiveness=reduction_p * (1 - calls_p)

    return cost_effectiveness

cost_effectiveness_scorer = sklearn.metrics.make_scorer(cost_effectiveness_score)
geometric_mean_scorer = sklearn.metrics.make_scorer(imblearn.metrics.geometric_mean_score)

f2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=2, pos_label=1)

class Encoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, sklearn.tree.DecisionTreeClassifier):
            return obj.__class__.__name__
        else:
            return super(Encoder, self).default(obj)

class NspModelDev:
    def __init__(self, features_train, label_train, subsample=None, models = MODELS, models_subset=False):
        if models_subset:
            self.models = [models[i] for i in models_subset]
        else:
            self.models = models
        self.features_train = pd.read_csv(features_train)
        self.label_train = pd.read_csv(label_train)
        self.train = np.concatenate([self.features_train, self.label_train], axis=1)
        self.train_complete= np.copy(self.train)
        if subsample:
            idx = np.random.randint(len(self.train), size=subsample)
            self.train = self.train[idx,:]

    def grid_search(self,report_location,n_jobs=4, scoring="cost_effectiveness"):
        self.gs_scores = {}
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            grid = model[1]
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            if scoring == "cost_effectiveness":
                scorer = cost_effectiveness_scorer
            if scoring == "geometric_mean":
                scorer = geometric_mean_scorer
            grid_search = sklearn.model_selection.RandomizedSearchCV(
                estimator=estimator,
                param_distributions=grid,
                scoring=scorer,
                n_jobs=n_jobs,
                verbose=2,
                random_state=11,
                return_train_score=True,
                cv=3
            )
            grid_search.fit(features,labels)
            self.gs_scores[model_name] = [grid_search.cv_results_,grid_search.best_params_,grid_search.best_score_]
            with open(report_location + 'grid_search_' + model_name + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.gs_scores[model_name], json_file, indent=2, ensure_ascii=False, cls=Encoder)
    def train_models(self, grid_search_results_location,n_jobs,report_location):
        self.cv_scores = {}
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            with open(grid_search_results_location + 'grid_search_' + model_name + '.json', "r") as read_file:
                data = json.load(read_file)
            best_hp=data[1]
            estimator.set_params(**best_hp)
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            cv_scores = sklearn.model_selection.cross_validate(
                estimator=estimator,
                X=features,
                y=labels,
                cv=10,
                n_jobs=n_jobs,
                scoring = {
                    'accuracy':'accuracy',
                    'f1_weighted':'f1_weighted',
                    'precision_weighted':'precision_weighted',
                    'recall_weighted':'recall_weighted',
                    'roc_auc':'roc_auc',
                    'f2_True':f2_scorer,
                    'cost_effectiveness':cost_effectiveness_scorer,
                    'geometric_mean':geometric_mean_scorer
                },
                verbose=2,
                return_train_score=True
            )
            self.cv_scores[model_name] = cv_scores
            with open(report_location + 'cross_val_' + model_name + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.cv_scores[model_name], json_file, indent=2, ensure_ascii=False, cls=Encoder)
    def train_best_models(self,models_location,grid_search_results_location,n_jobs=-1,complete=True):
        if complete:
            features = self.train_complete[:,:-1]
            label = self.train_complete[:,-1]
        else:
            features = self.train[:,:-1]
            label = self.train[:,-1]
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            with open(grid_search_results_location + 'grid_search_' + model_name + '.json', "r") as read_file:
                data = json.load(read_file)
            best_hp=data[1]
            try:
                estimator.set_params(**best_hp,n_jobs=n_jobs,verbose=2)
            except:
                try:
                    estimator.set_params(**best_hp,verbose=2)
                except:
                    estimator.set_params(**best_hp)
            estimator.fit(features,label)
            joblib.dump(estimator, models_location + model_name + ".joblib")
        
    def predict_best_models (self,models_location,features_test, label_test):
        features_test = pd.read_csv(features_test)
        label_test = pd.read_csv(label_test)
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = joblib.load(models_location + model_name + ".joblib")
            try:
                predictions_class = estimator.predict(features_test)
            except ValueError:
                predictions_class = estimator.predict(features_test.values)
            try:
                predictions_probs = estimator.predict_proba(features_test)
            except ValueError:
                predictions_probs = estimator.predict_proba(features_test.values)
            except:
                predictions_probs = np.zeros((len(predictions_class),2))
            results = np.column_stack([label_test,predictions_class,predictions_probs])
            np.savetxt(models_location + model_name + "_predictions.txt",results)
        estimator = sklearn.dummy.DummyClassifier()
        features = self.train[:,:-1]
        label = self.train[:,-1]
        estimator.fit(features,label)
        predictions_class = estimator.predict(features_test)
        predictions_probs = estimator.predict_proba(features_test)
        results = np.column_stack([label_test,predictions_class,predictions_probs])
        np.savetxt(models_location + "DummyClassifier" + "_predictions.txt",results)