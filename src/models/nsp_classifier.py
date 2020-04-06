import pandas as pd
import numpy as np
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
import sklearn.metrics
import joblib
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

f2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.fbeta_score, beta=2, pos_label=1)

class NpEncoder(json.JSONEncoder):
    def default(self, obj): # pylint: disable=E0202
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

models = [
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
            "class_weight":["balanced"]
        }
    ),
]

best_estimator = sklearn.ensemble.RandomForestClassifier()
best_hp = {
    "n_estimators": 1600,
    "min_samples_split": 5,
    "min_samples_leaf": 4,
    "max_features": "auto",
    "max_depth": 100,
    "bootstrap": True
}

class NspModelDev:
    def __init__(self, features_train, label_train, features_test, label_test, subsample=None, models = models):
        self.models = models
        self.features_train = pd.read_csv(features_train)
        self.label_train = pd.read_csv(label_train)
        self.train = np.concatenate([self.features_train, self.label_train], axis=1)
        if subsample:
            idx = np.random.randint(len(self.train), size=subsample)
            self.train = self.train[idx,:]

    def grid_search(self,report_location,n_jobs=4):
        self.gs_scores = {}
        for model in self.models:
            model_name = model[0].__class__.__name__
            estimator = model[0]
            grid = model[1]
            features = self.train[:,:-1]
            labels = self.train[:,-1]
            grid_search = sklearn.model_selection.RandomizedSearchCV(
                estimator=estimator,
                param_distributions=grid,
                scoring=f2_scorer,
                n_jobs=n_jobs,
                verbose=2,
                random_state=11,
                return_train_score=True,
                cv=3
            )
            grid_search.fit(features,labels)
            self.gs_scores[model_name] = [grid_search.cv_results_,grid_search.best_params_,grid_search.best_score_]
            with open(report_location + 'grid_search_' + model_name + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.gs_scores[model_name], json_file, indent=2, ensure_ascii=False, cls=NpEncoder)
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
                    'f2_True':f2_scorer
                },
                verbose=2,
                return_train_score=True
            )
            self.cv_scores[model_name] = cv_scores
            with open(report_location + 'cross_val_' + model_name + '.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.cv_scores[model_name], json_file, indent=2, ensure_ascii=False, cls=NpEncoder)

class Trainer:
    def __init__(self,features,label):
        self.features = pd.read_csv(features)
        self.label = pd.read_csv(label)
    def fit(self, classifier_location):
        self.classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=100, verbose=5, n_jobs=-1, oob_score=True, random_state=11, class_weight='balanced')
        self.classifier.fit(self.features,self.label)
        logger.info("oob score: {}".format(self.classifier.oob_score_))
        vil = pd.DataFrame(list(zip(self.features.columns,self.classifier.feature_importances_)),columns=["var","importance"]).sort_values("importance",ascending=False)
        vil.to_csv(classifier_location + "_vil.csv",index=False)
        joblib.dump(self.classifier, classifier_location)

class Evaluator:
    def __init__(self,classifier,features,label):
        self.classifier = joblib.load(classifier)
        self.label = pd.read_csv(label)
        self.features = pd.read_csv(features)
    def evaluate(self, report_location):
        self.predictions = self.classifier.predict(self.features)
        self.classification_report = sklearn.metrics.classification_report(self.label,self.predictions,output_dict=True)
        with open(report_location, 'w', encoding='utf-8') as json_file:
            json.dump(self.classification_report, json_file, indent=2, ensure_ascii=False, cls=NpEncoder)