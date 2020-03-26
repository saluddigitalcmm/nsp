import pandas as pd
import numpy as np
import sklearn.ensemble
import joblib
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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