from src.data.data_consolidation import Consolidator
from src.features.basic_features import Featurizer
from src.models.nsp_classifier import Trainer, Evaluator
from src.data.dataset_generator import Splitter

c = Consolidator(r"data/raw/informe_estadistico/",r"data/interim/Base.csv")
c.consolidate()

f = Featurizer("data/interim/Base.csv")
f.generate_features()
f.write("data/interim/data.csv")

s = Splitter("data/interim/data.csv")
s.split("data/processed/")

t = Trainer("data/processed/features_train.csv","data/processed/label_train.csv")
t.fit("models/nsp_classifier.joblib")

e = Evaluator("models/nsp_classifier.joblib","data/processed/features_test.csv","data/processed/label_test.csv")
e.evaluate("reports/classification_report.json")
