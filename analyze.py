from src.data.data_consolidation import Consolidator
from src.features.basic_features import Featurizer

c = Consolidator(r"data/raw/informe_estadistico/",r"data/interim/Base.csv")
c.consolidate()

f = Featurizer("data/interim/Base.csv")
f.generate_features()
f.write("data/interim/data.csv")
