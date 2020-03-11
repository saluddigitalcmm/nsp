from src.data.data_consolidation import Consolidator
from src.data.hlcm_data import HlcmData

# c = Consolidator(r"data/raw/informe_estadistico/",r"data/interim/Base.csv")
# c.consolidate()

h = HlcmData(r"data/interim/Base.csv")
h.transform()
h.write(r"data/interim/Base_procesada_no_dummies.csv")
