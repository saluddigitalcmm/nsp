import pandas as pd
import sklearn.preprocessing
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Featurizer:
    def __init__(self,interim_dataset_location):
        self.data = pd.read_csv(interim_dataset_location)
        logger.info("current shape: {}".format(self.data.shape))
    def generate_features(self):
        self.data["FechaCita"] = pd.to_datetime(self.data["FechaCita"])
        self.data["HoraCita"] = pd.to_datetime(self.data["HoraCita"])
        self.data["FechaNac"] = pd.to_datetime(self.data["FechaNac"])
        self.data["age"] = (self.data["FechaCita"]-self.data["FechaNac"]).astype('timedelta64[Y]')
        self.data["month"] = self.data["FechaCita"].dt.month_name()
        self.data["day"] = self.data["FechaCita"].dt.day_name()
        self.data["hour"] = self.data["HoraCita"].dt.hour
        self.data.drop(columns=["FechaCita","HoraCita","FechaNac"], inplace=True, axis=1)
        logger.info("current shape: {}".format(self.data.shape))

        self.data = self.data[self.data["EstadoCita"].isin(["Atendido","No Atendido"])]
        self.data = self.data[(self.data["age"] <= 15) & (self.data["age"] >= 0)]
        self.data = self.data[(self.data["hour"] <= 17) & (self.data["age"] >= 8)]
        self.data["hour"] = self.data["hour"].astype('category')
        logger.info("current shape: {}".format(self.data.shape))

        age_scaler = sklearn.preprocessing.MinMaxScaler()
        self.data["age"] = age_scaler.fit_transform(self.data["age"].values.reshape(-1, 1))

        self.data = pd.get_dummies(self.data)
        logger.info("current shape: {}".format(self.data.shape))
    def write(self,data_location):
        self.data.to_csv(data_location,index=False)


