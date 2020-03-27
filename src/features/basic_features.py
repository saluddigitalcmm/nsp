import pandas as pd
import sklearn.preprocessing
import logging
import numpy as np
import sqlite3
from io import StringIO
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Featurizer:
    def __init__(self,interim_dataset_location):
        self.data = pd.read_csv(interim_dataset_location)
        logger.info("current shape: {}".format(self.data.shape))
    def generate_basic_features(self):
        self.data["FechaCita"] = pd.to_datetime(self.data["FechaCita"])
        self.data["HoraCita"] = pd.to_datetime(self.data["HoraCita"])
        self.data["FechaNac"] = pd.to_datetime(self.data["FechaNac"])
        self.data["age"] = (self.data["FechaCita"]-self.data["FechaNac"]).astype('timedelta64[Y]')
        self.data["month"] = self.data["FechaCita"].dt.month_name()
        self.data["day"] = self.data["FechaCita"].dt.day_name()
        self.data["hour"] = self.data["HoraCita"].dt.hour

        self.data = self.data[self.data["EstadoCita"].isin(["Atendido","No Atendido"])]
        self.data["NSP"] = np.where(self.data["EstadoCita"] == "No Atendido",1,0)

        
        logger.info("current shape: {}".format(self.data.shape))

        self.data = self.data[(self.data["age"] <= 15) & (self.data["age"] >= 0)]
        self.data = self.data[(self.data["hour"] <= 17) & (self.data["hour"] >= 8)]
        self.data["hour"] = self.data["hour"].astype('category')
        logger.info("current shape: {}".format(self.data.shape))

        self.data["age"] = pd.cut(self.data["age"],[0,0.5,5,12,18],right=False,labels=["lactante","infante_1","infante_2","adolescente"])
        self.data_to_history = self.data[["PAID","Especialidad","FechaCita"]]
        self.data = self.data.drop(columns=["PAID","FechaCita","HoraCita","FechaNac","EstadoCita"], axis=1)
        self.data = pd.get_dummies(self.data)
        logger.info("current shape: {}".format(self.data.shape))
    def generate_history_feature(self,db_location):
        conn = sqlite3.connect(db_location)
        cur = conn.cursor()
        def get_history_from_db(PAID,Especialidad,FechaCita,span):
            cur.execute("""
            SELECT sum(NSP) as NSP_count, count(NSP) as citation_count
            FROM history
            WHERE PAID = {}
                AND Especialidad = "{}"
                AND FechaCita >= date("{}", "-{} month")
                AND FechaCita < date("{}")
            """.format(PAID,Especialidad,FechaCita,span,FechaCita)
            )
            row = cur.fetchone()
            if row[1] == 0:
                p_NSP = 0.5
            else:
                try:
                    p_NSP = row[0] / row[1]
                except TypeError:
                    p_NSP = 0.5
            logger.info("{} {} {} {} {} {}".format(PAID,Especialidad,FechaCita,row[0],row[1],p_NSP))
            return p_NSP

        def get_history_from_db_simple(df):
            history = pd.read_sql("""
            
            SELECT
                PAID,
                Especialidad,
                sum(NSP) as NSP_count,
                count(NSP) as citation_count
            FROM
                history
            GROUP BY
                PAID,
                Especialidad
            
            """,conn)
            logger.info(df.shape)
            df = df.merge(history,on=["PAID","Especialidad"],how="left")
            logger.info(df.shape)
            return df["NSP_count"] / df["citation_count"]
        logger.info(self.data.shape)
        self.data["p_NSP"] = get_history_from_db_simple(self.data_to_history[["PAID","Especialidad"]]).fillna(value=0.5)

    def write(self,data_location):
        self.data.dropna(inplace=True)
        self.data.to_csv(data_location,index=False)


