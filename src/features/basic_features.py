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

        ## Crear clasificacion TipoPrestacion medica/no medica a partir de los codigos FONASA

        # Lista de codigos de consultas medicas
        CodigosConsultaMedica = ['101111','101112','101113','903001']

        # Lista de codigos de consultas no medicas
        CodigosConsultaNoMedica = ['903002','903303','102001','102005','102006','102007',
                                '1101004','1101011','1101041','1101043','1101045','1201009',
                                '1301008','1301009']

        # Lista de codigos de procedimientos
        CodigosProcedimiento = ['305048','901005','1101009','1101010','1701003',
                                '1701006','1701045','1707002','1901030','2701013',
                                '2701015','2702001','AA09A3','AA09A4','Estudio']

        self.data['TipoPrestacionC'] = 'OTRO'

        self.data.loc[self.data['CodPrestacion'].isin(CodigosConsultaMedica),'TipoPrestacionC'] = 'ConsultaMedica'
        self.data.loc[self.data['CodPrestacion'].isin(CodigosConsultaNoMedica),'TipoPrestacionC'] = 'ConsultaNoMedica'
        self.data.loc[self.data['CodPrestacion'].isin(CodigosProcedimiento),'TipoPrestacionC'] = 'Procedimiento'
        #DD.loc[np.logical_not(DD['CodPrestacion'].isin(CodigosConsultaMedica+CodigosConsultaNoMedica+CodigosProcedimiento)),'TipoPrestacionC'] = 'OTRO'

        ## Crear clasificacion TipoAtencion Cita Nueva/Repetida/Otros

        # Lista de tipos de consultas nuevas
        TipoAtencion_Nueva = ['Consulta Nueva','Consulta Compleja Nueva']

        # Lista de tipos de consultas repetidas
        TipoAtencion_Repetida = ['Consulta Repetida','Consulta Compleja Repetida']

        # Lista de tipos de consultas nuevas en APS
        TipoAtencion_NuevaAPS = ['CN - ABIERTA APS','Consulta Nueva GES INCA']

        # Lista de otros tipos de atencion
        TipoAtencion_Otros = ['PROC - HLCM','Consulta Abreviada (Receta)',
                            'CN - Post Operados','CN - Consulta nueva GES',
                            'PROC - ABIERTA APS','CN - Post Operado',
                            'Alta de Tratamiento','Procedimiento']

        #map(unicode,Profesional_medico)
        #map(unicode,Profesional_medico)

        # definir la columna con clasificacion de tipo de atencion TipoAtencionC (clasificacion)
        self.data['TipoAtencionC'] = 'OTRO'

        # transformar las entradas de TipoAtencion de unicode a string
        self.data['TipoAtencion'].apply(lambda x: str(x))

        # decir si el tipo de atencion es nueva, repetida, nueva_APS u otra, 
        # guardar en TipoProfesionalC
        self.data.loc[self.data['TipoAtencion'].isin(TipoAtencion_Nueva),'TipoAtencionC'] = 'ConsultaNueva'
        self.data.loc[self.data['TipoAtencion'].isin(TipoAtencion_Repetida),'TipoAtencionC'] = 'ConsultaRepetida'
        self.data.loc[self.data['TipoAtencion'].isin(TipoAtencion_NuevaAPS),'TipoAtencionC'] = 'ConsultaNuevaAPS'
        #DD.loc[DD['TipoAtencion'].isin(TipoAtencion_Otros),'TipoAtencionC'] = 'Otros'

        self.data = self.data.drop(columns=["PAID","FechaCita","HoraCita","FechaNac","EstadoCita",'TipoProfesional', 'CodPrestacion', 'TipoAtencion'], axis=1)
        logger.info(self.data.columns)
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
        logger.info(self.data.columns)
        self.data.to_csv(data_location,index=False)


