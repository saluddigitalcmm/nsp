import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Consolidator:
    def __init__(self,raw_data_folder,consolidated_data_location,citas_excel_location):
        ## Abrir y leer archivos Excel
        self.consolidated_data_location = consolidated_data_location
        self.directorio = raw_data_folder

        # Definir dataframe
        self.Datos = pd.DataFrame([])

        # Expresion regular que busca archivos con nombre INFORME ESTADISTICO
        self.newlist_Informe = []
        for filename in os.listdir(self.directorio):
            if filename.endswith("xlsx"):
                self.newlist_Informe.append(self.directorio+filename)
        self.citas_excel_location = citas_excel_location
        logger.info( 'Los informes excel son {}'.format(self.newlist_Informe))

    def consolidate(self):
        # El archivo Base.csv sera donde se guarde la base de datos completa. 
        # Si no esta en el directorio, se crea
        
        
        columns = ["PAID",u'FechaNac', u'Sexo', u'Comuna', u'Prevision',
                  u'Especialidad', u'TipoAtencion', 'TipoProfesional', 'CodPrestacion',
                  u'FechaCita', u'EstadoCita', u'HoraCita']
        self.citas_datos = pd.read_excel(self.citas_excel_location, parse_dates=['FechaNac','FechaCita','FechaReserva'])[columns[:-1] + ['FechaReserva']]
        self.citas_datos["PAID"] = pd.to_numeric(self.citas_datos["PAID"],errors="coerce",downcast="integer")
        logger.info("citas_datos shape: {}".format(self.citas_datos.shape))
        self.citas_datos.dropna(inplace=True)
        logger.info("citas_datos shape: {}".format(self.citas_datos.shape))
        logger.info('Leer todos los excels para compilarlos')
        for archivo in self.newlist_Informe:

            # Abro el archivo excel
            df = pd.ExcelFile(archivo)
            # Leo la primera hoja del excel
            try:
                DatosDF = df.parse(sheet_name=df.sheet_names[0],parse_dates=['FechaNac','FechaCita'])
                DatosDF["PAID"] = pd.to_numeric(DatosDF["PAID"],errors="coerce",downcast="integer")
            
                DatosDF = DatosDF[columns]
                self.Datos = pd.concat([self.Datos,DatosDF])
                logger.info("{} cargado".format(archivo))
            except KeyError as e:
                logger.info("{} error: {}".format(archivo,e))
            except ValueError as e:
                logger.info("{} error: {}".format(archivo,e))
        logger.info("Datos shape: {}".format(self.Datos.shape))
        self.Datos.dropna(inplace=True)
        logger.info("Datos shape: {}".format(self.Datos.shape))
        self.Datos = self.Datos.merge(self.citas_datos[['FechaNac','FechaCita','CodPrestacion','FechaReserva']],how="left")
        logger.info("Datos shape: {}".format(self.Datos.shape))
        logger.info(self.Datos.columns)

        self.Datos.dropna(inplace=True)
        logger.info("Datos shape: {}".format(self.Datos.shape))
        
        #Datos.to_csv('Base.csv', sep='\t', encoding='utf-8',index=True)
        self.Datos.to_csv(self.consolidated_data_location, encoding='utf-8',index=False)
