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
    def __init__(self,raw_data_folder,consolidated_data_location):
        ## Abrir y leer archivos Excel
        self.consolidated_data_location = consolidated_data_location
        self.directorio = raw_data_folder

        # Definir dataframe
        self.Datos = pd.DataFrame([])

        # Expresion regular que busca archivos con nombre INFORME ESTADISTICO
        self.newlist_Informe = []
        for filename in os.listdir(self.directorio):
            self.newlist_Informe.append(self.directorio+filename)
        logger.info( 'Los informes excel son {}'.format(self.newlist_Informe))

    def consolidate(self):
        # El archivo Base.csv sera donde se guarde la base de datos completa. 
        # Si no esta en el directorio, se crea
        

        
        logger.info('Leer todos los excels para compilarlos')
        
        for archivo in self.newlist_Informe:
        
            # Abro el archivo excel
            df = pd.ExcelFile(archivo)
            # Leo la primera hoja del excel
            DatosDF = df.parse(sheet_name=df.sheet_names[0])
            self.Datos = pd.concat([self.Datos,DatosDF])
            logger.info("{} cargado".format(archivo))
        
        del df
        del DatosDF
            
        #Datos.to_csv('Base.csv', sep='\t', encoding='utf-8',index=True)
        self.Datos.to_csv(self.consolidated_data_location, encoding='utf-8',index=False)
