import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def historial_grupo(DD,variable,lista):
    # DD es la base de datos, y lista es una lista que contiene las variables 
    # por las cuales se quiere agrupar para hacer el historial de la variable
    # la lista tiene que estar ordenada por prioridad
    
    # crear las variables binarias correspondientes a la variable correspondiente
    dumies_variable = pd.get_dummies(DD[variable])
    
    categorias_variable = dumies_variable.columns
    
    # definir intercepto
    dumies_variable['Intercepto'] = 1
    # añadir variables en lista
    dumies_variable[lista] = DD[lista]

    
    NCitaGrupo = dumies_variable.groupby(lista)['Intercepto'].cumsum()
    dumies_variable['NCitaGrupo'] = NCitaGrupo
    # ver cuando un paciente aparece por primera vez en cada grupo
    NuevoEnGrupo = dumies_variable['NCitaGrupo']==1
    dumies_variable['NuevoEnGrupo'] = NuevoEnGrupo

    numeros_variable = dumies_variable[['NCitaGrupo','NuevoEnGrupo']]
    numeros_variable.columns = [variable+'_'+str(nombre) for nombre in numeros_variable.columns]
    
    # variable que cuenta el total citas hasta el momento, por estado
    #sub = [x for x in dumies_EstadoCita.columns if not (x in ['PAID','Agenda','Especialidad'])]
    categ_variable_acum_grupo = dumies_variable.groupby(lista)[categorias_variable].cumsum()
    
    # crear los indicadores de numero y porcentaje de citas historicas
    categ_variable_acum_grupo_hist = categ_variable_acum_grupo - dumies_variable[categorias_variable]
    
    # calcular el porcentaje de citas por tipo historicas
    categ_variable_acum_grupo_hist_porc = categ_variable_acum_grupo_hist.div(dumies_variable['NCitaGrupo']-1,axis='index')*100
    # priori para las citasn nuevas: suponer que todo es uniforme
    categ_variable_acum_grupo_hist_porc = categ_variable_acum_grupo_hist_porc.fillna(value=100/len(categorias_variable))
    
    # borrar las variables incorporadas desde lista
    dumies_variable.drop(lista,axis=1,inplace=True)

    # cambiar nombres a dummies
    dumies_variable.columns = [variable+'_'+str(nombre) for nombre in dumies_variable.columns]
    categ_variable_acum_grupo_hist.columns = [str(variable)+'_'+str(nombre)+'_AcumHist' for nombre in categ_variable_acum_grupo_hist.columns]
    categ_variable_acum_grupo_hist_porc.columns = [str(variable)+'_'+str(nombre)+'_AcumHistPorc' for nombre in categ_variable_acum_grupo_hist_porc.columns]
    
    
    
    historial = pd.concat([categ_variable_acum_grupo_hist,categ_variable_acum_grupo_hist_porc,
                           numeros_variable],axis=1)
    
    
    categorias_variable2 = [variable+'_'+str(nombre) for nombre in categorias_variable]
    dumies_variable2 = dumies_variable[categorias_variable2].copy()
    
    
    #for name,group in dumies_conc.groupby(lista):
    #    print name
    #    print group
    #    print ' '

    
    return historial
class HlcmData:
    def __init__(self,data_location):
        ## Leer la base de datos compilada


        logger.info('Leer el csv compilado')
        # creo el parseador de fechas
        date_parser = pd.to_datetime
        # las variables a ser parseadas como fecha
        parse_dates = ['FechaCita','FechaNac']
        # abro el archivo csv
        self.Datos = pd.read_csv(data_location, date_parser=date_parser, parse_dates=parse_dates)
        #Datos = pd.read_csv('Base_2017-2018_PAID.csv',sep='\t', date_parser=date_parser, parse_dates=parse_dates)
        # transformo las horas de string a tiempo
        self.Datos['HoraCita'] = pd.to_datetime(self.Datos['HoraCita']).dt.time
    def transform(self):
        listavariables = [u'PAID', u'FechaNac', u'Sexo', u'Edad', u'Comuna', u'Provincia', u'Region',
                  u'Celular', u'Telefono', u'Prevision', u'Plan', u'Establecimiento',
                  u'ProfesionalNombres', u'ProfesionalApellidos', u'RutProfesional',
                  u'Agenda', u'TipoProfesional', u'Especialidad', u'TipoAtencion',
                  u'CodPrestacion', u'Prestacion', u'AgendadoPor', u'Origen', u'NumeroLE',
                  u'FechaCita', u'HoraCita', u'EstadoCita', u'MotivoC', u'MotivoNA', u'MotivoS',
                  u'TiempoEspera',u'FechaReserva']

        lista_vars_mantener = list(set(listavariables) & set(self.Datos.columns) )

        # eliminar los identificadores de fila de las bases de cada año
        #Datos.drop(Datos.columns[[0,1]],axis=1,inplace=True)
        self.Datos = self.Datos[lista_vars_mantener]

        # imprimir los nombres de las variables en la base
        logger.info("la base tiene las columnas: {}".format(self.Datos.columns))
        logger.info("la base tiene la forma: {}".format(self.Datos.shape))

        # respaldar la base de datos leida

        # crear lista con nombres de variables a botar
        variables_borrar = ['Provincia', 'Region', 'Celular', 'Telefono', 'Establecimiento','AgendadoPor',
                            'Origen', 'NumeroLE', 'ProfesionalNombres', 'ProfesionalApellidos']
        # variables a borrar que estan en la lista de variables
        variables_borrar_definitivo = list(set(variables_borrar) & set(self.Datos.columns))


        self.Datos.drop(variables_borrar_definitivo, axis=1, inplace=True)
        self.Datos.drop_duplicates(inplace=True)

        logger.info("la base tiene la forma: {}".format(self.Datos.shape))
        # El Dataframe donde se guarde las modificaciones se llama DD

        # Eliminar citas con PAID = 201X, porque no tienen buen identificador
        paid_borrar = [2013,2014,2015,2016,2017,2018,583]
        DD = self.Datos[np.logical_not(self.Datos['PAID'].isin(paid_borrar))].copy()

        logger.info( 'La cantidad de etiquetas malas por PAID es '+ str(self.Datos.shape[0] - DD.shape[0]))

        # Eliminar los datos con campos importantes faltantes en Plan, Prevision, Sexo, Comuna, EstadoCita, CodPrestacion
        DD.dropna(subset=['PAID','Comuna','Sexo','Plan','Prevision',
                        'EstadoCita','CodPrestacion','TipoAtencion'],inplace=True)

        logger.info( DD.shape)

        # Rellenar campos fatantes en Especialidad, TipoProfesional
        DD['Especialidad'] = DD['Especialidad'].fillna('No Mencionada')
        DD['Agenda'] = DD['Agenda'].fillna('No Mencionada')
        DD['TipoProfesional'] = DD['TipoProfesional'].fillna('No Mencionada')

        # Eliminar entradas donde FechaNac > FechaCita
        DD = DD.loc[DD['FechaNac']<DD['FechaCita']]

        logger.info( DD.shape)

        # Calcular edad de paciente respecto al dia de la cita
        DD['Edad_calc'] = np.floor((DD['FechaCita']-DD['FechaNac']).dt.days/365.2425)

        # Eliminar edades negativas

        negative_age = list(set(DD.loc[DD['Edad_calc']<0]['PAID'].values))
        DD = DD.loc[np.logical_not(DD['PAID'].isin(negative_age))]

        logger.info( DD.shape)

        DD = DD.loc[DD['Sexo'].isin(['Hombre','Mujer'])]

        logger.info(DD.shape)

        ## Añadido: arreglar PAIDs y CodPrestacion

        # pasar PAID a entero
        DD['PAID'] = DD['PAID'].apply(lambda x:int(x))

        # pasar CodPrestacion a string, sin enteros ni floats ni cosas con punto
        DD['CodPrestacion'] = DD['CodPrestacion'].apply(lambda x:str(x)).str.replace(r'[.].*', '')

        DD.sort_values(['PAID', 'FechaCita', 'HoraCita'], ascending=True,inplace=True)

        ## Crear variables temporales de la cita

        # obtener la hora de la cita
        DD['Hora'] = DD['HoraCita'].apply(lambda x: x.hour)

        # obtener el día de semana de la cita
        DD['Dia'] = pd.to_datetime(DD['FechaCita']).dt.weekday_name

        # obtener el mes de la cita
        DD['Mes'] = DD['FechaCita'].apply(lambda x: x.month)

        # obtener el año de la cita
        DD[u'Anyo'] = DD['FechaCita'].apply(lambda x: x.year)

        bins = list(np.arange(0,20,1)-1)+[200]

        DD['Edad_ints'] = pd.cut(DD['Edad_calc'],bins)

        ## Crear clasificacion Profesional Medico/No medico

        # lista de tipos de profesionales medicos
        Profesional_medico = ['Médico','Médico Cirujano','Odontólogo/Dentista',
                            'Cirujano(a) Dentista','Ginecólogo(a)','Psiquiatra']

        #map(unicode,Profesional_medico)

        # lista de tipos de profesionales no medicos
        Profesional_noMedico = ['Enfermera (o)','Psicólogo (a)',#'No Mencionada',
                                'Kinesiólogo (a)','Fonoaudiólogo (a)','Tecnólogo Médico',
                                'Nutricionista','Terapeuta Ocupacional','Asistente Social',
                                'Técnico Paramédico']

        #map(unicode,Profesional_medico)

        # crear columna TipoProfesionalC (Clasificacion)
        DD['TipoProfesionalC'] = 'OTRO'

        # transformar las entradas de TipoProfesional unicode a string
        DD['TipoProfesional'].apply(lambda x: str(x))

        # decir si el tipo de profesional es medico o no medico, guardar en TipoProfesionalC
        DD.loc[DD['TipoProfesional'].astype(str).isin(Profesional_medico),'TipoProfesionalC'] = 'Medico'
        DD.loc[DD['TipoProfesional'].astype(str).isin(Profesional_noMedico),'TipoProfesionalC'] = 'NoMedico'
        #DD.loc[np.logical_not(DD['TipoProfesional'].isin(Profesional_medico)),'TipoProfesionalC'] = 'NoMedico'

        #print DD.loc[DD['TipoProfesionalC']=='Medico']['TipoProfesional'].value_counts()
        #print ''
        #
        #print DD.loc[DD['TipoProfesionalC']=='NoMedico']['TipoProfesional'].value_counts()

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

        DD['TipoPrestacionC'] = 'OTRO'

        DD.loc[DD['CodPrestacion'].isin(CodigosConsultaMedica),'TipoPrestacionC'] = 'ConsultaMedica'
        DD.loc[DD['CodPrestacion'].isin(CodigosConsultaNoMedica),'TipoPrestacionC'] = 'ConsultaNoMedica'
        DD.loc[DD['CodPrestacion'].isin(CodigosProcedimiento),'TipoPrestacionC'] = 'Procedimiento'
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
        DD['TipoAtencionC'] = 'OTRO'

        # transformar las entradas de TipoAtencion de unicode a string
        DD['TipoAtencion'].apply(lambda x: str(x))

        # decir si el tipo de atencion es nueva, repetida, nueva_APS u otra, 
        # guardar en TipoProfesionalC
        DD.loc[DD['TipoAtencion'].isin(TipoAtencion_Nueva),'TipoAtencionC'] = 'ConsultaNueva'
        DD.loc[DD['TipoAtencion'].isin(TipoAtencion_Repetida),'TipoAtencionC'] = 'ConsultaRepetida'
        DD.loc[DD['TipoAtencion'].isin(TipoAtencion_NuevaAPS),'TipoAtencionC'] = 'ConsultaNuevaAPS'
        #DD.loc[DD['TipoAtencion'].isin(TipoAtencion_Otros),'TipoAtencionC'] = 'Otros'

        ## Guardar menos grupos en Prevision

        # Lista de grupos de prevision a mantener
        lista_Prevision = ['Grupo A','Grupo B','Grupo C','Grupo D','PREVISION PROVISORIA']

        # crear variable de clasificacion de prevision
        DD['PrevisionC'] = DD['Prevision']

        # eliminar opciones en Prevision, a OTROS
        DD.loc[np.logical_not(DD['Prevision'].isin(lista_Prevision)),'PrevisionC'] = 'OTRO'

        ## Guardar menos grupos en EstadoCita
        ## Quedan Atendido, No Atendido, Cancelado, y OTRO

        # Listas de grupos de EstadoCita de interes
        lista_ECita = ['Atendido','Alta','No Atendido','Cancelado']

        # crear variable de clasificacion de EstadoCita
        DD['ECitaC'] = DD['EstadoCita']

        # los EstadoCita que no esten en la lista, son OTRO
        DD.loc[np.logical_not(DD['EstadoCita'].isin(lista_ECita)),'ECitaC'] = 'OTRO'
        # las altas son citas antendidas
        DD.loc[DD['EstadoCita']=='Alta','ECitaC'] = 'Atendido'

        logger.info(DD['ECitaC'].value_counts())

        ## Crear clasificacion de Comunas 
        ## Dejo solo las que concentran el 80% de las citas. El resto son Otras_RM y Otras_Regiones
        ## El 80% es arbitrario, se podria hacer de otra forma

        # Lista de comunas de la Region Metropolitana
        Comunas_RM = ['Santiago', 'Cerrillos','Cerro Navia','Conchalí','El Bosque','Estación Central',
                    'Huechuraba','Independencia','La Cisterna','La Florida','La Granja','La Pintana',
                    'La Reina','Las Condes','Lo Barnechea','Lo Espejo','Lo Prado','Macul','Maipú',
                    'Ñuñoa','Peñaflor','Pedro Aguirre Cerda','Peñalolén','Providencia','Pudahuel','Quilicura',
                    'Quinta Normal','Recoleta','Renca','San Joaquín','San Miguel','San Ramón','Vitacura',
                    'Puente Alto','Pirque','San José de Maipo','Colina','Lampa','Tiltil','San Bernardo',
                    'Buin','Calera de Tango','Paine','Melipilla','Alhué','Curacaví','María Pinto',
                    'San Pedro','Talagante','El Monte','Isla de Maipo','Padre Hurtado']

        #map(unicode,Comunas_RM)

        # Lista de comunas de mayor publico junto con las nuevas clases
        ## creamos diccionario de tablas con frecuencias, porcentajes, y porcentajes acumulados por variable
        ## con los datos corregidos
        frecuencias2 = {}
        for i in DD.columns: 
            serie = DD[i].value_counts()
            serie_acum = serie.cumsum()
            serie_porc = serie/serie_acum.iat[-1]*100
            serie_porc_acum = serie_acum/serie_acum.iat[-1]*100
            df = pd.concat([serie, serie_porc, serie_porc_acum],axis=1)
            df.index.name = i
            df.columns = ['frecuencia','%','% acumumado']
            frecuencias2[i]=df
            #print frecuencias2[i].iloc[0:10,:]
            #print ' ' 

        # frequency table cutting presitge and whether or not someone was admitted
        #print pd.crosstab(df['admit'], df['prestige'], rownames=['admit'])
            
        # cosa.info()
        # drop_duplicate

        logger.info('Tamaño base de datos : {}'.format(str(DD.shape)))
        Comunas_80 = frecuencias2['Comuna'][frecuencias2['Comuna']['% acumumado']<80.5].index.tolist()
        Comunas_80.append(u'Otras_Regiones')
        Comunas_80.append(u'Otras_RM')

        # crear nueva variable ComunaC
        DD['ComunaC'] = DD['Comuna']

        DD.loc[DD['Comuna'].isin(Comunas_RM) & 
            np.logical_not(DD['Comuna'].isin(Comunas_80)),'ComunaC'] = 'Otras_RM'
        DD.loc[np.logical_not(DD['Comuna'].isin(Comunas_RM)) & 
            np.logical_not(DD['Comuna'].isin(Comunas_80)),'ComunaC'] = 'Otras_Regiones'

        logger.info(DD['ComunaC'].value_counts())

        ## Arreglar Especialidades

        # Mezclar Infantil con adulto
        DD['Especialidad'] = DD['Especialidad'].str.replace(" Infantil", "")

        # Eliminar No Mencionada
        DD.loc[(DD.Agenda == 'Policlínico de Yeso')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Traumatología'
        DD.loc[(DD.Agenda == 'Policlínico  Cardiología')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Cardiología'
        DD.loc[(DD.Agenda == 'Policlínico de Medula Osea')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Oncología'
        DD.loc[(DD.Agenda == 'Policlínico de Oncología')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Oncología'
        DD.loc[(DD.Agenda == 'Policlinico de Urologia')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Urología'
        DD.loc[(DD.Agenda == 'Policlínico Cirugía')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Cirugía'
        DD.loc[(DD.Agenda == 'Policlínico de Traumatología')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Traumatología'
        DD.loc[(DD.Agenda == 'Policlinico de Neurologia')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Neurología'
        DD.loc[(DD.Agenda == 'Policlínico de Psiquiatría')&(DD.Especialidad == 'No Mencionada'),'Especialidad'] = 'Psiquiatría'


        '''
        # guardado como csv
        DD.to_csv('Base_procesada_no_dummies_medica.csv', sep='\t', encoding='utf-8',index=True)


        DD['Especialidad'].value_counts()'''


        logger.info(DD['Especialidad'].value_counts())

        ECita_total = historial_grupo(DD,'ECitaC',['PAID'])
        ECita_total.columns = [str(x)+'_total' for x in ECita_total.columns]

        ECita_especialidad = historial_grupo(DD,'ECitaC',['PAID','Especialidad'])
        ECita_especialidad.columns = [str(x)+'_esp' for x in ECita_especialidad.columns]

        #ECita_especialidad_tp = historial_grupo(DD,'ECitaC',['PAID','Especialidad','TipoPrestacionC'])
        #ECita_especialidad_tp.columns = [str(x)+'_esp_tp' for x in ECita_especialidad_tp.columns]

        logger.info( ECita_total.columns)

        DD_conc = pd.concat([DD, ECita_total, ECita_especialidad],axis=1)#, ECita_especialidad_tp],axis=1) 

        logger.info(DD_conc.columns)

        ## sacar una base de datos mas chica para la regresion logistica

        # lista de variables que quiero eliminar
        vars_a_eliminar = ['Edad','PAID_esp','PAID_total','Especialidad_esp','PAID_esp_tp','Especialidad_esp_tp']
                            #'Comuna','Sexo','Hora','Dia','Mes',
                            #u'Año','Prevision','Plan','TipoProfesional',
                            #'RutProfesional','Agenda','TipoProfesional','FechaNac'
                            #'Especialidad','TipoAtencion','Prestacion','AgendadoPor',
                            #'MotivoC','MotivoNA','MotivoS','Prevision','Plan',
                            #'Hora','Dia','Mes']

        # variables a eliminar que si estan en el conjunto de variables
        vars_a_eliminar_final = list(set(vars_a_eliminar) & set(DD_conc.columns))

        # eliminado de las variables
        DD_conc.drop(vars_a_eliminar_final,axis=1,inplace=True)

        logger.info( 'Base modificada armada'     )       

        # lista de tablas con frecuencias, porcentajes, y porcentajes acumulados por variable

        Base_analizar = DD_conc

        Variables = Base_analizar.columns

        frecuencias = {}

        for i in Variables:
            # hacer la serie de frecuencias para cada variable
            serie = Base_analizar[i].value_counts()
            # calcular el porcentaje acumulado relativo a los no vacios
            serie_acum = serie.cumsum()
            serie_porc = serie/serie_acum.iat[-1]*100
            serie_porc_acum = serie_acum/serie_acum.iat[-1]*100
            df = pd.concat([serie, serie_porc, serie_porc_acum],axis=1)
            df.index.name = i
            df.columns = ['frecuencia','%','% acumumado']
            frecuencias[i] = df
            
            logger.info( frecuencias[i].iloc[0:10,:])
            logger.info( ' ' )


        

        ## Calcular numero de visitas de cada paciente por especialidad y tipo de prestacion

        aux = DD.groupby(['PAID','Especialidad','TipoPrestacionC'],as_index=False)
        #aux = DD.groupby(['PAID','Agenda','TipoPrestacionC'],as_index=False)

        Visitas_paciente_especialidad = aux.size()
        logger.info(Visitas_paciente_especialidad)
        logger.info(DD.loc[DD['Especialidad']=='No Mencionada']['Prestacion'].value_counts())
        logger.info(DD_conc.shape)
        logger.info(DD_conc.loc[DD_conc['Anyo']==2017,'Mes'].value_counts())
        self.DD_conc = DD_conc

    def write(self,location):
        


        self.DD_conc.to_csv(location, encoding='utf-8',index=False)

        logger.info("database saved")


        