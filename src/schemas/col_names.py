from enum import Enum

class HlcmDataColNames(Enum):
    FechaNacColName = 'FechaNac'
    SexoColName = "Sexo"
    ComunaColName = 'Comuna'
    PrevisionColName = 'Prevision'
    EspecialidadColName = 'Especialidad'
    TipoAtencionName = 'TipoAtencion'
    TipoProfesionalColName = 'TipoProfesional'
    CodPrestacionColName = 'CodPrestacion'
    FechaCitaColName = 'FechaCita'
    EstadoCitaColName = 'EstadoCita'
    HoraCitaColName = 'HoraCita'
    
class HlcmNewDataColNames(Enum):
    
    FechaNacColName = 'FechadeNac'
    PrevisionColName = 'Plan'
    SexoColName = "Sexo"
    ComunaColName = 'Comuna'
    EspecialidadColName = 'Especialidad'
    TipoAtencionName = 'TipoAtencionSobreescrita'
    TipoProfesionalColName = 'TipoProfesional'
    CodPrestacionColName = 'CodigoPrestacion'
    FechaCitaColName = 'FechaCita'
    EstadoCitaColName = 'EstadoCita'
    HoraCitaColName = 'HoraCita'
