from bidict import bidict

class Translator:
    """Class in charge of translating several languages to column names and vice versa"""

    __dictionary = bidict()

    def __init__(self, language="es"):
        """Constructor for Translator"""
        dict_mapper = {
            'es': self.__es_to_column_dict,
            'en': self.__en_to_column_dict
        }
        self.__dictionary = dict_mapper.get(language)

    def translate_to_language(self, terms):
        # print(self.__dictionary.get(terms[0]))
        return list(map((lambda x: self.__dictionary.inv.get(x)), terms))

    def translate_to_column(self, terms):
        # print(self.__dictionary.get(terms[0]))
        return list(map((lambda x: self.__dictionary.get(x)), terms))

    """ Dictionaries """
    # TODO: Migrate to database
    __es_to_column_dict = bidict({
        'Hora': 'Datetime',
        'Tipo de registro': 'Register_Type',
        'Histórico glucosa (mg/dL)': 'Glucose_Auto',
        'Glucosa leída (mg/dL)': 'Glucose_Manual',
        'Insulina de acción rápida sin valor numérico': 'Rapid_Insulin_No_Val',
        'Insulina de acción rápida (unidades)': 'Rapid_Insulin',
        'Alimentos sin valor numérico': 'Carbo_No_Val',
        'Carbohidratos (raciones)': 'Carbo_U',
        'Carbohidratos (gramos)': 'Carbo_G',
        'Insulina de acción lenta sin valor numérico': 'Long_Insulin_No_Val',
        'Insulina de acción lenta (unidades)': 'Long_Insulin',
        'Bloque': 'Block',
        'Hora(0-24)': 'Hour',
        'Día (Definido por bloques)': 'Day_Block',
        'Día y hora de la última comida': 'Last_Meal',
        'Bloque solapado': 'Overlapped_Block',
        'El bloque está solapado': 'Overlapped_Block_T',
        'El bloque no está solapado': 'Overlapped_Block_F',
        'Variabilidad glucémica (MAGE)': 'MAGE',
        'Día de la semana': 'Weekday',
        'Minutos transcurridos desde la última comida':'Minutes_Last_Meal',
        'Hora de la última comida': 'Last_Meal_Hour',
        'Media del nivel de glucosa del bloque anterior':'Glucose_Mean_Prev_Block',
        'Desviación estandar del nivel de glucosa del bloque anterior':'Glucose_Std_Prev_Block',
        'Nivel de glucosa mínimo del bloque anterior':'Glucose_Min_Prev_Block',
        'Nivel de glucosa máximo del bloque anterior':'Glucose_Max_Prev_Block',
        'Insulina de acción rápida (unidades) del bloque anterior':'Rapid_Insulin_Prev_Block',
        'Carbohidratos (raciones) del bloque anterior':'Carbo_Prev_Block_U',
        'Carbohidratos (gramos) del bloque anterior': 'Carbo_Prev_Block_G',
        'Media del nivel de glucosa del día anterior':'Glucose_Mean_Prev_Day',
        'Desviación estandar del nivel de glucosa del día anterior':'Glucose_Std_Prev_Day',
        'Nivel de glucosa mínimo del día anterior':'Glucose_Min_Prev_Day',
        'Nivel de glucosa máximo del día anterior':'Glucose_Max_Prev_Day',
        'Nivel de glucosa 24 horas antes':'Glucose_Auto_Prev_Day',
        'Diferencia del nivel de glucosa actual con 24 horas antes':'Delta_Glucose_Prev_Day',
        'Variabilidad glucémica (MAGE) del día anterior': 'MAGE_Prev_Day',
        'Diagnóstico de hiperglucemia en siguiente bloque':'Hyperglycemia_Diagnosis_Next_Block',
        'Diagnóstico de hipoglucemia en siguiente bloque':'Hypoglycemia_Diagnosis_Next_Block',
        'Diagnóstico en rango en siguiente bloque':'In_Range_Diagnosis_Next_Block',
        'Diagnóstico de hiperglucemia severa en siguiente bloque':'Severe_Hyperglycemia_Diagnosis_Next_Block',
        'Reglas': 'Rules',
        'Muestras': 'Samples',
        'Impureza': 'Impurity',
        'Número de muestras positivas': 'Number_Pos',
        'Número de muestras negativas': 'Number_Neg',
        'Patrones de hiperglucemia': 'Hyperglycemia_Patterns',
        'Patrones de hipoglucemia': 'Hypoglycemia_Patterns',
        'Patrones de hiperglucemia severa': 'Severe_Hyperglycemia_Patterns',
        'es menor que': '<',
        'es mayor que': '>',
        'es igual que': '=',
        'es mayor o igual que': '>=',
        'es menor o igual que': '<=',
        'y': 'and'
        })

    __en_to_column_dict = bidict({
        'Time': 'Datetime',
        'Register type': 'Register_Type'
        # TODO: Complete English dictionary
    })
