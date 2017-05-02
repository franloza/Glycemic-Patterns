from bidict import bidict


class Translator:
    """Class in charge of translating several languages to column names and vice versa"""

    """ Dictionaries """
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
        'Hora del día': 'Hour',
        'Día (Definido por bloques)': 'Day_Block',
        'Día y hora de la última comida': 'Last_Meal',
        'Bloque solapado': 'Overlapped_Block',
        'El bloque está solapado': 'Overlapped_Block_T',
        'El bloque no está solapado': 'Overlapped_Block_F',
        'Variabilidad glucémica (MAGE)': 'MAGE',
        'Día de la semana': 'Weekday',
        'Minutos transcurridos desde la última comida': 'Minutes_Last_Meal',
        'Hora de la última comida': 'Last_Meal_Hour',
        'Media del nivel de glucosa del bloque': 'Glucose_Mean_Block',
        'Desviación estandar del nivel de glucosa del bloque': 'Glucose_Std_Block',
        'Nivel de glucosa mínimo del bloque': 'Glucose_Min_Block',
        'Nivel de glucosa máximo del bloque': 'Glucose_Max_Block',
        'Insulina de acción rápida (unidades) del bloque': 'Rapid_Insulin_Block',
        'Carbohidratos (raciones) del bloque': 'Carbo_Block_U',
        'Carbohidratos (gramos) del bloque ': 'Carbo_Block_G',
        'Media del nivel de glucosa del bloque anterior': 'Glucose_Mean_Prev_Block',
        'Desviación estandar del nivel de glucosa del bloque anterior': 'Glucose_Std_Prev_Block',
        'Nivel de glucosa mínimo del bloque anterior': 'Glucose_Min_Prev_Block',
        'Nivel de glucosa máximo del bloque anterior': 'Glucose_Max_Prev_Block',
        'Insulina de acción rápida (unidades) del bloque anterior': 'Rapid_Insulin_Prev_Block',
        'Carbohidratos (raciones) del bloque anterior': 'Carbo_Prev_Block_U',
        'Carbohidratos (gramos) del bloque anterior': 'Carbo_Prev_Block_G',
        'Media del nivel de glucosa del día': 'Glucose_Mean_Day',
        'Desviación estandar del nivel de glucosa del día': 'Glucose_Std_Day',
        'Nivel de glucosa mínimo del día': 'Glucose_Min_Day',
        'Nivel de glucosa máximo del día': 'Glucose_Max_Day',
        'Media del nivel de glucosa del día anterior': 'Glucose_Mean_Prev_Day',
        'Desviación estandar del nivel de glucosa del día anterior': 'Glucose_Std_Prev_Day',
        'Nivel de glucosa mínimo del día anterior': 'Glucose_Min_Prev_Day',
        'Nivel de glucosa máximo del día anterior': 'Glucose_Max_Prev_Day',
        'Nivel de glucosa 24 horas antes': 'Glucose_Auto_Prev_Day',
        'Diferencia del nivel de glucosa actual con 24 horas antes': 'Delta_Glucose_Prev_Day',
        'Variabilidad glucémica (MAGE) del día anterior': 'MAGE_Prev_Day',
        'Diagnóstico de hiperglucemia en siguiente bloque': 'Hyperglycemia_Diagnosis_Next_Block',
        'Diagnóstico de hipoglucemia en siguiente bloque': 'Hypoglycemia_Diagnosis_Next_Block',
        'Diagnóstico en rango en siguiente bloque': 'In_Range_Diagnosis_Next_Block',
        'Diagnóstico de hiperglucemia severa en siguiente bloque': 'Severe_Hyperglycemia_Diagnosis_Next_Block',
        'Patrón': 'Pattern',
        'Reglas': 'Rules',
        'Muestras': 'Samples',
        'Impureza': 'Impurity',
        'Número de muestras positivas': 'Number_Pos',
        'Número de muestras negativas': 'Number_Neg',
        'Hiperglucemia': 'Hyperglycemia',
        'Hipoglucemia': 'Hypoglycemia',
        'Hiperglucemia severa': 'Severe_Hyperglycemia',
        'Patrones de hiperglucemia': 'Hyperglycemia_Patterns',
        'Patrones de hipoglucemia': 'Hypoglycemia_Patterns',
        'Patrones de hiperglucemia severa': 'Severe_Hyperglycemia_Patterns',
        'Informe de patrones': 'Pattern_Report',
        'Información de bloques': 'Blocks_Information',
        'Resumen de los valores de glucosa del día': 'Day_Summary',
        'Árboles de decisión': 'Decision_Trees',
        'Media': 'Mean',
        'Desviación típica': 'Std',
        'Máximo': 'Max',
        'Mínimo': 'Min',
        'Estadísticas de niveles de glucosa': 'Glucose_Stats',
        'es menor que': '<',
        'es mayor que': '>',
        'es igual que': '=',
        'es mayor o igual que': '>=',
        'es menor o igual que': '<=',
        'y': 'and',
        'Lunes': 'Monday',
        'Martes': 'Tuesday',
        'Miércoles': 'Wednesday',
        'Jueves': 'Thursday',
        'Viernes': 'Friday',
        'Sábado': 'Saturday',
        'Domingo': 'Sunday',
        # Warning translation
        'Avisos': 'Warnings',
        'El número de registros de carbohidratos (Tipo 5) es menor de uno por día. Los patrones podrían no ser precisos': 'W0001',
        'No se han podido extraer patrones de hiperglucemia': 'W0011',
        'No se han podido extraer patrones de hipoglucemia': 'W0012',
        'No se han podido extraer patrones de hiperglucemia severa': 'W0013',
    })

    __en_to_column_dict = bidict({
        'Time': 'Datetime',
        'Register type': 'Register_Type',
        'Historical glucose (mg/dL)': 'Glucose_Auto',
        'Read glucose (mg/dL)': 'Glucose_Manual',
        'Rapid-acting insulin with no numeric value': 'Rapid_Insulin_No_Val',
        'Rapid-acting insulin (units)': 'Rapid_Insulin',
        'Food with no numeric value': 'Carbo_No_Val',
        'Carbohydrates (portions)': 'Carbo_U',
        'Carbohydrates (grams)': 'Carbo_G',
        'Slow-acting insulin with no numeric value': 'Long_Insulin_No_Val',
        'Slow-acting insulin (units)': 'Long_Insulin',
        'Block': 'Block',
        'Hour': 'Hour',
        'Day (Framed by blocks)': 'Day_Block',
        'Day and time of the last meal': 'Last_Meal',
        'Overlapped Block': 'Overlapped_Block',
        'The block is overlapped': 'Overlapped_Block_T',
        'The block is not overlapped': 'Overlapped_Block_F',
        'Glycemic variability (MAGE)': 'MAGE',
        'Day of the week': 'Weekday',
        'Elapsed minutes since the last meal': 'Minutes_Last_Meal',
        'Hour of the last meal': 'Last_Meal_Hour',
        'Mean level of glucose of the block': 'Glucose_Mean_Block',
        'Standard deviation of the level of glucose of the block': 'Glucose_Std_Block',
        'Minimum level of glucose of the block': 'Glucose_Min_Block',
        'Maximum level of glucose of the block': 'Glucose_Max_Block',
        'Rapid-acting insulin (units) of the block': 'Rapid_Insulin_Block',
        'Carbohydrates (portions) of the block': 'Carbo_Block_U',
        'Carbohydrates (blocks) of the block': 'Carbo_Block_G',
        'Mean level of glucose of the previous block': 'Glucose_Mean_Prev_Block',
        'Standard deviation of the level of glucose of the previous block': 'Glucose_Std_Prev_Block',
        'Minimum level of glucose of the previous block': 'Glucose_Min_Prev_Block',
        'Maximum level of glucose of the previous block': 'Glucose_Max_Prev_Block',
        'Rapid-acting insulin (units) of the previous block': 'Rapid_Insulin_Prev_Block',
        'Carbohydrates (portions) of the previous block': 'Carbo_Prev_Block_U',
        'Carbohydrates (blocks) of the previous block': 'Carbo_Prev_Block_G',
        'Mean of the level of glucose of the day': 'Glucose_Mean_Day',
        'Standard deviation of the level of glucose of the day': 'Glucose_Std_Day',
        'Minimum level of glucose of the day': 'Glucose_Min_Day',
        'Maximum level of glucose of the day': 'Glucose_Max_Day',
        'Mean of the level of glucose of the previous day': 'Glucose_Mean_Prev_Day',
        'Standard deviation of the level of glucose of the previous day': 'Glucose_Std_Prev_Day',
        'Minimum level of glucose of the previous day': 'Glucose_Min_Prev_Day',
        'Maximum level of glucose of the previous day': 'Glucose_Max_Prev_Day',
        'Glucose level 24 hours before': 'Glucose_Auto_Prev_Day',
        'Glucose level difference with 24 hours before': 'Delta_Glucose_Prev_Day',
        'Glycemic variability (MAGE) of the previous day': 'MAGE_Prev_Day',
        'Hyperglucemia diagnosis for the following block': 'Hyperglycemia_Diagnosis_Next_Block',
        'Hypoglucemia diagnosis for the following block': 'Hypoglycemia_Diagnosis_Next_Block',
        'In-range diagnosis for the following block': 'In_Range_Diagnosis_Next_Block',
        'Severe hyperglucemia diagnosis for the following block': 'Severe_Hyperglycemia_Diagnosis_Next_Block',
        'Pattern': 'Pattern',
        'Rules': 'Rules',
        'Samples': 'Samples',
        'Impurity': 'Impurity',
        'Number of positive samples': 'Number_Pos',
        'Number of negative samples': 'Number_Neg',
        'Hyperglycemia': 'Hyperglycemia',
        'Hypoglycemia': 'Hypoglycemia',
        'Severe hyperglycemia': 'Severe_Hyperglycemia',
        'Hyperglycemia patterns': 'Hyperglycemia_Patterns',
        'Hypoglycemia patterns': 'Hypoglycemia_Patterns',
        'Severe hyperglycemia patterns': 'Severe_Hyperglycemia_Patterns',
        'Report of patterns': 'Pattern_Report',
        'Information of the blocks': 'Blocks_Information',
        'Day summary of glucose values': 'Day_Summary',
        'Decision trees': 'Decision_Trees',
        'Mean': 'Mean',
        'Standard deviation': 'Std',
        'Maximum': 'Max',
        'Minimum': 'Min',
        'Glucose level statistics': 'Glucose_Stats',
        'is lower than': '<',
        'is greater than': '>',
        'is equal to': '=',
        'is greater or equal than': '>=',
        'is lower or equal than': '<=',
        'and': 'and',
        'Monday': 'Monday',
        'Tuesday': 'Tuesday',
        'Wednesday': 'Wednesday',
        'Thursday': 'Thursday',
        'Friday': 'Friday',
        'Saturday': 'Saturday',
        'Sunday': 'Sunday',
        # Warning translation
        'Warnings': 'Warnings',
        'The number of carbohydrate registers (Type 5) is lower than one per day. The patterns may not be precise': 'W0001',
        'Hyperglycemia patterns were been able to be extracted': 'W0011',
        'Hypoglycemia patterns were been able to be extracted': 'W0012',
        'Severe hyperglycemia patterns were been able to be extracted': 'W0013',
    })

    _dict_mapper = {
        'es': __es_to_column_dict,
        'en': __en_to_column_dict
    }

    def __init__(self, language="es"):
        """Initializer for Translator"""
        self.language = language
        self.__dictionary = Translator._dict_mapper.get(language)

    @property
    def language(self):
        return self.language

    @language.setter
    def language(self, language):
        self.language = language
        self.__dictionary = Translator._dict_mapper.get(language)

    def translate_to_language(self, terms):
        translated_terms = []
        for term in terms:
            try:
                translated_terms.append(self.__dictionary.inv.get(term))
            except KeyError:
                translated_terms.append(term)
        return translated_terms

    def translate_to_column(self, terms):
        translated_terms = []
        for term in terms:
            try:
                translated_terms.append(self.__dictionary.get(term))
            except KeyError:
                translated_terms.append(term)
        return translated_terms
