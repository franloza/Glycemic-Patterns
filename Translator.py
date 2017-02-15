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
        'Carbohidratos (raciones)': 'Carbo',
        'Insulina de acción lenta sin valor numérico': 'Long_Insulin_No_Val',
        'Insulina de acción lenta (unidades)': 'Long_Insulin',
    })

    __en_to_column_dict = bidict({
        'Time': 'Datetime',
        'Register type': 'Register_Type'
        # TODO: Complete English dictionary
    })
