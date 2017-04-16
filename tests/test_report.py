#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from glycemic_patterns.model import Translator
from pandas import DataFrame

__author__ = "Fran Lozano"
__copyright__ = "Fran Lozano"
__license__ = "mit"

class TestTranslator(unittest.TestCase):
    def test_column_translation(self):
        translator = Translator(language='es')
        to_lang = translator.translate_to_language
        to_col = translator.translate_to_column

        df = DataFrame(columns=['Hora', 'Tipo de registro', 'Histórico glucosa (mg/dL)',
                                'Glucosa leída (mg/dL)', 'Insulina de acción rápida sin valor numérico',
                                'Insulina de acción rápida (unidades)', 'Alimentos sin valor numérico',
                                'Carbohidratos (raciones)'])

        translated_columns = to_col(df.columns)
        self.assertCountEqual(translated_columns, ['Datetime', 'Register_Type', 'Glucose_Auto', 'Glucose_Manual',
                               'Rapid_Insulin_No_Val', 'Rapid_Insulin', 'Carbo_No_Val', 'Carbo_U'])

        translated_columns = to_lang(translated_columns)
        self.assertCountEqual(translated_columns, df.columns.values)



