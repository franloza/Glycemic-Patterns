from math import floor
from calendar import day_name

from .Translator import Translator

class Rule:
    """Class that represents a rule composed of a feature, an operator and a threshold value"""

    def __init__(self, feature, operator,threshold, translator=Translator()):

        """Initializer for Rule"""
        if not feature:
            raise ValueError("feature cannot be empty")
        if operator not in ['>', '<', '>=', '<=', '=']:
            raise ValueError("operator is not valid")
        if threshold < 0:
            raise ValueError("threshold must be greater or equal than 0")

        self.feature = feature
        self.operator = operator
        self.threshold = threshold
        self.__translator = translator

    def __str__(self):
        if self.is_boolean():
            suffix = '_T' if self.operator == '>' else '_F'
            rule_list = self.__translator.translate_to_language([self.feature + suffix])
        else:
            rule_list = self.__translator.translate_to_language([self.feature, self.operator])
            if self.is_weekday():
                if isinstance(self.threshold, float):
                    self.threshold =  self.__translator.translate_to_language([str(day_name[floor(self.threshold) - 1])])[0]
                rule_list.append(str(self.threshold))
            elif self.is_hour():
                rule_list.append('{:d}:00'.format(floor(self.threshold)))
            else:
                rule_list.append('{:.4g}'.format(self.threshold))
        return " ".join(rule_list)

    def is_boolean(self):
        """ Method that returns if the feature of the rule can be expressed as either true or false
        :return: True if the feature is boolean
        """
        return self.feature == 'Overlapped_Block'

    def is_weekday(self):
        """ Method that returns if the feature of the rule is a day of the week (1-7)
        :return: True if the feature is a week day
        """
        return self.feature == 'Weekday'

    def is_hour(self):
        """ Method that returns if the feature of the rule is a hour of the day (0-24)
        :return: True if the feature is an hour
        """
        return self.feature in ['Hour', 'Last_Meal_Hour']