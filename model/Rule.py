from model.Translator import Translator

class Rule:
    """Class that represents a rule composed of a feature, an operator and a threshold value"""

    def __init__(self, feature, operator,threshold, translator=Translator()):

        """Constructor for Rule"""
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
        rule_list = self.__translator.translate_to_language([self.feature, self.operator])
        rule_list.append('{:.4f}'.format(self.threshold))
        return " ".join(rule_list)