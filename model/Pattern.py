from model.Translator import Translator
from model.Rule import Rule

class Pattern:
    """Class that represents a pattern composed of a set of rules, number of samples, impurity and number of positive
    and negative examples"""

    def __init__(self, rules, samples, impurity, number_pos, number_neg, translator=Translator()):

        """Constructor for Pattern"""
        if not all(isinstance(item, Rule) for item in rules):
            TypeError("rules must be composed of Rule objects")
        if samples < 0:
            raise ValueError("number of samples must be positive")
        if impurity < 0 or impurity > 1:
            raise ValueError("impurity must be in a range of [0,1]")

        self.rules = rules
        self.samples = samples
        self.impurity = impurity
        self.number_pos = number_pos
        self.number_neg = number_neg

        self.__compacted_rules = rules
        self.__translator = translator

    def __str__(self):
        terms = self.__translator.translate_to_language(['Rules', 'Samples', 'Impurity', 'Number_Pos', 'Number_Neg'])
        pattern_str = '{0}:\n'.format(terms[0])
        for rule in self.__compacted_rules:
            pattern_str += '\t{0}\n'.format(rule)
        pattern_str += '{:s}: {:.4g}\n'.format(terms[1], self.samples)
        pattern_str += '{:s}: {:.4g}\n'.format(terms[2], self.impurity)
        pattern_str += '{:s}: {:.4g}\n'.format(terms[3], self.number_pos)
        pattern_str += '{:s}: {:.4g}\n'.format(terms[4], self.number_neg)
        return pattern_str