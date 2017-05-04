from calendar import day_name
from math import floor

from .Translator import Translator
from .Rule import Rule


class Pattern:
    """Class that represents a pattern composed of a set of rules, number of samples, impurity and number of positive
    and negative examples"""

    def __init__(self, rules, total_pos, total_neg, impurity, sample_size_pos, sample_size_neg, translator=Translator()):

        """Initializer for Pattern"""
        if not all(isinstance(item, Rule) for item in rules):
            TypeError("rules must be composed of Rule objects")
        if sample_size_pos < 0 or sample_size_neg < 0:
            raise ValueError("number of samples must be positive")
        if impurity < 0 or impurity > 1:
            raise ValueError("impurity must be in a range of [0,1]")

        self.total_pos = total_pos
        self.total_neg = total_neg
        self.impurity = impurity
        self.sample_size_pos = sample_size_pos
        self.sample_size_neg = sample_size_neg

        self._compacted_rules = self.__compact_rules(rules, translator)
        self._translator = translator


    def __str__(self):
        terms = self._translator.translate_to_language(['Rules', 'Samples', 'Impurity', 'Number_Pos', 'Number_Neg'])
        pattern_str = '{0}:\n'.format(terms[0])
        for rule in self.rules:
            pattern_str += '\t{0}\n'.format(rule)
        pattern_str += '{:s}: {:.4g} ({:.2%})\n'.format(terms[1], self.sample_size, self.sample_size/self.total_records)
        pattern_str += '{:s}: {:.4g}\n'.format(terms[2], self.impurity)
        pattern_str += '{:s}: {:.4g} ({:.2%})\n'.format(terms[3], self.sample_size_pos, self.total_pos)
        pattern_str += '{:s}: {:.4g} ({:.2%})\n'.format(terms[4], self.sample_size_neg, self.total_neg)
        return pattern_str

    @property
    def total_records(self):
        return self.total_pos + self.total_neg

    @property
    def sample_size(self):
        return self.sample_size_pos + self.sample_size_neg

    @property
    def rules(self):
        return [rule for _,rule in self._compacted_rules.items()]

    @staticmethod
    def __compact_rules(rules, translator):
        compacted_rules = {}
        for rule in rules:
            if rule.feature in compacted_rules:
                stored_rule = compacted_rules[rule.feature]
                if isinstance(stored_rule, _CombinedRule):
                    if rule.operator == '<=':
                        stored_rule.min_threshold = rule.threshold
                    else:
                        stored_rule.max_threshold = rule.threshold
                    compacted_rules[rule.feature] = stored_rule
                elif rule.operator != stored_rule.operator:
                    if rule.operator == '<=':
                        max_threshold = rule.threshold
                        min_threshold = compacted_rules[rule.feature].threshold
                    else:
                        max_threshold = compacted_rules[rule.feature].threshold
                        min_threshold = rule.threshold
                    compacted_rules[rule.feature] = _CombinedRule(rule.feature, min_threshold, max_threshold, translator)
                else:
                    compacted_rules[rule.feature] = rule
            else:
                compacted_rules[rule.feature] = rule
        return compacted_rules


class _CombinedRule:
    """Private class that combine two rules with the same feature"""

    def __init__(self, feature, min_threshold, max_threshold, translator=Translator()):
        self.feature = feature
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.__translator = translator

    def __str__(self):
        rule_list = self.__translator.translate_to_language([self.feature, '>'])
        if self.is_weekday():
            if isinstance(self.min_threshold, float):
                self.min_threshold = self.__translator.translate_to_language([str(day_name[floor(self.min_threshold) - 1])])[0]
                rule_list.append(str(self.min_threshold))
        elif self.is_hour():
            rule_list.append('{:d}:00'.format(floor(self.min_threshold)))
        else:
            rule_list.append('{:.4g}'.format(self.min_threshold))
        rule_list.extend(self.__translator.translate_to_language(['and', '<=']))
        if self.is_weekday():
            if isinstance(self.max_threshold, float):
                self.max_threshold = self.__translator.translate_to_language([str(day_name[floor(self.max_threshold) - 1])])[0]
                rule_list.append(str(self.max_threshold))
        elif self.is_hour():
            rule_list.append('{:d}:00'.format(floor(self.max_threshold)))
        else:
            rule_list.append('{:.4g}'.format(self.max_threshold))
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