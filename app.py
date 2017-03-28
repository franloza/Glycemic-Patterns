import pandas as pd
import os
import sys

# Add modules path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model.Translator import Translator
from model.DecisionTree import DecisionTree
import preprocessor as pp

# Add modules path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def app(filepath, max_impurity=0, language="es", output_path=None):
    # Define translator functions
    translator = Translator(language)
    to_lang = translator.translate_to_language
    to_col = translator.translate_to_column

    # Load data
    raw_data = pd.read_csv(filepath, header=0, skiprows=1, delimiter="\t", index_col=0, usecols=list(range(0, 9)),
                           parse_dates=to_lang(["Datetime"]), decimal=",",
                           date_parser=lambda x: pd.to_datetime(x, format="%Y/%m/%d %H:%M"))
    # Translate column names
    raw_data.columns = (to_col(raw_data.columns))

    # Divide in blocks, extend dataset and clean data
    block_data = pp.define_blocks(raw_data)
    cleaned_block_data = pp.clean_processed_data(block_data)
    extended_data = pp.extend_data(cleaned_block_data)
    cleaned_extended_data = pp.clean_extended_data(extended_data)

    # Create decision trees
    [data, labels] = pp.prepare_to_decision_trees(cleaned_extended_data)
    hyper_dt = DecisionTree(data, labels["Hyperglycemia_Diagnosis_Next_Block"])
    hypo_dt = DecisionTree(data, labels["Hypoglycemia_Diagnosis_Next_Block"])
    severe_dt = DecisionTree(data, labels["Severe_Hyperglycemia_Diagnosis_Next_Block"])

    # Generate graph images
    if output_path is not None:
        hyper_dt.graph.write_png(output_path + '/Hyperglycemia_Tree.png')
        hypo_dt.graph.write_png(output_path + '/Hypoglycemia_Tree.png')
        severe_dt.graph.write_png(output_path + '/Severe_Hyperglycemia_Tree.png')

    # Print patterns
    terms = translator.translate_to_language(['Hyperglycemia_Patterns', 'Hypoglycemia_Patterns',
                                              'Severe_Hyperglycemia_Patterns'])

    # Hyperglycemia patterns
    patterns = hyper_dt.get_patterns(max_impurity=max_impurity)
    if patterns:
        print('{0}'. format(terms[0].center(50, '=')))
        for pattern in patterns:
            print(pattern, end='\n\n')

    # Hypoglycemia patterns
    patterns = hypo_dt.get_patterns(max_impurity=max_impurity)
    if patterns:
        print('{0}'.format(terms[1].center(50, '=')))
        for pattern in patterns:
            print(pattern, end='\n\n')

    # Severe Hyperglycemia patterns
    patterns = severe_dt.get_patterns(max_impurity=max_impurity)
    if patterns:
        print('{0}'.format(terms[2].center(50, '=')))
        for pattern in patterns:
            print(pattern, end='\n\n')
