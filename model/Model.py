import pandas as pd
from model.Translator import Translator
from model.DecisionTree import DecisionTree
import preprocessor as pp


class Model:
    """Class that contains a model of the data, composed of decision trees for each label and capable of extract 
    patterns """

    def __init__(self, file_paths, max_impurity=0, language="es"):

        """Initializer for Model"""

        # Check if is a list of files or a string
        if not isinstance(file_paths, (list, tuple) and isinstance(file_paths, str)):
            file_paths = [file_paths]
        else:
            raise ValueError('The filepath(s) must be a string or a list of strings')

        # Define translator functions
        self._translator = Translator(language)

        # Read and preprocess every data file
        self._dataset = self._process_data(file_paths)

        self._hyper_dt = None
        self._hypo_dt = None
        self._severe_dt = None

    def fit(self):
        """ Create and fit the decision trees used to extract the patterns """

        [data, labels] = pp.prepare_to_decision_trees(self._dataset)
        self._hyper_dt = DecisionTree(data, labels["Hyperglycemia_Diagnosis_Next_Block"])
        self._hypo_dt = DecisionTree(data, labels["Hypoglycemia_Diagnosis_Next_Block"])
        self._severe_dt = DecisionTree(data, labels["Severe_Hyperglycemia_Diagnosis_Next_Block"])


    def generate_report(self):
        """ Generate a PDF report with the patterns """
        #TODO: Complete method. See http://pbpython.com/pdf-reports.html
        raise NotImplementedError


    def _process_data(self, file_paths):

        """ Read, preprocess and join all the data files specified in file_paths

        :rtype: DataFrame dataset containing all the data files divided in blocks and preprocessed
        """
        to_lang = self._translator.translate_to_language
        to_col = self._translator.translate_to_column

        dataset = pd.DataFrame(file_paths)

        for path in file_paths:
            # Load data
            try:
                raw_data = pd.read_csv(path, header=0, skiprows=1, delimiter="\t", index_col=0,
                                       usecols=list(range(0, 9)),
                                       parse_dates=to_lang(["Datetime"]), decimal=",",
                                       date_parser=lambda x: pd.to_datetime(x, format="%Y/%m/%d %H:%M"))
            except:
                raise Exception("There was an error reading the data file {}".format(path))

            # Translate column names
            raw_data.columns = (to_col(raw_data.columns))

            # Check anomalies in the data
            try:
                pp.check_data(raw_data)
            except Exception as e:
                raise DataFormatException(e)

            # Divide in blocks, extend dataset and clean data
            block_data = pp.define_blocks(raw_data)
            cleaned_block_data = pp.clean_processed_data(block_data)
            extended_data = pp.extend_data(cleaned_block_data)
            cleaned_extended_data = pp.clean_extended_data(extended_data)

            # Append to dataset
            dataset = dataset.append(cleaned_extended_data, ignore_index=True)

        return dataset


class DataFormatException(ValueError):
    """Raised when the format of the data file is not the one expected"""
    pass


class NotFittedError(ValueError, AttributeError):
    """Raised when the model decisions trees have not been created"""
    pass
