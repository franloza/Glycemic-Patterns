import datetime
import errno
import os
import uuid
import warnings
import logging
import time
from os.path import join

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from .. import preprocessor as pp
from .DecisionTree import DecisionTree
from .Translator import Translator


class Model:
    """Class that contains a model of the data, composed of decision trees for each label and capable of extract 
    patterns """

    def __init__(self, file_paths, metadata=None, language="es", logger=None):

        """Initializer for Model"""

        # Get logger
        self.logger = logger or logging.getLogger(__name__)

        # Check if is a list of files or a string
        if isinstance(file_paths, (list, tuple)) or isinstance(file_paths, str):
            if isinstance(file_paths, str):
                file_paths = [file_paths]
        else:
            raise ValueError('The filepath(s) must be a string or a list of strings')

        self.logger.info('Setting the translator with language "{}"'.format(language))
        # Define translator functions
        self._translator = Translator(language)

        # Read and preprocess every data file
        self._dataset = self._process_data(file_paths)

        self._hyper_dt = None
        self._hypo_dt = None
        self._severe_dt = None

        self.logger.info('Setting the metadata')
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        # Add initial and end dates to metadata
        self.metadata["Init_Date"] = self._dataset.iloc[0]['Datetime']
        self.metadata["End_Date"] = self._dataset.iloc[-1]['Datetime']
        self.logger.debug('metadata: {}: '.format(str(self.metadata)))

    def fit(self, features=None):
        """ Create and fit the decision trees used to extract the patterns """

        [data, labels] = pp.prepare_to_decision_trees(self._dataset, features)
        self._hyper_dt = DecisionTree(data, labels["Hyperglycemia_Diagnosis_Next_Block"])
        self._hypo_dt = DecisionTree(data, labels["Hypoglycemia_Diagnosis_Next_Block"])
        self._severe_dt = DecisionTree(data, labels["Severe_Hyperglycemia_Diagnosis_Next_Block"])

    def generate_report(self, max_impurity=0.3, min_sample_size=0, format="pdf", to_file=True, output_path=''):
        """ Generate a PDF report with the patterns """

        if self._hyper_dt is None or self._hypo_dt is None or self._severe_dt is None:
            raise NotFittedError("It is necessary to fit the model before generating the report")

        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), '..', 'templates')))
        template = env.get_template("report.html")

        if "Patient_Name" in self.metadata:
            title = '{0}_{1}'.format(self.metadata["Patient_Name"].replace(' ', '_'),
                                     datetime.datetime.now().strftime("%d%m%y_%H%M"))
        else:
            title = 'Report_{}'.format(datetime.datetime.now().strftime("%d%m%y_%H%M"))

        template_vars = {"title": title, "metadata": self.metadata}

        subtitles = self._translator.translate_to_language(['Hyperglycemia_Patterns', 'Hypoglycemia_Patterns',
                                                            'Severe_Hyperglycemia_Patterns', 'Pattern',
                                                            'Pattern_Report',
                                                            'Decision_Trees', 'Hyperglycemia', 'Hypoglycemia',
                                                            'Severe_Hyperglycemia'])

        terms = self._translator.translate_to_language(['Samples', 'Impurity', 'Number_Pos', 'Number_Neg'])

        template_vars["pattern_title"] = subtitles[3]
        template_vars["report_title"] = subtitles[4]
        template_vars["decision_trees_title"] = subtitles[5]
        template_vars["hyper_dt_title"] = subtitles[6]
        template_vars["hypo_dt_title"] = subtitles[7]
        template_vars["severe_dt_title"] = subtitles[8]
        template_vars["samples_title"] = terms[0]
        template_vars["impurity_title"] = terms[1]
        template_vars["number_pos"] = terms[2]
        template_vars["number_neg"] = terms[3]

        # Hyperglycemia patterns
        try:
            patterns = self._hyper_dt.get_patterns(max_impurity=max_impurity, min_sample_size=0)
            if patterns:
                template_vars["hyperglycemia_patterns_title"] = subtitles[0]
                template_vars["hyperglycemia_patterns"] = patterns
        except ValueError as e:
            warnings.warn("W0011: {0}. {1}".format(subtitles[0], str(e)))
            self._warnings.append("W0011")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[0], e))

        # Hypoglycemia patterns
        try:
            patterns = self._hypo_dt.get_patterns(max_impurity=max_impurity, min_sample_size=min_sample_size)
            if patterns:
                template_vars["hypoglycemia_patterns_title"] = subtitles[1]
                template_vars["hypoglycemia_patterns"] = patterns
        except ValueError as e:
            warnings.warn("W0012: {0}. {1}".format(subtitles[1], str(e)))
            self._warnings.append("W0012")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[1], e))

        # Severe Hyperglycemia patterns
        try:
            patterns = self._severe_dt.get_patterns(max_impurity=max_impurity, min_sample_size=min_sample_size)
            if patterns:
                template_vars["severe_hyperglycemia_patterns_title"] = subtitles[2]
                template_vars["severe_hyperglycemia_patterns"] = patterns
        except ValueError as e:
            warnings.warn("W0013: {0}. {1}".format(subtitles[2], str(e)))
            self._warnings.append("W0012")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[2], e))

        # Add warnings
        if self._warnings:
            warning_list = ['Warnings']
            for warning in self._warnings:
                warning_list.append(warning)
            warning_list = self._translator.translate_to_language(warning_list)
            template_vars["warnings_title"] = warning_list.pop(0)
            template_vars["warnings"] = warning_list

        # Generate graph images
        if "UUID" in self.metadata:
            uuid_str = str(self.metadata["UUID"])
        elif "Patient_Name" in self.metadata:
            uuid_str = str(uuid.uuid3(uuid.NAMESPACE_DNS, self.metadata["Patient_Name"]))
        else:
            uuid_str = str(uuid.uuid4())

        output_path = join(output_path, uuid_str)

        try:
            os.makedirs(output_path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

        hyper_dt_graph_path = output_path + '/Hyperglycemia_Tree.png'
        hypo_dt_graph_path = output_path + '/Hypoglycemia_Tree.png'
        severe_dt_graph_path = output_path + '/Severe_Hyperglycemia_Tree.png'

        self._hyper_dt.graph.write_png(hyper_dt_graph_path)
        self._hypo_dt.graph.write_png(hypo_dt_graph_path)
        self._severe_dt.graph.write_png(severe_dt_graph_path)

        template_vars["hyper_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(hyper_dt_graph_path))
        template_vars["hypo_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(hypo_dt_graph_path))
        template_vars["severe_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(severe_dt_graph_path))

        html_out = template.render(template_vars)

        if format == "pdf":
            if to_file:
                HTML(string=html_out).write_pdf(join(output_path,"{}.pdf".format(title)))
            else:
                result = HTML(string=html_out).write_pdf()
        elif format == "html":
            if to_file:
                f = open("{}.html".format(title), 'w')
                f.write(html_out)
                f.close()
            else:
                result = HTML(string=html_out)
        else:
            raise ValueError("File format must be pdf or html")

        if not to_file:
            return result


    def _process_data(self, file_paths):

        """ Read, preprocess and join all the data files specified in file_paths

        :param file_paths: List of strings containing absolute paths to the CSV files
        :param features: Features that will be included in the training dataset
        :return: DataFrame dataset containing all the data files divided in blocks and preprocessed
        """

        self.logger.info('Data pre-processing started')
        to_lang = self._translator.translate_to_language
        to_col = self._translator.translate_to_column

        dataset = pd.DataFrame()

        for path in file_paths:
            # Load data
            self.logger.info('Reading file in path {}'.format(path))
            try:
                raw_data = pd.read_csv(path, header=0, skiprows=1, delimiter="\t", index_col=0,
                                       usecols=list(range(0, 9)),
                                       parse_dates=to_lang(["Datetime"]), decimal=",",
                                       date_parser=lambda x: pd.to_datetime(x, format="%Y/%m/%d %H:%M"))
            except Exception as e:
                raise IOError("There was an error reading the data file {}: {}".format(path, e))

            # Translate column names
            self.logger.debug('Columns data file: {}'.format(str(raw_data.columns.values)))
            raw_data.columns = (to_col(raw_data.columns))
            self.logger.debug('Translated columns: {}'.format(str(raw_data.columns.values)))

            # Check anomalies in the data
            try:
                self.logger.info('Checking data')
                self._warnings = pp.check_data(raw_data)
            except Exception as e:
                raise DataFormatException(e)

            # Divide in blocks, extend dataset and clean data
            time.process_time()
            self.logger.info('Defining blocks')
            block_data = pp.define_blocks(raw_data)
            ptime = time.process_time()
            self.logger.debug('define_blocks Process Time: {}'.format(ptime))
            cleaned_block_data = pp.clean_processed_data(block_data)
            ptime_new = time.process_time()
            self.logger.debug('clean_processed_data Process Time: {}'.format(ptime_new - ptime))
            ptime = ptime_new
            self.logger.info('Adding features to dataset')
            extended_data = pp.extend_data(cleaned_block_data)
            ptime_new = time.process_time()
            self.logger.debug('extend_data Process Time: {}'.format(ptime_new - ptime))
            ptime = ptime_new
            cleaned_extended_data = pp.clean_extended_data(extended_data)
            ptime_new = time.process_time()
            self.logger.debug('extend_data Process Time: {}'.format(ptime_new - ptime))

            # Append to dataset
            dataset = dataset.append(cleaned_extended_data, ignore_index=True)
            self.logger.info("Data file has been preprocessed and appended to main dataset")

        self.logger.info('Data pre-processing finished')

        return dataset


class DataFormatException(ValueError):
    """Raised when the format of the data file is not the one expected"""
    pass


class NotFittedError(ValueError, AttributeError):
    """Raised when the model decisions trees have not been created"""
    pass