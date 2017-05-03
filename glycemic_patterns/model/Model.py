import datetime
import errno
import os
import uuid
import logging
import time
import pandas as pd

from collections import namedtuple, OrderedDict
from os.path import join
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from .. import visualization as vis
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

        # Read and preprocess every data file, initializing raw and main dataset
        self._process_data(file_paths)

        self._hyper_dt = None
        self._hypo_dt = None
        self._severe_dt = None

        self.logger.info('Setting the metadata')
        if metadata is None:
            self.metadata = dict()
        else:
            self.metadata = metadata

        # Add initial and end dates to metadata
        self.metadata["Init_Date"] = self._base_dataset.iloc[0]['Datetime']
        self.metadata["End_Date"] = self._base_dataset.iloc[-1]['Datetime']
        self.logger.debug('metadata: {}: '.format(str(self.metadata)))

    @property
    def language(self):
        return self._translator.language

    @language.setter
    def language(self, language):
        self._translator.language = language
        self._hyper_dt.translator = self._translator
        self._hyper_dt.translator = self._translator
        self._hypo_dt.translator = self._translator
        self._severe_dt.translator = self._translator

    def fit(self, features=None):
        """ Create and fit the decision trees used to extract the patterns """

        [data, labels] = pp.prepare_to_decision_trees(self._extended_dataset, features)
        start_time = time.time()
        self._hyper_dt = DecisionTree(data, labels["Hyperglycemia_Diagnosis_Next_Block"])
        self._hypo_dt = DecisionTree(data, labels["Hypoglycemia_Diagnosis_Next_Block"])
        self._severe_dt = DecisionTree(data, labels["Severe_Hyperglycemia_Diagnosis_Next_Block"])
        self.logger.debug('Time ellapsed fitting the model: {:.4f}'.format(time.time() - start_time))

    def generate_report(self, max_impurity=0.3, min_sample_size=0, format="pdf", to_file=True, output_path='',
                        block_info=True, language=None):

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

        if language is not None:
            self.language = language

        subtitles = self._translator.translate_to_language(['Hyperglycemia_Patterns', 'Hypoglycemia_Patterns',
                                                            'Severe_Hyperglycemia_Patterns', 'Pattern',
                                                            'Pattern_Report',
                                                            'Decision_Trees', 'Hyperglycemia', 'Hypoglycemia',
                                                            'Severe_Hyperglycemia', 'Blocks_Information',
                                                            'Day_Summary'])

        terms = self._translator.translate_to_language(['Samples', 'Impurity', 'Number_Pos', 'Number_Neg'])

        template_vars["pattern_title"] = subtitles[3]
        template_vars["report_title"] = subtitles[4]
        template_vars["decision_trees_title"] = subtitles[5]
        template_vars["hyper_dt_title"] = subtitles[6]
        template_vars["hypo_dt_title"] = subtitles[7]
        template_vars["severe_dt_title"] = subtitles[8]
        template_vars["blocks_title"] = subtitles[9]
        template_vars["day_summary_title"] = subtitles[10]

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
            self.logger.warning("W0011: {0}. {1}".format(subtitles[0], str(e)))
            self._warnings.append("W0011")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[0], str(e)))

        # Hypoglycemia patterns
        try:
            patterns = self._hypo_dt.get_patterns(max_impurity=max_impurity, min_sample_size=min_sample_size)
            if patterns:
                template_vars["hypoglycemia_patterns_title"] = subtitles[1]
                template_vars["hypoglycemia_patterns"] = patterns
        except ValueError as e:
            self.logger.warning("W0012: {0}. {1}".format(subtitles[1], str(e)))
            self._warnings.append("W0012")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[1], str(e)))

        # Severe Hyperglycemia patterns
        try:
            patterns = self._severe_dt.get_patterns(max_impurity=max_impurity, min_sample_size=min_sample_size)
            if patterns:
                template_vars["severe_hyperglycemia_patterns_title"] = subtitles[2]
                template_vars["severe_hyperglycemia_patterns"] = patterns
        except ValueError as e:
            self.logger.warning("W0013: {0}. {1}".format(subtitles[2], str(e)))
            self._warnings.append("W0012")
        except Exception as e:
            raise Exception('{0} : {1}'.format(subtitles[2], str(e)))

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

        hyper_dt_graph_path = join(output_path, 'Hyperglycemia_Tree.png')
        hypo_dt_graph_path = join(output_path, 'Hypoglycemia_Tree.png')
        severe_dt_graph_path = join(output_path, 'Severe_Hyperglycemia_Tree.png')

        self._hyper_dt.graph.write_png(hyper_dt_graph_path)
        self._hypo_dt.graph.write_png(hypo_dt_graph_path)
        self._severe_dt.graph.write_png(severe_dt_graph_path)

        template_vars["hyper_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(hyper_dt_graph_path))
        template_vars["hypo_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(hypo_dt_graph_path))
        template_vars["severe_dt_graph_path"] = 'file:///{0}'.format(os.path.abspath(severe_dt_graph_path))

        if block_info:
            # Generate graphics of each day
            block_section_data = OrderedDict()
            carbo_column = next(column_name
                                for column_name in self.info_blocks.columns
                                if column_name in ['Carbo_Block_U', 'Carbo_Block_G'])

            BlockInfo = namedtuple('BlockInfo', ['block_num', 'carbo', 'rapid_insulin', 'mean', 'std', 'max', 'min'])
            DayInfo = namedtuple('DayInfo', ['day', 'plot_path', 'block_data', 'mean', 'std', 'max', 'min', 'mage'])

            # Iterate over the days
            for day in self.info_blocks["Day_Block"].unique():
                block_data = []

                # Generate the plot
                plot_path = vis.plot_blocks(self._base_dataset, day, self._translator, block_info=self.info_blocks,
                                            to_file=True, output_path=output_path)

                day_block_info = self.info_blocks[self.info_blocks["Day_Block"] == day]

                # Iterate over the blocks
                for index, block in day_block_info.iterrows():
                    block_data.append(BlockInfo(block["Block"], block[carbo_column],
                                                block["Rapid_Insulin_Block"], block["Glucose_Mean_Block"],
                                                block["Glucose_Std_Block"]
                                                , block["Glucose_Max_Block"], block["Glucose_Min_Block"]))
                block_section_data[day] = DayInfo(day, 'file:///{0}'.format(os.path.abspath(plot_path)), block_data,
                                                  day_block_info["Glucose_Mean_Day"].iloc[0],
                                                  day_block_info["Glucose_Std_Day"].iloc[0],
                                                  day_block_info["Glucose_Max_Day"].iloc[0],
                                                  day_block_info["Glucose_Min_Day"].iloc[0],
                                                  day_block_info["MAGE"].iloc[0])

            # Translate labels of blocks section

            template_vars["block_section_data"] = block_section_data

            carbo_label = 'Carbo_{}'.format(carbo_column[-1])

            block_labels = self._translator.translate_to_language(['Block', 'Mean', 'Std', 'Max', 'Min',
                                                                   carbo_label, 'Rapid_Insulin', 'Glucose_Stats'])

            template_vars["block_label"] = block_labels[0]
            template_vars["mean_label"] = block_labels[1]
            template_vars["std_label"] = block_labels[2]
            template_vars["max_label"] = block_labels[3]
            template_vars["min_label"] = block_labels[4]
            template_vars["carbo_label"] = block_labels[5]
            template_vars["rapid_insulin_label"] = block_labels[6]
            template_vars["glucose_stats_label"] = block_labels[7]

            day_labels = self._translator.translate_to_language(['Glucose_Mean_Day', 'Glucose_Std_Prev_Day',
                                                                 'Glucose_Max_Prev_Day', 'Glucose_Min_Prev_Day',
                                                                 'MAGE'])
            template_vars["mean_day_label"] = day_labels[0]
            template_vars["std_day_label"] = day_labels[1]
            template_vars["max_day_label"] = day_labels[2]
            template_vars["min_day_label"] = day_labels[3]
            template_vars["mage_label"] = day_labels[4]

        terms = self._translator.translate_to_language(['Samples', 'Impurity', 'Number_Pos', 'Number_Neg'])

        template_vars["pattern_label"] = subtitles[3]
        template_vars["report_label"] = subtitles[4]
        template_vars["decision_trees_label"] = subtitles[5]
        template_vars["hyper_dt_label"] = subtitles[6]
        template_vars["hypo_dt_label"] = subtitles[7]
        template_vars["severe_dt_label"] = subtitles[8]
        template_vars["samples_label"] = terms[0]
        template_vars["impurity_label"] = terms[1]

        template_vars["language"] = self._translator.language
        template_vars["decision_tree_legend_path"] = 'file://{0}'.format(
            os.path.abspath(os.path.join('templates', 'decision_tree_color_legend_{}.png'.format(language))))

        html_out = template.render(template_vars)

        if format == "pdf":
            if to_file:
                HTML(string=html_out).write_pdf(join(output_path, "{}.pdf".format(title)))
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
        """

        self.logger.info('Data pre-processing started')
        start_time = time.time()
        to_lang = self._translator.translate_to_language
        to_col = self._translator.translate_to_column

        self._base_dataset = pd.DataFrame()
        self._extended_dataset = pd.DataFrame()
        self.info_blocks = pd.DataFrame()

        for index, path in enumerate(file_paths):
            # Load data
            self.logger.info('Reading file {} in path {}'.format(index + 1, path))
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
            except ValueError as e:
                raise DataFormatException(e)

            # Divide in blocks, extend dataset and clean data
            self.logger.info('Defining blocks')
            time.process_time()
            block_data = pp.define_blocks(raw_data)
            ptime = time.process_time()
            self.logger.debug('define_blocks() Process Time: {:.8f}'.format(ptime))
            self.logger.info('Adding features to dataset')

            ptime = time.process_time()
            extended_data = pp.extend_data(block_data)
            ptime_new = time.process_time()
            self.logger.debug('extend_data() Process Time: {:.8f}'.format(ptime_new - ptime))

            cleaned_extended_data = pp.clean_extended_data(extended_data)

            # block_data.to_csv(path_or_buf='block_data.csv')
            # extended_data.to_csv(path_or_buf='extended_data.csv')
            # cleaned_extended_data.to_csv(path_or_buf='cleaned_extended_data.csv')

            # Join meal time
            info_blocks = extended_data[['Day_Block', 'Block', 'Block_Meal',
                                         'Carbo_Block_U', 'Rapid_Insulin_Block',
                                         'Glucose_Mean_Block', 'Glucose_Std_Block', 'Glucose_Max_Block',
                                         'Glucose_Min_Block', 'Glucose_Mean_Day', 'Glucose_Std_Day',
                                         'Glucose_Max_Day', 'Glucose_Min_Day', 'MAGE']].drop_duplicates(
                subset=['Day_Block', 'Block'])

            # Append to raw_data and main dataset
            self._base_dataset = self._base_dataset.append(block_data, ignore_index=True)
            self._extended_dataset = self._extended_dataset.append(cleaned_extended_data, ignore_index=True)
            self.info_blocks = self.info_blocks.append(info_blocks, ignore_index=True)
            self.logger.info("Data file has been preprocessed and appended to main dataset")

        self.logger.info('Data pre-processing finished')
        self.logger.debug('Time process data: {:.4f} seconds'.format(time.time() - start_time))


class DataFormatException(ValueError):
    """Raised when the format of the data file is not the one expected"""
    pass


class NotFittedError(ValueError, AttributeError):
    """Raised when the model decisions trees have not been created"""
    pass
