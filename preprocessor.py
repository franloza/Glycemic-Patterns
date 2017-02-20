import numpy as np
import datetime
import pandas as pd
import peakdetect
from sklearn.preprocessing import LabelBinarizer, Imputer


def define_blocks(data):
    """Function that divides the dataset in blocks by days according to carbohydrate measures. Furthermore, it adds
    insulin values to each block

    :param data: Dataset containing register type for each entry
    :return: Dataset of register type 1 with a block a day_block associated to it and carbohydrates and insuline values
    """

    # Number of hours from the middle of the time window to the beginning and to the end
    pre_time_interval = 2
    post_time_interval = 4

    # Get information of automatic measurement of glucose
    auto_gluc_blocks = data[data["Register_Type"] == 0].drop(
        ["Register_Type", "Glucose_Manual", "Carbo_No_Val", "Carbo",
         "Rapid_Insulin_No_Val", "Rapid_Insulin"], axis=1)

    # Get information of carbohydrates
    carbo_data = data[data["Register_Type"] == 5].drop(["Register_Type", "Glucose_Auto", "Glucose_Manual",
                                                        "Rapid_Insulin_No_Val", "Rapid_Insulin"], axis=1)

    # Merge columns Carbo and Carbo_No_Val into Carbo
    carbo_data['Carbo'] = carbo_data[['Carbo', 'Carbo_No_Val']].fillna(0).sum(axis=1)
    carbo_data = carbo_data.drop(['Carbo_No_Val'], axis=1)

    dates = carbo_data["Datetime"].dt.date.unique()

    # Get information of insulin
    insulin_data = data[data["Register_Type"] == 4].drop(["Register_Type", "Glucose_Auto", "Glucose_Manual",
                                                          "Carbo_No_Val", "Carbo"], axis=1)

    # Merge columns Rapid_Insulin and Rapid Insulin_No_Val into Rapid_Insulin
    insulin_data['Rapid_Insulin'] = insulin_data[['Rapid_Insulin', 'Rapid_Insulin_No_Val']].fillna(0).sum(axis=1)
    insulin_data = insulin_data.drop(['Rapid_Insulin_No_Val'], axis=1)

    # Join insuline and carbohydrates data
    carbo_insulin_data = pd.merge(carbo_data, insulin_data, how='left', on="Datetime").fillna(0)

    # Initialize new columns
    auto_gluc_blocks.loc[:, "Block"] = 0
    auto_gluc_blocks.loc[:, "Day_Block"] = np.nan
    auto_gluc_blocks.loc[:, "Last_Meal"] = np.nan
    auto_gluc_blocks.loc[:, "Overlapped_Block"] = False
    auto_gluc_blocks.loc[:, "Carbo_Block"] = 0
    auto_gluc_blocks.loc[:, "Rapid_Insulin_Block"] = 0
    carbo_insulin_data.loc[:, "Block"] = 0
    carbo_insulin_data.loc[:, "Day_Block"] = np.nan

    # Iterate over days and define their corresponding blocks from 0 to n (Block 0 corresponds to no block)
    for date_i in dates:
        carbo_day = carbo_insulin_data[carbo_insulin_data["Datetime"].dt.date == date_i]
        block_idx = 1
        # Iterate over all the occurences of carbohydrates in a day
        for index, mid_block in carbo_day.iterrows():
            mid_block_datetime = mid_block["Datetime"]
            block_time_window = pd.Series((mid_block_datetime - datetime.timedelta(hours=int(pre_time_interval))
                                           + datetime.timedelta(minutes=x)
                                           for x in range(0, int((pre_time_interval + post_time_interval) * 60))))
            # Define block if empty
            rapid_insulin = mid_block["Rapid_Insulin"]
            carbo = mid_block["Carbo"]
            auto_gluc_blocks.loc[
                (auto_gluc_blocks["Datetime"].isin(block_time_window))
                & (auto_gluc_blocks["Block"] == 0), ["Block", "Day_Block", "Carbo_Block",
                                                     "Rapid_Insulin_Block"]] \
                = [block_idx, mid_block_datetime.date(), carbo, rapid_insulin]

            # Associate block to each insulin and carbohydrates entry to use in overlapped blocks
            mask = carbo_insulin_data["Datetime"] == mid_block_datetime
            carbo_insulin_data.loc[mask, "Block"] = block_idx
            carbo_insulin_data.loc[mask, "Day_Block"] = mid_block_datetime.date()

            # Define and add entry with overlapped block
            overlapped_mask = (auto_gluc_blocks["Datetime"].isin(block_time_window)) \
                              & (auto_gluc_blocks["Block"] != 0) \
                              & (auto_gluc_blocks["Block"] != block_idx) \
                              & (auto_gluc_blocks["Day_Block"] == mid_block_datetime.date())
            auto_gluc_blocks.loc[overlapped_mask, "Overlapped_Block"] = True
            overlapped = auto_gluc_blocks[overlapped_mask].copy()
            overlapped.loc[:, "Block"] = block_idx

            # Add overlapped entry
            auto_gluc_blocks = auto_gluc_blocks.append(overlapped, ignore_index=True)
            block_idx += 1

    # Update time of last meal, insulin and carbo information of the overlapped blocks
    for index, block_data in carbo_insulin_data.iterrows():
        rapid_insulin = block_data["Rapid_Insulin"]
        carbo = block_data["Carbo"]
        mask = ((auto_gluc_blocks["Block"] == block_data["Block"])
                & (auto_gluc_blocks["Day_Block"] == block_data["Day_Block"]))
        auto_gluc_blocks.loc[mask, ["Carbo_Block", "Rapid_Insulin_Block"]] = [carbo, rapid_insulin]
        auto_gluc_blocks.loc[auto_gluc_blocks["Datetime"] >= block_data["Datetime"], "Last_Meal"] =\
            block_data["Datetime"]

    auto_gluc_blocks.loc[auto_gluc_blocks["Day_Block"].isnull(), "Day_Block"] = \
        auto_gluc_blocks["Datetime"].dt.date

    auto_gluc_blocks.sort_values(by=["Datetime", "Block"], inplace=True)

    return auto_gluc_blocks


def mage(data):
    """Function that return the MAGE (Mean Amplitude of Glycemic Excursions) given a dataset

    :param data: Dataset containing entries with Datetime and glucose values (Glucose_Auto)
    :return: MAGE
    """

    # Calculate vector of glucose values
    values = data[["Datetime", "Glucose_Auto"]].reset_index(drop=True)
    vector = values["Glucose_Auto"]

    # Calculate standard deviation of the values
    std = np.std(vector)

    # Calculate peaks of the values (Starting points of excursions) - Indexes and values
    peaks = peakdetect.peakdetect(np.array(vector), lookahead=2, delta=std)
    indexes = []
    peak_values = []
    for posOrNegPeaks in peaks:
        for peak in posOrNegPeaks:
            indexes.append(peak[0])
            peak_values.append((peak[1]))

    # Calculate differences between consecutive peaks
    differences = []
    for first, second in zip(peak_values, peak_values[1:]):
        differences.append(np.abs(first - second))

    # Filter differences greater than standard deviation
    valid_differences = [elem for elem in differences if elem > std]

    # Return MAGE
    if len(valid_differences) == 0:
        MAGE = np.nan
    else:
        MAGE = sum(valid_differences) / len(valid_differences)

    return MAGE


def extend_data(data):
    """
    Function that add columns to the dataset with information about the days and blocks

    :param data: Dataset to be expanded with block and day_block information
    :return: Extended dataset
    """

    # Add block information (Mean, Standard deviation, minimum value and maximum value) for each day
    new_columns = data.groupby(['Block', 'Day_Block']).agg({'Glucose_Auto': [np.mean, np.std, np.min, np.max]})[
        "Glucose_Auto"]
    new_columns.columns = ["Glucose_Mean_Block", "Glucose_Std_Block", "Glucose_Min_Block", "Glucose_Max_Block"]
    new_columns = new_columns.reset_index(level=[0, 1])
    new_data = pd.merge(data, new_columns, on=["Block", "Day_Block"], how='left')

    # Add day information (Mean, Standard deviation, minimum value and maximum value)
    new_columns = new_data.groupby(['Day_Block']).agg({'Glucose_Auto': [np.mean, np.std, np.min, np.max]})[
        "Glucose_Auto"]
    new_columns.columns = ["Glucose_Mean_Day", "Glucose_Std_Day", "Glucose_Min_Day", "Glucose_Max_Day"]
    new_columns = new_columns.reset_index(level=0)
    new_data = pd.merge(new_data, new_columns, on='Day_Block', how='left')

    # Add MAGE
    days = new_data['Day_Block'].unique()
    for day in days:
        new_data.loc[new_data['Day_Block'] == day, "MAGE"] = mage(new_data[new_data['Day_Block'] == day])

    # Add additional information (Weekday and minutes since last meal)
    new_data.loc[:, "Weekday"] = new_data.apply(lambda row: row["Day_Block"].weekday() + 1, axis=1)
    new_data.loc[:, "Minutes_Last_Meal"] = new_data.apply(lambda row: int((row["Datetime"] - row["Last_Meal"])
                                                                          .total_seconds() / 60), axis=1)

    # Add label to each entry (Diagnosis)
    new_data.loc[:, "Diagnosis"] = new_data["Glucose_Auto"].apply(label_map)

    return new_data


def label_map(value):
    """ Function that determines the diagnosis according to the Glucose level of an entry.
    The three possible diagnosis are: Hypoglycemia, hyperglycemia and normal

    :param value: Glucose level
    :return: Diagnosis (String)
    """
    hypoglycemia_threshold = 70
    hyperglycemia_threshold = 180

    if value < hypoglycemia_threshold:
        return 'Hypoglycemia'
    elif value > hyperglycemia_threshold:
        return 'Hyperglycemia'
    else:
        return 'Normal'


def clean_processed_data(data):

    """ Function that handle entries with NaN values produced by its division in blocks with the function
    define_blocks.
    IMPORTANT: It can cause information loss by removing entries. Don't use it to plot information

    :param data: data returned by define_blocks
    :return: cleaned data
    """

    new_data = data.copy()

    # Delete rows with no previous meal (Initial values)
    new_data.dropna(inplace='True', subset=['Last_Meal'])

    return new_data


def clean_extended_data(data):
    """ Function that handle entries with NaN values produced by extending the dataset with the function
    extend_data
    IMPORTANT: It can cause information loss by removing entries. Don't use it to plot information

    :param data: data returned by extended_data
    :return: cleaned data
    """

    new_data = data.copy()

    # Infer entries with no MAGE with mean
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputed_mage = imp.fit_transform(new_data["MAGE"].reshape(-1, 1))
    new_data.loc[:, "MAGE"] = imputed_mage

    return new_data

def prepare_to_decision_trees (data):
    """ Function that returns the data and the labels ready to create a Decision tree
        IMPORTANT: It can cause information loss by removing entries. Don't use it to plot information

        :param data: cleaned extended data
        :return: data and labels to create DecisionTree object
        """

    # Remove columns that cannot be passed to the estimator
    new_data = data.drop(["Datetime", "Day_Block", "Last_Meal"], axis=1)

    # Remove columns that contains information that should not be used to discover patterns.
    # i.e. current glucose level
    new_data.drop("Glucose_Auto", axis=1, inplace=True)

    # Binarize labels in a one-vs-all fashion (Hyperglycemia, Hypoglycemia and Normal)
    # to get binary labels
    lb = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    lb.fit(data["Diagnosis"])
    labels = pd.DataFrame(index=data.index)
    for x in lb.classes_:
        labels[x + "_Diagnosis"] = np.nan
    labels.loc[:, [x + "_Diagnosis" for x in lb.classes_]] = lb.transform(data["Diagnosis"])

    #Delete diagnosis columnn (label)
    new_data.drop("Diagnosis", axis=1, inplace=True)

    return [new_data, labels]