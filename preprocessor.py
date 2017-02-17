import numpy as np
import datetime
import pandas as pd


def define_blocks(data):
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
                & (auto_gluc_blocks["Block"] == 0), ["Block", "Day_Block", "Carbo_Block", "Rapid_Insulin_Block"]] \
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

    # Update insulin and carbo information of the overlapped blocks
    for index, block_data in carbo_insulin_data.iterrows():
        rapid_insulin = block_data["Rapid_Insulin"]
        carbo = block_data["Carbo"]
        mask = ((auto_gluc_blocks["Block"] == block_data["Block"]) \
                                            & (auto_gluc_blocks["Day_Block"] == block_data["Day_Block"]))
        auto_gluc_blocks.loc[mask, ["Carbo_Block", "Rapid_Insulin_Block"]] = [carbo, rapid_insulin]

    auto_gluc_blocks.loc[auto_gluc_blocks["Day_Block"].isnull(), "Day_Block"] = \
        auto_gluc_blocks["Datetime"].dt.date

    auto_gluc_blocks.sort_values(by=["Datetime", "Block"], inplace=True)

    return auto_gluc_blocks
