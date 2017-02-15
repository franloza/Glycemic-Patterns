import numpy as np
import datetime
import pandas as pd


def define_blocks(data):
    # Number of hours from the middle of the time window to the beginning and to the end
    pre_time_interval = 2
    post_time_interval = 4

    # Get information of automatic measurement of glucose
    auto_gluc_blocks = data[data["Register_Type"] == 0].copy()

    # Get information of carbohydrates
    carbo_data = data[data[data.columns[1]] == 5].copy()
    carbo_datetimes = carbo_data.drop(data.columns[[6, 7]], axis=1)
    dates = carbo_datetimes["Datetime"].dt.date.unique()

    # TODO: Add insulin data and carbo units
    # TODO: Comment better the function

    auto_gluc_blocks.loc[:, "Block"] = 0
    auto_gluc_blocks.loc[:, "Day_Block"] = np.nan
    auto_gluc_blocks.loc[:, "Overlapped_Block"] = False
    for date_i in dates:
        carbo_day = carbo_datetimes[carbo_data["Datetime"].dt.date == date_i]
        block_idx = 1
        for mid_block in carbo_day["Datetime"]:
            block_time_window = pd.Series((mid_block - datetime.timedelta(hours=int(pre_time_interval))
                                           + datetime.timedelta(minutes=x)
                                           for x in range(0, int((pre_time_interval + post_time_interval) * 60))))
            # Define block if empty
            auto_gluc_blocks.loc[
                (auto_gluc_blocks["Datetime"].isin(block_time_window))
                & (auto_gluc_blocks["Block"] == 0), ["Block",  "Day_Block"]] \
                = [block_idx, mid_block.date()]

            # Overlapped blocks
            auto_gluc_blocks.loc[
                (auto_gluc_blocks["Datetime"].isin(block_time_window)) & (auto_gluc_blocks["Block"] != 0)
                & (auto_gluc_blocks["Block"] != block_idx) & (
                    auto_gluc_blocks["Day_Block"] == mid_block.date()), "Overlapped_Block"] = True
            overlapped = auto_gluc_blocks[
                (auto_gluc_blocks["Datetime"].isin(block_time_window)) & (auto_gluc_blocks["Block"] != 0)
                & (auto_gluc_blocks["Block"] != block_idx) & (auto_gluc_blocks["Day_Block"] == mid_block.date())].copy()
            overlapped.loc[:, "Block"] = block_idx
            auto_gluc_blocks = auto_gluc_blocks.append(overlapped, ignore_index=True)
            block_idx += 1

    auto_gluc_blocks.loc[auto_gluc_blocks["Day_Block"].isnull(), "Day_Block"] = \
        auto_gluc_blocks["Datetime"].dt.date

    auto_gluc_blocks.sort_values(by=["Datetime", "Block"], inplace=True)

    # Delete register type column
    auto_gluc_blocks = auto_gluc_blocks.drop("Register_Type", axis=1)

    return auto_gluc_blocks
