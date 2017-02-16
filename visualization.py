import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np


def plot_blocks(data, init_day, translator, end_day=None):

    if end_day == None:
        end_day = init_day

    # Check if it is only one day
    if init_day == end_day:
        one_day = True
    else:
        one_day = False

    # Convert init_day and end_day to datetime.day
    if not isinstance(init_day, datetime.date):
        if isinstance(init_day, datetime):
            init_day = init_day.date()
        else:
            raise TypeError("init_day must be a datetime object")

    if not isinstance(end_day, datetime.date):
        if isinstance(end_day, datetime):
            end_day = end_day.date()
        else:
            raise TypeError("end_day must be a datetime object")

    # Get sample from init_datetime to end_datetime
    auto_gluc_blocks_sample = data[(data["Day_Block"] >= init_day) & (data["Day_Block"] <= end_day)]

    # Smooth glucose data
    smoothed_sample = smooth_plot(auto_gluc_blocks_sample)

    # Generate figure
    fig, ax = plt.subplots()
    labels = []
    for key, grp in smoothed_sample.groupby(['Block', 'Day_Block']):
        ax = grp.plot(ax=ax, kind='line', x="Datetime", y="Glucose_Auto")
        if one_day:
            labels.append("Block " + str(key[0]))
        else:
            labels.append("Block {:d} ({:%d/%m}) ".format(key[0], key[1]))
    lines, _ = ax.get_legend_handles_labels()
    if one_day:
        ax.legend(lines, labels, loc='best')
    else:
        ax.legend(lines, labels)
    plt.show()


def smooth_plot(data):
    # Level of granularity of time axis (Time difference between points)
    interval_min = 1

    # Define bounds of the plot
    min_time = data["Datetime"].min()
    max_time = data["Datetime"].max()

    difference = (max_time - min_time + datetime.timedelta(minutes=1))
    min_diff = (difference.days * 24 * 60) + (difference.seconds / 60)

    smoothed_data = pd.DataFrame((min_time + datetime.timedelta(minutes=x * interval_min)
                                  for x in range(0, int((min_diff / interval_min)))), columns=["Datetime"])
    smoothed_data = pd.merge(smoothed_data, data, how='left', on="Datetime")

    # Cubic spline interpolation
    smoothed_data["Glucose_Auto"] = smoothed_data["Glucose_Auto"].interpolate(method='cubic')

    # Propagate Block and Day_Block data
    smoothed_data.fillna(method='pad', inplace=True, downcast='infer')

    return smoothed_data
