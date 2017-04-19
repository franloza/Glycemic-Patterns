import matplotlib.pyplot as plt
import datetime
import pandas as pd
import subprocess
import pydot
import numpy as np
from os.path import join

from sklearn.tree import export_graphviz
from io import StringIO


def plot_blocks(data, init_day,translator, block_info=None, end_day=None, to_file=False, output_path=None):
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
    if block_info is not None:
        block_info_sample = block_info[(block_info["Day_Block"] >= init_day) & (block_info["Day_Block"] <= end_day)]

    # Smooth glucose data
    smoothed_sample = smooth_plot(auto_gluc_blocks_sample)

    # Generate figure
    fig, ax = plt.subplots()
    labels = []
    for key, grp in smoothed_sample.groupby(['Block', 'Day_Block']):
        grp_axis = smoothed_sample[smoothed_sample['Day_Block'] == key[1]]
        grp_axis.loc[grp_axis['Block'] != key[0], "Glucose_Auto"] = np.nan
        if one_day:
            label = "{} {:d}".format(translator.translate_to_language(["Block"])[0], key[0])
        else:
            label = "{} {:d} ({:%d/%m}) ".format(translator.translate_to_language(["Block"])[0], key[0], key[1])
        ax = grp_axis.plot(ax=ax, kind='line', x="Datetime", y="Glucose_Auto", label=label)
    if block_info is not None:
        for i, dt in enumerate(block_info_sample["Datetime"]):
            plt.axvline(dt, color='grey', linestyle='--', label='Carbo.' if i == 0 else "")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    if one_day:
        #ax.legend(loc='best')
        ax.legend(loc='center left', bbox_to_anchor = (1, 0.5))
    else:
        ax.legend()

    #Set figure size
    fig.set_facecolor("white")
    fig.set_size_inches(12, 5, forward=True)

    #Export figure
    if to_file:
        if output_path is not None:
            path = join(output_path, "{}.png".format(init_day.strftime("%d-%m-%y")))
        else:
            path = "{}.png".format(init_day.strftime("%d-%m-%y"))
        plt.savefig(path)
        return path
    else:
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


def generate_png_tree(tree, feature_names, target_names):
    """Create PNG image of a decision tree using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("tree.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names,
                        class_names=target_names,
                        filled=True, rounded=True,
                        special_characters=True)

    command = ["dot", "-Tpng", "tree.dot", "-o", "dt.png"]
    try:
        subprocess.check_call(command)
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")


def generate_graph_tree(tree, feature_names, target_names):
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data,
                    feature_names=feature_names,
                    class_names=target_names,
                    filled=True, rounded=True,
                    special_characters=True)
    return pydot.graph_from_dot_data(dot_data.getvalue())
