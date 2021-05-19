import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_saliency_plot(mean_importance, sd_importance, feature_names):
    """ Creates and stores plots for torch saliency specifically.

    Args:
        mean_importance:    Mean saliency/importance values as computed by the torch saliency function.
        sd_importance:      Respective standard deviation.
        feature_names:      List of the feature names in the dataset.
        clf_name:           Name of the classifier.
    
    """
    value_col = "Importance"
    feature_col = "Name"
    sd_col = "SD" if sd_importance is not None else None
    timestamp = time.strftime("%Y-%model_type-%d-%H-%M-%S")
    # Visualize saliencies
    if sd_importance is not None:
        df_all = pd.DataFrame({value_col: mean_importance, feature_col: feature_names, sd_col: sd_importance}
                              ).sort_values(value_col, ascending=True)
    else:
        df_all = pd.DataFrame({value_col: mean_importance, feature_col: feature_names}
                              ).sort_values(value_col, ascending=True)

    # Create chunks of features sorted by intervals
    stride = 0.25  # determines value range that belongs to one interval
    dfs = _chunk_by_importance(df_all, value_col, stride=stride)
    plot_subfolder = os.path.join("..", "plots", value_col, timestamp)
    os.makedirs(plot_subfolder, exist_ok=True)

    for i, df in enumerate(dfs):
        lower_bound = 0.0 + i * stride
        upper_bound = stride + i * stride
        title = f"SmoothGrad Saliencies for POD\n Range {lower_bound} to {upper_bound}"
        file_name = f"importances_range_{lower_bound}_to_{upper_bound}.pdf"
        horizontal_importance_plot(df, value_col, feature_col, title, sd_col=sd_col, xtick_step=stride,
                                   out_path=plot_subfolder, file_name=file_name)


def _chunk_by_importance(df, value_col, stride=0.2):
    """ If long lists of features are provided, this function can be used to chunk the importance values in order to create
    multiple plots.

    Args:
        df:                 DataFrame with importance values to be plotted.
        value_col (str):    Name of the column in which the importances are located.
        stride:             Interval of values that should belong to one chunk.
    """
    dfs = []
    num_chunks = int(1 / stride)
    bounds = [0.0, stride]
    for chunk in range(num_chunks):
        dfs += [df[(df[value_col] >= bounds[0]) & (df[value_col] < bounds[1])]]
        bounds = [bound + stride for bound in bounds]
    return dfs


def horizontal_importance_plot(df, value_col, feature_col, title, sd_col=None, xtick_step=0.1, out_path='../plots',
                               file_name='horizontal_plot'):
    """ Create plot for importances with horizontal bars.

    Args:
        df:                 DataFrame with importance values to be plotted.
        value_col (str):    Name of the column in which the importances are located.
        feature_col (str):  Name of the column in which the feature names are located.
        title (str):        Title of the plot.
        sd_col (str):       Plot SD whisker if name of the column with SD's are provided.
        xtick_step:         Stepsize between ticks on x-axis.
        out_path:           Where to store the plot.        
        file_name:          Name of the plot file.   
    """

    # Size the plot such that it fits a DinA4 page
    din4_len = 11 * (17 / 24)  # length in inches
    din4_width = 8.25  # width in inches
    text_frame_color = '#25404B'
    plot_color = "#0073EE"
    plot_len = len(df) / din4_len if len(df) > din4_len else len(df) * 1.5
    plt.figure(figsize=(din4_width, plot_len))

    if len(df) > 0:  # Make sure df is not empty since it would throw an error
        if sd_col is not None:
            sns.barplot(x=value_col, y=feature_col, data=df, label=df[feature_col], color=plot_color, ci=sd_col)
        else:
            sns.barplot(x=value_col, y=feature_col, data=df, label=df[feature_col], color=plot_color)

        # Adjust frame, ticks and text colors
        ax = plt.gca()
        plt.setp(ax.spines.values(), color=text_frame_color)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=text_frame_color)
        ax.set_ylabel('')
        ax.set_xlabel('Importance', fontsize=12, color=text_frame_color)
        plt.xticks(ticks=np.arange(0.0, 1.1, xtick_step), fontsize=10, color=text_frame_color)
        plt.yticks(fontsize=10, color=text_frame_color)
        plt.title(title, size=14, color=text_frame_color)

        # Store
        dest = os.path.join(out_path, file_name)
        plt.tight_layout()
        plt.savefig(dest)
        plt.close()

        print("Stored importance plots at:", dest)

    else:
        print("WARNING - Tried to plot empty DataFrame.")
