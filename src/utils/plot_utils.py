import os

import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import auc, plot_precision_recall_curve, plot_roc_curve, plot_confusion_matrix


def _set_plot_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in the document.
    Copied from https://jwalton.info/Embed-Publication-Matplotlib-Latex/.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def plot_hist(df, name="Original", bin_size=100, plots_dir="plots/"):
    """Plot a histogram and save it to disk."""
    height = int((len(df.columns) // 3) + (len(df.columns) % 3 != 0))
    fig, axes = plt.subplots(height, 3, figsize=(12, 3 + height * 2))
    df = df.astype('float')
    i = 0
    for triaxis in axes:
        for axis in triaxis:
            if i == len(df.columns):
                break
            df.hist(column=df.columns[i], ax=axis, bins=bin_size)
            i = i + 1
    plt.tight_layout()
    # Save image to disk
    os.makedirs(plots_dir, exist_ok=True)
    path = f"{plots_dir}{bin_size}_{name}_Histograms.png"
    plt.savefig(path)


def plot_embedding(embedding, labels, title="", color_scheme="dark"):
    """Plot the UMAP embedding."""
    plt.style.use('dark_background')
    classes = np.unique(labels)
    fig, ax = plt.subplots(1, figsize=(14, 10))

    if color_scheme == "light":
        color_list = ['#25404B', '#387387', '#64D488', '#16A68B', '#FFBF67']
        cmap = colors.ListedColormap(color_list[:len(classes)])
        ax.spines['left'].set_color('#25404B')
        ax.spines['bottom'].set_color('#25404B')
        ax.tick_params(axis='x', colors='#25404B')
        ax.tick_params(axis='y', colors='#25404B')
        ax.yaxis.label.set_color('#25404B')
        ax.xaxis.label.set_color('#25404B')
        ax.title.set_color('#25404B')
    elif color_scheme == "dark":
        cmap = plt.get_cmap('rainbow', len(classes))

    plt.scatter(*embedding.T, s=6, c=np.array(labels), cmap=cmap)
    plt.setp(ax, xticks=[], yticks=[])
    cbar = plt.colorbar()
    cbar.set_ticks(classes)
    cbar.set_ticklabels(classes)
    plt.title(title)
    return fig


def plot_pr_curve(estimator, x, y, ax):
    """Create a precision-recall plot on the provided ax."""
    disp_pr = plot_precision_recall_curve(estimator, x, y, ax=ax)
    disp_pr.ax_.set_title("Precision-Recall curve")


def plot_conf_matrix(estimator, X, y, ax):
    """Create a confusion matrix plot on the provided ax."""
    # Plot the matrix
    disp_cm = plot_confusion_matrix(estimator, X, y, cmap=plt.cm.Blues, ax=ax)
    disp_cm.ax_.set_title("Normalized confusion matrix")


def plot_roc(estimator, X, y, ax):
    """Create a receiver operating characteristic plot on the provided ax."""
    # Plot the ROC curve
    plot_roc_curve(estimator, X, y, ax=ax, color='darkorange', lw=2)
    # Shrink the boundaries
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Set a title
    plt.title('Receiver operating characteristic (ROC)')
    # Add the diagonal plot line
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    # Get the legend out of the way - (removed this line because it prints the warning "No label handles to be found..")
    # plt.legend(loc="lower right")


def plot_roc_kfold(estimators, X, y, ax):
    """Create a receiver operating characteristic plot for every k-fold split set on the provided ax."""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    # Individual curves
    for idx, estimator in enumerate(estimators):
        viz = plot_roc_curve(estimator, X, y, name=f'ROC fold {idx}', alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    # The diagonal line
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    # Mean curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b', lw=2, alpha=.8, label=r'Mean ROC (AUC = %0.2f %0.2f)' %
                                                                 (mean_auc, std_auc))
    # Grey confidence intervals
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)  # label=r'$\pmodel_type$ 1 std. dev.')
    # Title and legend
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], title="Receiver operating characteristic (ROC)")
    ax.legend(loc="lower right")


def violin_plot(df, x_name, y_name=None, xticks=None, title='Violin Plot', out_path='../plots',
                file_name='violin_plot'):
    figdim = _set_plot_size(426)
    text_frame_color = '#25404B'
    plot_colors = ["#0073EE", "#FFBF15", "#64D488", "#C52230", "#A222C7"]

    plt.figure(figsize=figdim)

    sns.set_palette(plot_colors)
    sns.violinplot(x=x_name, y=y_name, data=df)

    # Adjust frame, ticks and text colors
    ax = plt.gca()
    plt.setp(ax.spines.values(), color=text_frame_color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=text_frame_color)
    if y_name != None:
        ax.set_ylabel(y_name, fontsize=12, color=text_frame_color)
    ax.set_xlabel(x_name, fontsize=12, color=text_frame_color)
    if len(xticks) != 0:
        plt.xticks(ticks=xticks, fontsize=10, color=text_frame_color)
    plt.yticks(fontsize=10, color=text_frame_color)
    plt.title(title, size=14, color=text_frame_color)

    # Store
    dest = os.path.join(out_path, file_name)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

    print("Stored violin plots at:", dest)


def boxplot(df, x_name, y_name=None, xticks=None, title='Box Plot', out_path='../plots', file_name='box_plot'):
    figdim = _set_plot_size(426)
    text_frame_color = '#25404B'
    plot_colors = ["#0073EE", "#FFBF15", "#64D488", "#C52230", "#A222C7"]

    plt.figure(figsize=figdim)

    sns.set_palette(plot_colors)
    sns.boxplot(x=x_name, y=y_name, data=df)

    # Adjust frame, ticks and text colors
    ax = plt.gca()
    plt.setp(ax.spines.values(), color=text_frame_color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=text_frame_color)
    if y_name != None:
        ax.set_ylabel(y_name, fontsize=12, color=text_frame_color)
    if len(xticks) != 0:
        plt.xticks(ticks=xticks, fontsize=10, color=text_frame_color)
    plt.yticks(fontsize=10, color=text_frame_color)
    plt.title(title, size=14, color=text_frame_color)

    # Store
    dest = os.path.join(out_path, file_name)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

    print("Stored box plots at:", dest)


def violin_plot(df, x_name, y_name=None, xticks=None, title='Violin Plot', out_path='../plots',
                file_name='violin_plot'):
    figdim = _set_plot_size(426)
    text_frame_color = '#25404B'
    plot_colors = ["#0073EE", "#FFBF15", "#64D488", "#C52230", "#A222C7"]

    plt.figure(figsize=figdim)

    sns.set_palette(plot_colors)
    sns.violinplot(x=x_name, y=y_name, data=df)

    # Adjust frame, ticks and text colors
    ax = plt.gca()
    plt.setp(ax.spines.values(), color=text_frame_color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=text_frame_color)
    if y_name != None:
        ax.set_ylabel(y_name, fontsize=12, color=text_frame_color)
    ax.set_xlabel(x_name, fontsize=12, color=text_frame_color)
    if len(xticks) != 0:
        plt.xticks(ticks=xticks, fontsize=10, color=text_frame_color)
    plt.yticks(fontsize=10, color=text_frame_color)
    plt.title(title, size=14, color=text_frame_color)

    # Store
    dest = os.path.join(out_path, file_name)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

    print("Stored violin plots at:", dest)


def boxplot(df, x_name, y_name=None, xticks=None, title='Box Plot', out_path='../plots', file_name='box_plot'):
    figdim = _set_plot_size(426)
    text_frame_color = '#25404B'
    plot_colors = ["#0073EE", "#FFBF15", "#64D488", "#C52230", "#A222C7"]

    plt.figure(figsize=figdim)

    sns.set_palette(plot_colors)
    sns.boxplot(x=x_name, y=y_name, data=df)

    # Adjust frame, ticks and text colors
    ax = plt.gca()
    plt.setp(ax.spines.values(), color=text_frame_color)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=text_frame_color)
    if y_name != None:
        ax.set_ylabel(y_name, fontsize=12, color=text_frame_color)
    if len(xticks) != 0:
        plt.xticks(ticks=xticks, fontsize=10, color=text_frame_color)
    plt.yticks(fontsize=10, color=text_frame_color)
    plt.title(title, size=14, color=text_frame_color)

    # Store
    dest = os.path.join(out_path, file_name)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()

    print("Stored box plots at:", dest)


def plot_rfe_scores(scores, fig_path):
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(scores) + 1), scores)
    plt.savefig(fig_path)
