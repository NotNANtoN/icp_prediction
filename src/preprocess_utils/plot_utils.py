import matplotlib.pyplot as plt

import src.constants as constants

label_col = constants.label_col

def make_plots(df, prefix):
    ncols = 4
    nrows = len(df.columns) // 4
    nrows += (len(df.columns) % 4) > 0
    bins = 30

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True, figsize=(10, 25))
    axs = axs.reshape(-1)
    for idx, name in enumerate(df.columns):
        pod = df.loc[df[label_col] == 1, name]
        no_pod = df.loc[df[label_col] == 0, name]
        ax = axs[idx]
        ax.hist(pod, bins=bins, label=label_col, density=True)
        ax.hist(no_pod, bins=bins, label=f"No {label_col}", density=True, alpha=0.7)
        ax.legend()
        ax.set_title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"data/{prefix}_norm_hists_two_groups.pdf", dpi=300)

    plt.figure(figsize=(20, 20))
    df.hist(bins=bins, figsize=(20, 20))
    plt.tight_layout()
    plt.savefig(f"data/{prefix}_hists.pdf", dpi=300)

    plt.figure(figsize=(12, 12))
    df.isna().mean().sort_values().plot.barh(figsize=(12, 12))
    plt.savefig(f"data/{prefix}_missing.pdf", dpi=300)
