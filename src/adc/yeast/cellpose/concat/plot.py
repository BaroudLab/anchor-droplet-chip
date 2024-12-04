import seaborn as sns
from matplotlib import pyplot as plt

from ..measure.intensities import get_title


def plot_10(df, save_path="all-top10px.png", legend=False):
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry' ")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    data.loc[:, "title"] = data.path.map(get_title).values
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        hue=data.title,
        legend=legend,
    )
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        # hue=data.path,
        linewidth=5,
        color="k",
        legend=legend,
    )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    # ax2.set_ylabel("cell count")
    # ax2.legend(loc="upper right")
    if legend:
        ax1.legend(loc="upper right")
    ax1.set_title("top 10 px")
    if save_path:
        fig.savefig(save_path)
    # plt.close()


def plot_10_gfp(df, save_path="all-top10px-gfphour.png"):
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.GFPhour,
        y=data.top10px / data.mean_intensity,
        hue=data.path,
        legend=False,
    )
    sns.lineplot(
        ax=ax1,
        x=data.GFPhour,
        y=data.top10px / data.mean_intensity,
        # hue=data.path,
        linewidth=5,
        color="k",
        legend=False,
    )
    ax2 = ax1.twinx()
    # get_gfp_positive_number(df).plot(ax=ax2, label="GFP positive cells", linewidth=3)
    # # get_cell_number(df).plot(ax=ax2, label="total number of cells", linewidth=2)
    # sns.lineplot(ax=ax2,
    #              x="GFPhour",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    # ax2.set_ylabel("cell count")
    # ax2.legend(loc="upper right")
    # ax1.legend(loc="upper right")
    ax1.set_title("top 10 px")
    fig.savefig(save_path)
    # plt.close()
