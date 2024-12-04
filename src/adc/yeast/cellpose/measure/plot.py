
import os
import shutil

import seaborn as sns
from matplotlib import pyplot as plt


def plot_table(df, x="hours", title="", save_path="all_measurements.png"):
    fdf = df

    fig, ax = plt.subplots()

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"),
        x=x,
        y="mean_intensity",
        label="cellpose",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query("channel == 'mCherry' and mask == 'cellpose'"),
        x=x,
        y="top10px",
        label="top10",
    )

    sns.lineplot(
        ax=ax,
        data=fdf.query(
            "channel == 'GFP' and mask == 'cellpose' and mean_intensity > 0"
        ),
        x=x,
        y="mean_intensity",
        label="GFP",
    )
    ax.set_title(title)
    if save_path:
        fig.savefig(save_path)
        plt.close()
    return ax


def plot_max(df, title="", save_path="P=19-max_intensity.png"):
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.max_intensity / data.mean_intensity,
        label="mCherry-Rad53",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_gfp_positive_number(df).plot(
        ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3
    )
    get_cell_number(df).plot(
        ax=ax2, label="total number of cells", color="steelblue", linewidth=2
    )

    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title(f"max/mean intensity ratio {title}")

    fig.savefig(save_path)
    plt.close()


def plot_10(df, title="", save_path="P=19-top10px.png", backup_folder=""):
    if os.path.exists(save_path):
        if backup_folder:
            backup_path = os.path.join(
                os.path.dirname(save_path), backup_folder
            )
            os.makedirs(backup_path, exist_ok=True)
            shutil.move(
                save_path,
                os.path.join(backup_path, os.path.basename(save_path)),
            )
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry'")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        label="mCherry-Rad53",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_gfp_positive_number(df).plot(
        ax=ax2, label="GFP positive cells", color="seagreen", linewidth=3
    )
    get_cell_number(df).plot(
        ax=ax2, label="total number of cells", color="steelblue", linewidth=2
    )
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    ax2.set_ylabel("cell count")
    ax2.legend(loc="upper right")
    ax1.legend(loc="upper left")
    ax1.set_title(f"top 10 px {title}")
    if save_path:
        fig.savefig(save_path)
        plt.close()
    plt.show()


def get_gfp_positive_number(df):
    return (
        df.query("channel == 'GFP' and mask == 'cellpose'")
        .groupby("hours")
        .sum()
        .GFP_positive
    )


def get_cell_number(df):
    return (
        df.query("mask == 'cellpose' and channel=='mCherry'")
        .groupby("hours")
        .count()["frame"]
    )
