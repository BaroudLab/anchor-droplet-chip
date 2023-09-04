from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .single import get_title


def get_gfp_positive_number(df):
    return (
        df.query("channel == 'GFP' and mask == 'cellpose'")
        .groupby(["hours", "path"])
        .sum()
        .GFP_positive
    )


def count_nuc_number(df):
    return (
        df.query("mask == 'nuclei' and channel == 'mCherry'")
        .groupby(["hours", "path"])
        .count()["channel"]
    )


def count_nuc_number(df):
    return (
        df.query("mask == 'nuclei' and channel=='mCherry'")
        .groupby(["hours", "path"])
        .count()["frame"]
    )


def get_cell_number(df):
    return (
        df.query("mask == 'cellpose' and channel=='mCherry'")
        .groupby(["hours", "path"])
        .count()["frame"]
    )


def add_gfp_hour(
    single_well_table: pd.DataFrame,
    new_col: str = "GFPhour",
    query="channel == 'GFP' and mask == 'cellpose' and GFP_positive",
    hours_col="hours",
):
    """
    Find the first frame where GFP is positive, shift the time accrodingly and put it to the hew column
    """
    t = single_well_table.copy()
    GFPhour = (
        t.query(query).groupby(hours_col).sum().reset_index()[hours_col].min()
    )
    t.loc[:, new_col] = t.hours - GFPhour
    return t


def merge_with_gfp_hour(tables: Iterable[pd.DataFrame], op=add_gfp_hour):
    """
    Add GFP hour to the tables and return concatenated one
    """
    return pd.concat(map(op, tables), ignore_index=True)


def plot_10(df, save_path="all-top10px.png"):
    plt.rc("font", family="Arial")
    data = df.query("mask=='cellpose' and channel=='mCherry' ")
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    data.loc[:, "title"] = data["path"].map(get_title)
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        hue=data.title,
        legend=False,
    )
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        # hue=data.path,
        linewidth=5,
        color="k",
        legend=False,
    )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    # ax2.set_ylabel("cell count")
    # ax2.legend(loc="upper right")
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


def plot_max(df, save_path="all-max_intensity.png"):
    plt.rc("font", family="Arial")
    data = df.query(
        "mask=='cellpose' and channel=='mCherry' and manual_filter"
    )
    fig, ax1 = plt.subplots(figsize=(5, 5), dpi=150, facecolor="w")
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.max_intensity / data.mean_intensity,
        hue="path",
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
    ax1.set_title("max/mean intensity ratio")

    fig.savefig(save_path)
    # plt.close()


def plot_ilastik_intensity(
    pv_nuc, save_path="all-nuc-cyto-intensity-ratio.png"
):
    fig, ax = plt.subplots(dpi=150, facecolor="w")
    ax = sns.lineplot(
        data=pv_nuc,
        x="hours",
        y="ratio",
        hue="path",
    )
    ax = sns.lineplot(
        data=pv_nuc, x="hours", y="ratio", linewidth=5, color="k"
    )
    ax.set_title("Ratio mean_intensity Ilastik nuc / cellpose cyto")
    ax.legend(loc="upper right")
    plt.savefig(save_path)
    # plt.close()


def plot_num_nuc(filt_df, save_path="all_nuc_ilastik.png"):
    fig, ax1 = plt.subplots(dpi=150, facecolor="w")
    (count_nuc_number(filt_df) / get_cell_number(filt_df)).plot(
        ax=ax1,
        label="Ilastik nuclei detections vs cell number",
        color="mediumvioletred",
    )
    ax2 = ax1.twinx()
    get_cell_number(filt_df).plot(
        ax=ax2, color="steelblue", label="total number of cells"
    )
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_ylabel("ratio")
    ax2.set_ylabel("count")
    ax1.set_title("Nuc Numbers")
    fig.savefig(save_path)
    # plt.close()
