from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re


def add_gfp_final_state(merged_table: pd.DataFrame):
    merged_table.loc[:, "ratio"] = merged_table.top10px / merged_table.mean_intensity

    # merged_table = merged.merge_with_gfp_hour(map(pd.read_csv, tables))
    merged_gfp = merged_table.query("channel == 'GFP'")
    
    gfp_positive_labels = merged_gfp.groupby(["path","label"]).max()[["GFP_positive"]]
    gfp_positive_labels#.to_csv("pos0.csv")
    
    merged_table_with_final_gfp = merged_table.set_index(["path","label"]).join(gfp_positive_labels,  rsuffix="_final").reset_index()
    return merged_table_with_final_gfp


#plot below will show only the mCherry channel data
#different colored lines indicate different labels
#confidence interval shows spread of different paths for each label 

def plot_top10px_for_channel(df, title="", channel_name="mCherry"):
    '''
    
        Example usage:
        _ = plot_top10px_for_channel(merged_table_with_final_gfp, channel_name="mCherry")

    '''
    
    # Filter the DataFrame for the specific channel
    df_channel = df[df['channel'] == channel_name]
    
    # Create a 2x1 subplot layout
    fig, ax = plt.subplots(ncols=2, figsize=(10, 6))
    
    df_channel.label = df_channel.label.astype("category")
    # Create a line plot for top10px over GFPhour with specific settings
    sns.lineplot(
        ax=ax[0],
        data=df_channel,
        x='GFPhour',  # Use GFPhour for x-axis
        y='top10px',
        # hue='label',
        palette='Set2',
        style_order=[True, False],
        style="GFP_positive_final",
        # style="path",
        legend=False  # Remove the legend
    )
    ax[0].set_title(title)
    
    # Create a line plot for ratio over GFPhour with specific settings
    sns.lineplot(
        ax=ax[1],
        data=df_channel,
        x='GFPhour',  # Use GFPhour for x-axis
        y='ratio',
        # hue='label',
        style_order=[True, False],
        style="GFP_positive_final",
        legend=False,  # Remove the legend
        palette='Set2'
    )
    ax[1].legend(loc=(1.1, 0))  # Adjust the legend position
    ax[1].set_title(title)
    
    plt.tight_layout()  # Adjust layout for better spacing
    # return fig


def plot_top10px_for_channel_gfp_final(df, title="", channel_name="mCherry", ylim1=(100, 400), ylim2=(1, 2), xlim=(-10, 6)):
    # Calculate the ratio column by dividing top10px by mean_intensity
    df.loc[:, "ratio"] = df.top10px.values / df.mean_intensity.values
    
    # Filter the DataFrame for the specific channel
    df_channel = df
    
    # Create a 2x1 subplot layout
    fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(10, 6))
    
    df_channel.label = df_channel.label.astype("category")
    # Create a line plot for top10px over GFPhour with specific settings
    sns.lineplot(
        size="channel",
        ax=ax[0],
        data=df_channel,
        x='GFPhour',  # Use GFPhour for x-axis
        y='top10px',
        hue='label',
        palette='Set2',
        style="GFP_positive_final",
        style_order=[True, False],
        legend=False  # Remove the legend
    )
    ax[0].set_title(title)
    ax[0].set_ylim(*ylim1)
    
    # Create a line plot for ratio over GFPhour with specific settings
    sns.lineplot(
        size="channel",
        ax=ax[1],
        data=df_channel,
        x='GFPhour',  # Use GFPhour for x-axis
        y='ratio',
        hue='label',        
        style_order=[True, False],
        style="GFP_positive_final",
        # legend=False,  # Remove the legend
        palette='Set2'
    )
    ax[1].legend(loc=(1.1, 0))  # Adjust the legend position
    ax[1].set_title(title)
    ax[1].set_ylim(*ylim2)
    ax[1].set_xlim(*xlim)
    
    plt.tight_layout()  # Adjust layout for better spacing
    # return fig

def plot_intensitites_with_gfp_final(merged_table_with_final_gfp, ylim1=(100, 400), ylim2=(1, 2), xlim=(-10, 6)):
    for p in merged_table_with_final_gfp.path.unique():
        print(f"path=='{p}'")
        try:
            plot_top10px_for_channel_gfp_final(
                merged_table_with_final_gfp.query(f"path=='{p}'"), 
                channel_name="mCherry", 
                title = re.search(r"pos\d+", p).group(),
                ylim1=ylim1,
                ylim2=ylim2,
                xlim=xlim)
        except Exception as e:
            print(e, e.args)

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
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        hue=data.path,
    )
    sns.lineplot(
        ax=ax1,
        x=data.hours,
        y=data.top10px / data.mean_intensity,
        # hue=data.path,
        linewidth=5,
        color="k",
    )
    # ax2=ax1.twinx()
    # get_gfp_positive_number(df).plot(ax=ax2, label="GFP positive cells", linewidth=3)
    # get_cell_number(df).plot(ax=ax2, label="total number of cells", linewidth=2)
    # sns.lineplot(ax=ax2,
    #              x="frame",
    #              y="GFP_positive",
    #              label="GFP positive cells",
    #              data=df
    #             )
    ax1.set_ylim(1, 2.5)
    ax1.set_ylabel("norm intensity")
    # ax2.set_ylabel("cell count")
    # ax2.legend(loc="upper right")
    ax1.legend(loc="upper right")
    ax1.set_title("top 10 px")
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
    )
    sns.lineplot(
        ax=ax1,
        x=data.GFPhour,
        y=data.top10px / data.mean_intensity,
        # hue=data.path,
        linewidth=5,
        color="k",
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
    ax1.legend(loc="upper right")
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
