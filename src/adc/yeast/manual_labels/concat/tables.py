
from typing import Iterable
from matplotlib import pyplot as plt
import pandas as pd 
import seaborn as sns

def merge_with_gfp_hour_and_final_state(tables: Iterable[pd.DataFrame]):
    """
    Add GFP hour to the tables and return concatenated one
    Add ratio: top10px / mean_intensity
    """
    ttt = pd.concat(map(_add_gfp_hour, tables), ignore_index=True)
    ttt.loc[:, "ratio"] = ttt.top10px / ttt.mean_intensity
    ttt.loc[:, "ratio10"] = ttt.top10px / ttt.mean_intensity
    ttt.loc[:, "ratio20"] = ttt.top20px / ttt.mean_intensity
    ttt.loc[:, "ratio100"] = ttt.top100px / ttt.mean_intensity
    ttt_fs = _add_gfp_final_state(ttt)
    return ttt_fs

def merge_with_gfp_hour(tables: Iterable[pd.DataFrame]):
    """
    Add GFP hour to the tables and return concatenated one
    Add ratio: top10px / mean_intensity
    """
    ttt = pd.concat(map(_add_gfp_hour, tables), ignore_index=True)
    ttt.loc[:, "ratio"] = ttt.top10px / ttt.mean_intensity
    return ttt


def merge_with_gfp_per_cell_hour(tables: Iterable[pd.DataFrame]):
    """
    Add GFP hour to the tables and return concatenated one
    Add ratio: top10px / mean_intensity
    """
    ttt = pd.concat(map(_add_gfp_hour, tables), ignore_index=True)
    ttt.loc[:, "ratio"] = ttt.top10px / ttt.mean_intensity
    return ttt


def _add_gfp_final_state(merged_table: pd.DataFrame):
    merged_table.loc[:, "ratio"] = (
        merged_table.top10px / merged_table.mean_intensity
    )

    # merged_table = merged.merge_with_gfp_hour(map(pd.read_csv, tables))
    merged_gfp = merged_table.query("channel == 'GFP'")

    gfp_positive_labels = merged_gfp.groupby(["path", "label"]).max()[
        ["GFP_positive"]
    ]
    gfp_positive_labels  # .to_csv("pos0.csv")

    merged_table_with_final_gfp = (
        merged_table.set_index(["path", "label"])
        .join(gfp_positive_labels, rsuffix="_final")
        .reset_index()
    )
    return merged_table_with_final_gfp


def _add_gfp_hour(
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


def plot_top10px_split_labels(
    df, title="", ylim1=(100, 500), ylim2=(1, 2), xlim=(-10, 6)
):
    """Provide a table from a single well. Intensities will be plotted 1 panel per label"""

    labels = sorted(df.label.unique())
    # Create a 2x1 subplot layout
    fig, rows = plt.subplots(
        ncols=len(labels), nrows=2, sharex=True, figsize=(2 * len(labels), 5)
    )

    # df.label = df.label.astype("category")
    # Create a line plot for top10px over GFPhour with specific settings
    for i, ax in enumerate(rows[0]):
        sns.lineplot(
            size="channel",
            ax=ax,
            data=df[df.label == (i + 1)],
            x="GFPhour",  # Use GFPhour for x-axis
            y="top10px",
            palette="Set2",
            legend=False,  # Remove the legend
        )
        ax.set_title(f"{title}: cell# {i}")
        ax.set_ylim(*ylim1)

    # Create a line plot for ratio over GFPhour with specific settings
    for i, ax in enumerate(rows[1]):
        sns.lineplot(
            size="channel",
            ax=ax,
            data=df[df.label == (i + 1)],
            x="GFPhour",  # Use GFPhour for x-axis
            y="ratio",
            palette="Set2",
            legend=False,  # Remove the legend
        )
        ax.set_title(f"{title}: cell# {i}")
        ax.set_ylim(*ylim2)
        ax.set_xlim(*xlim)

    plt.tight_layout()  # Adjust layout for better spacing
    return fig