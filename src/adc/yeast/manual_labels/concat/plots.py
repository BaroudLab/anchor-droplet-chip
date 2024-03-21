import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot_top10px_for_channel_gfp_final(
    df,
    title="",
    ylim1=(100, 400),
    ylim2=(1, 2),
    xlim=(-10, 6),
):
    # Filter the DataFrame for the specific channel
    df = df

    # Create a 2x1 subplot layout
    fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(10, 6))

    df.label = df.label.astype("category")
    # Create a line plot for top10px over GFPhour with specific settings
    sns.lineplot(
        size="channel",
        ax=ax[0],
        data=df,
        x="GFPhour",  # Use GFPhour for x-axis
        y="top10px",
        hue="label",
        palette="Set2",
        style="GFP_positive_final",
        style_order=[True, False],
        legend=False,  # Remove the legend
    )
    ax[0].set_title(title)
    ax[0].set_ylim(*ylim1)

    # Create a line plot for ratio over GFPhour with specific settings
    sns.lineplot(
        size="channel",
        ax=ax[1],
        data=df,
        x="GFPhour",  # Use GFPhour for x-axis
        y="ratio",
        hue="label",
        style_order=[True, False],
        style="GFP_positive_final",
        # legend=False,  # Remove the legend
        palette="Set2",
    )
    ax[1].legend(loc=(1.1, 0))  # Adjust the legend position
    ax[1].set_title(title)
    ax[1].set_ylim(*ylim2)
    ax[1].set_xlim(*xlim)

    plt.tight_layout()  # Adjust layout for better spacing
    # return fig


def plot_top10px_split_labels(
    df, title="", ylim1=(100, 500), ylim2=(1, 2), xlim=(-20, 6), min_len=5
):
    """Provide a table from a single well. Intensities will be plotted 1 panel per label"""
    long_tracks = df.groupby(["label", "path"]).count().query(f"area  >= {min_len}").reset_index()
    labels = sorted(long_tracks.label.unique())
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
            color="mediumvioletred",
            legend=False,  # Remove the legend
        )
        
        sns.lineplot(
            size="channel",
            ax=ax,
            data=df[df.label == (i + 1)],
            x="GFPhour",  # Use GFPhour for x-axis
            y="mean_intensity",
            color="green",
            legend=False,  # Remove the legend
        )
        # sns.lineplot(
        #     size="channel",
        #     ax=ax,
        #     data=df[df.label == (i + 1)],
        #     x="GFPhour",  # Use GFPhour for x-axis
        #     y="cyto_wo100px",
        #     legend=False,  # Remove the legend
        # )
        ax.set_title(f"{title}: cell# {i}")
        ax.set_ylim(*ylim1)

    # Create a line plot for ratio over GFPhour with specific settings
    for i, ax in enumerate(rows[1]):
        data = df[df.label == (i + 1)]
        # data.loc[:,"ratio_wo100px"] = data.top10px / data.cyto_wo100px
        sns.lineplot(
            ax=ax,
            data=data,
            x="GFPhour",  # Use GFPhour for x-axis
            y="ratio",
            lw=3,
            hue="channel",
            palette=["mediumvioletred", "green"],
            # hue="GFP_positive_final" if "GFP_positive_final" in data.columns else None,
            # hue_order=[True, False],
            legend=False,  # Remove the legend
        )
        # sns.lineplot(
        #     ax=ax,
        #     data=df[df.channel == "GFP"],
        #     x="GFPhour",  # Use GFPhour for x-axis
        #     y="mean_intensity",
        #     # hue="GFP_positive_final" if "GFP_positive_final" in data.columns else None,
        #     # hue_order=[True, False],
        #     legend=False,  # Remove the legend
        # )
        
        ax.set_title(f"{title}: cell {i}")
        ax.set_ylim(*ylim2)
        ax.set_xlim(*xlim)

    plt.tight_layout()  # Adjust layout for better spacing
    return fig



def plot_intensitites_with_gfp_final(
    merged_table_with_final_gfp,
    op=plot_top10px_split_labels,
    ylim1=(100, 400),
    ylim2=(1, 2),
    xlim=(-10, 6),
):
    for p in sorted(merged_table_with_final_gfp.path.unique()):
        # print(f"path:'{p}'")
        title = re.search(r"pos\d+", p).group(),
        try:
            op(
                merged_table_with_final_gfp.query(f"path=='{p}'"),
                title=title,
                ylim1=ylim1,
                ylim2=ylim2,
                xlim=xlim,
            )
        except Exception as e:
            print(f"{title} failed: {e} {e.args}")


#function to plot tracked labels over time for GFP data with individual paths
def plot_top10px_ratio_all(df, title=""):
    #calculate the ratio columnby dividing top10px over mean_intensity
    df.loc[:, "ratio"] = df.top10px / df.mean_intensity
    
    fig, ax = plt.subplots(ncols=2, figsize = (10,6))
    df.label = df.label.astype("category")
    #create a line plot for top10px over GFP hour with specific settings (with paths as individual lines)
    sns.lineplot(
        ax=ax[0],
        data=df,
        x='GFPhour',
        y='top10px',
        hue='label',
        palette='Set2',
        style="channel",
        legend=False,
        estimator=None,
        units="path"
    )
    ax[0].set_title(title)
    #create a line plot for ratio over GFPhour with specific settings (with paths as individual lines)
    sns.lineplot(ax=ax[1],
        data=df,
        x='GFPhour',
        y='ratio',
        hue='label',
        palette='Set2',
        style="channel",
        estimator=None,
        units="path"
    )
    ax[1].legend(loc=(1.1,0))
    ax[1].set_title(title)
    plt.tight_layout()
    return fig
