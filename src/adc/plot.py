import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _swarm_counts(
    table: pd.DataFrame, n_max: int = 5, threshold=None, path=None
):
    fig, ax = plt.subplots(dpi=300)
    g = sns.swarmplot(
        ax=ax,
        data=table.query(f"n_cells < {n_max}"),
        x="concentration",
        y="n_cells_final",
        hue="n_cells",
        dodge=True,
        size=1,
    )
    if threshold is not None:
        g.axhline(
            y=threshold, color="k", lw=1, alpha=0.5, ls="--", label="threshold"
        )
    fig.savefig(path)


def _swarm_intensity(table: pd.DataFrame, n_max: int = 5, path=None):
    fig, ax = plt.subplots(dpi=300)
    sns.swarmplot(
        ax=ax,
        data=table.query(f"n_cells < {n_max}"),
        x="concentration",
        y="intensity_final",
        hue="n_cells",
        dodge=True,
        size=1,
    )
    fig.savefig(path)


def _probs(table: pd.DataFrame, n_max: int = 5, path=None):

    fig, ax = plt.subplots(dpi=300)
    sns.lineplot(
        ax=ax,
        data=table.query(f"n_cells < {n_max}"),
        x="concentration",
        y="final_state",
        hue="n_cells",
    )
    fig.savefig(path)


def _probs_log(table: pd.DataFrame, n_max: int = 5, path=None):
    fig, ax = plt.subplots(dpi=300)
    sns.lineplot(
        ax=ax,
        data=table.query(f"n_cells < {n_max}"),
        x="concentration",
        y="final_state",
        hue="n_cells",
    )
    ax.set_xscale("log")
    fig.savefig(path)
