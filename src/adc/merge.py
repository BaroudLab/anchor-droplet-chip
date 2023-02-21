import logging
import os
from importlib.metadata import PackageNotFoundError, version

import fire
import pandas as pd

from adc import plot

logging.basicConfig(level="INFO")

logger = logging.getLogger("adc.merge")
try:
    __version__ = version("anchor-droplet-chip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"


def merge(
    counts_day1: pd.DataFrame,
    counts_day2: pd.DataFrame,
    concentration: float = None,
    msg="",
) -> pd.DataFrame:
    """
    Copies the first input into the output and adds the
    `n_cells` from the second input into the output as `n_cells_final`
    """
    logger.info(msg)
    table = counts_day1.copy()
    logger.info(f"Copy day1 to the output")
    table.loc[:, "n_cells_final"] = counts_day2.n_cells
    logger.info("Added the the column `n_cells_final` from day2 to the output")
    if concentration is not None:
        table.loc[:, "concentration"] = concentration
        logger.info(f"Add concentration: {concentration}")
    else:
        logger.info("No concentration")
    return table


def merge_csv(
    csv_day1: str,
    csv_day2: str,
    csv_out: str = "",
):
    """
    Reads the data and performs merging (see ads.merge.merge)
    """
    if not csv_out:
        csv_out = os.path.join(
            os.path.commonpath([csv_day1, csv_day2]), "table.csv"
        )
    fh = logging.FileHandler(log_path := csv_out.replace(".csv", ".log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"anchor-droplet-chip {__version__}")
    logger.info(f"Log will be saved in {os.path.abspath(log_path)}")
    try:
        logger.info(f"Reading {csv_day1}")
        counts_day1 = pd.read_csv(csv_day1)
        logger.info(f"Reading {csv_day2}")
        counts_day2 = pd.read_csv(csv_day2)
        logger.info(f"Merging both")
        table = merge(counts_day1=counts_day1, counts_day2=counts_day2)
        logger.info(f"Saving the output: {csv_out}")
        table.to_csv(csv_out)

    except Exception as e:
        logger.error(f"Merge failed due to {e}")
        raise e


def merge_all(
    paths_day1: "list[str]",
    paths_day2: "list[str]",
    concentrations: "list[int]",
    threshold: int = 10,
    table_path: str = "",
    swarm_path: str = "",
    prob_path: str = "",
    prob_log_path: str = "",
) -> None:
    fh = logging.FileHandler(
        log_path := str(table_path).replace(".csv", ".log")
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s : %(message)s"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.debug(
        "args1: ",
        paths_day1,
    )
    logger.debug(
        "args2: ",
        paths_day2,
    )
    logger.debug("c:", concentrations)
    logger.debug("out:", table_path)
    m = [
        merge(
            counts_day1=pd.read_csv(d1),
            counts_day2=pd.read_csv(d2),
            concentration=c,
            msg=f"Merging {d1} and {d2}",
        )
        for d1, d2, c in zip(paths_day1, paths_day2, concentrations)
    ]
    df = pd.concat(m, ignore_index=True)

    df.loc[:, "final_state"] = df.n_cells_final > threshold
    df.loc[:, "threshold"] = threshold

    logger.debug(df.head())
    if table_path:
        df.to_csv(str(table_path))
        logger.info(f"Saved table: {table_path}")

    n_max = int(df.n_cells.mean() * 3)

    plot._swarm_counts(
        df, n_max=n_max, threshold=threshold, path=str(swarm_path)
    )
    plot._probs(df, n_max=n_max, path=str(prob_path))
    plot._probs_log(df, n_max=n_max, path=str(prob_log_path))


if __name__ == "__main__":
    fire.Fire(merge_all)
