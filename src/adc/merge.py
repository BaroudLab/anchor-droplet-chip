import logging
import os
from importlib.metadata import PackageNotFoundError, version

import fire
import pandas as pd

logger = logging.getLogger("adc.merge")
try:
    __version__ = version("anchor-droplet-chip")
except PackageNotFoundError:
    # package is not installed
    __version__ = "Unknown"


def merge(
    counts_day1: pd.DataFrame, counts_day2: pd.DataFrame
) -> pd.DataFrame:
    """
    Copies the first input into the output and adds the
    `n_cells` from the second input into the output as `n_cells_final`
    """
    table = counts_day1.copy()
    logger.info(f"Copy day1 to the output")
    table.loc[:, "n_cells_final"] = counts_day2.n_cells
    logger.info("Added the the column `n_cells_final` from day2 to the output")
    return table


def merge_csv(
    csv_day1: str, csv_day2: str, csv_out: str = "", concentration=None
):
    """
    Reads the data and performs merging (see ads.merge.merge)
    """
    if not csv_out:
        csv_out = os.path.join(
            os.path.commonpath([csv_day1, csv_day2]), "table.csv"
        )
    logging.basicConfig(level="INFO")
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
        if concentration is not None:
            table.loc["concentration"] = concentration
        logger.info(f"Saving the output: {csv_out}")
        table.to_csv(csv_out)
    except Exception as e:
        logger.error(f"Merge failed due to {e}")
        raise e


if __name__ == "__main__":
    fire.Fire(merge_csv)
