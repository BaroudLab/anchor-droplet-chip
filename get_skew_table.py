import logging
import os
from functools import reduce
from operator import add
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tf
from fire import Fire
from scipy.stats import skewnorm
from skimage.measure import regionprops
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)


def get_cell_intensities(prop):
    I = prop.image_intensity.ravel()
    cell_intensities = I[I > 0]
    return cell_intensities


def get_hist_df(prop, step=10):
    try:
        cell_intensities = get_cell_intensities(prop)
        a, loc, scale = skewnorm.fit(cell_intensities)
        h = np.histogram(
            cell_intensities, bins=range(100, cell_intensities.max(), step)
        )

        return pd.DataFrame(
            {
                "bins": h[1][:-1],
                "intensities": h[0],
                "label": prop.label,
                "area": prop.area,
                "skew": a,
                "mod": loc,
                "mean": prop.mean_intensity,
                "max": prop.max_intensity,
                "scale": scale,
                "pdf": skewnorm.pdf(h[1][:-1], a, loc, scale)
                * prop.area
                * step,
            }
        )
    except IndexError:
        log.error("index error: %d", prop.label)


def get_skew_table(filt_table_path: str) -> (pd.DataFrame, str):
    tpath = Path(filt_table_path)
    out_path = str(tpath).replace("table", "skew_table")
    log.debug("out_path %s", out_path)
    assert not os.path.exists(
        out_path
    ), f"skipping: already exists! {out_path}"
    log.debug("reading table %s", tpath)
    good_labels = pd.read_csv(tpath)[["label", "frame"]]
    pos_path = Path(tpath).parent.parent
    log.debug("pos_path %s", pos_path)
    cellpose_path = pos_path / "input" / "cellpose.tif"
    log.debug("cellpose_path %s", cellpose_path)
    labels = tf.imread(cellpose_path)
    log.debug("labels %s", labels.shape)
    stack_path = pos_path / "input" / "stack.tif"
    mcherry = tf.imread(stack_path)[:, 1]
    log.debug("mcherry: %s", mcherry.shape)
    all_props = reduce(
        add,
        [
            regionprops(lb, intensity_image=img)
            for lb, img in tqdm(zip(labels, mcherry))
        ],
    )
    log.debug("all_props: %d props", len(all_props))
    good_props = list(
        filter(lambda p: p.label in good_labels.label.values, all_props)
    )
    log.debug("good_props: %d props", len(good_props))
    df = pd.concat(
        map(get_hist_df, tqdm(good_props)), ignore_index=True
    ).merge(good_labels, on="label")
    log.debug(df.head())
    df.to_csv(out_path, index=None)
    log.debug(f"saved to %s", out_path)
    log.debug("returning df, out_path")
    return df, out_path


def process(path):
    print(path)
    log.info(f"processing {path}")
    if not len(path):
        log.error("provide a path or a list of paths")
    try:
        df, out_path = get_skew_table(path)
    except AssertionError:
        pass


def main(*paths):
    if not len(paths):
        log.error("provide a path or a list of paths")
    list(map(process, tqdm(paths)))
    return True


if __name__ == "__main__":
    Fire(main)
