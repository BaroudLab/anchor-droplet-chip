from __future__ import annotations

import os
import pathlib

import pandas as pd
import tifffile as tf

URL = "https://github.com/BaroudLab/anchor-droplet-chip/releases/download/v0.0.5/"


def read_tif(path, **kwargs):
    return [(tf.imread(path), {**kwargs}, "image")]


def read_csv(path, **kwargs):
    centers = pd.read_csv(path)
    yx = centers[["y", "x"]].values * 8
    return [(yx, {**kwargs}, "points")]


DATA = [
    ("template_bin16_v3.tif", read_tif),
    ("centers_bin16.csv", read_csv),
]


def make_template():
    return (
        _load_sample_data(
            *DATA[0],
            name="template_bin16",
            colormap="cyan",
            opacity=0.5,
            scale=(8, 8),
        )
        + make_centers()
    )


def make_centers():

    return  _load_sample_data(*DATA[1], name="centers", size=100)


def download_url_to_file(
    url,
    file_path,
):
    import shutil

    import urllib3

    print(f"Downloading {url}")
    c = urllib3.PoolManager()
    with c.request("GET", url, preload_content=False) as resp, open(
        file_path, "wb"
    ) as out_file:
        shutil.copyfileobj(resp, out_file)
    resp.release_conn()
    print(f"Saved {file_path}")
    return file_path


def _load_sample_data(image_name, readfun=read_tif, **kwargs):

    cp_dir = pathlib.Path.home().joinpath(".anchor-droplet-chip")
    cp_dir.mkdir(exist_ok=True)
    data_dir = cp_dir.joinpath("data")
    data_dir.mkdir(exist_ok=True)

    url = URL + image_name

    cached_file = str(data_dir.joinpath(image_name))
    if not os.path.exists(cached_file):
        print(f"Downloading {image_name}")
        download_url_to_file(url, cached_file)
    return readfun(cached_file, **kwargs)
