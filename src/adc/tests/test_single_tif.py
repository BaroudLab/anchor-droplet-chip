import pathlib
import shutil
from importlib.metadata import version

import tifffile as tf
import urllib3
import yaml

from adc import align, count, fit

__version__ = version("anchor-droplet-chip")
print(__version__)


def load_data(url, filename, dir=None):
    if dir is not None:
        cwd = pathlib.Path.cwd()
        subfolder = cwd.joinpath(dir)
        subfolder.mkdir(exist_ok=True)
        filename = subfolder.joinpath(filename)
    else:
        filename = pathlib.Path(filename)

    if not filename.exists():
        print(f"loading {filename}")
        c = urllib3.PoolManager()

        with c.request(
            "GET", url, preload_content=False
        ) as resp, filename.open(mode="wb") as out_file:
            shutil.copyfileobj(resp, out_file)

        resp.release_conn()
    else:
        print(f"{filename} already exists")
    print(f"reading from disk {filename}")
    return tf.imread(filename)


def process_urls(key, config):
    if "url" in config[key]:
        return load_data(**config[key])
    else:
        return config[key]


def load_dataset(path):
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        print(config)

    return {key: process_urls(key, config) for key in config}


def test_pipeline():
    data_0h = load_dataset("test_data_0h.yaml")
    data_24h = load_dataset("test_data_24h.yaml")

    aligned_stack_0h, tvec_0h = align.align_stack(**data_0h)
    aligned_stack_24h, tvec_0h = align.align_stack(**data_24h)

    tf.imwrite(
        "day1/00ng_BF_TRITC_bin2-aligned.tif",
        aligned_stack_0h,
        imagej=True,
        metadata=align.META_ALIGNED,
    )
    tf.imwrite(
        "day2/00ng_BF_TRITC_bin2-24h-aligned.tif",
        aligned_stack_24h,
        imagej=True,
        metadata=align.META_ALIGNED,
    )

    counts_0h = count.stack(aligned_stack_0h)
    assert len(counts_0h) == 501
    assert len(counts_0h.columns) == 4
    poisson_lambda = fit.poisson(counts_0h.n_cells)
    assert 1 < poisson_lambda < 1.5
    counts_24h = count.stack(aligned_stack_24h)
    assert len(counts_24h) == 501
    assert len(counts_24h.columns) == 4
    table = counts_0h.copy()
    table.loc[:, "n_cells_final"] = counts_24h.n_cells
    # sns.swarmplot(data=table, x='n_cells', y='n_cells_final')
