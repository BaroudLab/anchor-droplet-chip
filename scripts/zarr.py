import dask.array as da
import tifffile as tf
from zarr_tools import convert

kwargs = {
    "channel_axis": 1,
    "colormap": ["gray", "green", "blue"],
    "lut": [[1000, 30000], [440, 600], [0, 501]],
    "name": ["BF", "TRITC", "mask"],
}

data = da.stack([tf.imread(f) for f in snakemake.input.day1])  # noqa
convert.to_zarr(data, snakemake.output.zarr1, **kwargs)  # noqa

data = da.stack([tf.imread(f) for f in snakemake.input.day2])  # noqa
convert.to_zarr(data, snakemake.output.zarr2, **kwargs)  # noqa
