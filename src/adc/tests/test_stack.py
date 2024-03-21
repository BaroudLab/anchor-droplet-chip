import logging
import os
import shutil
import time
from glob import glob

import dask.array as da
import numpy as np
from pytest import fixture
from tifffile import imread

from adc._projection_stack import ProjectAlong
from adc._split_stack import SplitAlong
from adc._sub_stack import SubStack

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TEST_DIR_NAME = "test_split"
T, P, C, Z, Y, X = (8, 10, 3, 15, 256, 256)
MAX_WAIT_SEC = 20


@fixture
def test_stack():
    sizes = {"T": T, "P": P, "C": C, "Z": Z, "Y": Y, "X": Y}
    tpcz_stack = np.zeros(tuple(sizes.values()), dtype="uint16")
    return {
        "data": tpcz_stack,
        "metadata": {
            "dask_data": da.from_array(tpcz_stack),
            "sizes": sizes,
            "pixel_size_um": 0.3,
            "path": "./test.tif"
        },
        "channel_axis": 2,
    }


def test_substack(make_napari_viewer, test_stack):
    v = make_napari_viewer()
    v.add_image(**test_stack)
    assert len(v.layers) == 3

    ss = SubStack(v)
    # v.window.add_dock_widget(ss)
    assert v.dims.axis_labels == tuple("TPZYX")  # C used for as channel axis

    ss.slice_container[3].start.value = 5  # z
    ss.slice_container[3].stop.value = 10  # z
    ss.make_new_layer()
    assert len(v.layers) == 6
    assert v.layers[3].data.shape == (T, P, 5, Y, X)
    assert v.layers[3].metadata["dask_data"].shape == (T, P, C, 5, Y, X)

    ps = ProjectAlong(v)
    # v.window.add_dock_widget(ps)
    ps.axis_selector.value = "Z:15"
    ps.make_projection()
    assert len(v.layers) == 9
    assert v.layers[6].data.shape == (T, P, Y, X)
    assert v.layers[6].metadata["dask_data"].shape == (T, P, C, Y, X)

    for i in v.layers[:6]:
        v.layers.remove(i)
    assert len(v.layers) == 3

    st = SplitAlong(v)
    # v.window.add_dock_widget(st)

    st.axis_selector.value = f"P:{P}"
    st.split_data()
    assert len(st.data_list) == P
    assert st.data_list[0].shape == (T, C, Y, X)
    testdir = "./pos/pos*/input/"
    try:
        st.start_export()
        start = time.time()
        while (
            len(glob(os.path.join(testdir, "*.tif"))) < P
            and time.time() - start < MAX_WAIT_SEC
        ):
            time.sleep(1)
            logger.debug("waiting for tifs")
        assert len(flist := glob(os.path.join(testdir, "*.tif"))) == P
        assert imread(flist[0]).shape == (T, C, Y, X)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree("./pos")


def test_projection(make_napari_viewer, test_stack):
    v = make_napari_viewer()
    v.add_image(**test_stack)

    m = v.layers[0].metadata.copy()
    p = ProjectAlong(v)
    assert v.dims.axis_labels == tuple("TPZYX")  # C used for as channel axis

    p.axis_selector.value = f"Z:{Z}"
    p.make_projection()
    assert v.layers[0].metadata == m
