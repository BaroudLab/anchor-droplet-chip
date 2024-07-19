import napari

from adc._sub_stack import SubStack
from adc.tests.test_projection import TIF_PATH, tif_file
from adc.tests.test_reader import PLUGIN


def test_substack_single_channel(make_napari_viewer, tif_file):
    v = make_napari_viewer()
    layers = v.open(TIF_PATH, plugin=PLUGIN)
    assert len(layers) == 4  # ZCYX
    s = SubStack(v)

    s.slice_container[1].start.value = 2  # C[2:3]
    s.slice_container[1].stop.value = 3
    s.slice_container[2].step.value = 2  # y skip 1px
    s.slice_container[3].step.value = 2  # x
    new_layers = s.make_new_layer()
    assert isinstance(new_layers, napari.layers.Image)
