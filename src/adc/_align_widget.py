import logging
import os
from asyncio.log import logger
from functools import partial

import dask.array as da
import numpy as np
from magicgui.widgets import Container, create_widget
from napari import Viewer, layers
from napari.layers import Image, Layer, Points
from napari.utils import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from tqdm import tqdm

from adc import _sample_data, align

from .tools.log_decorator import log

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


DROPLETS_LAYER_PROPS = dict(
    name="Droplets",
    size=300,
    face_color="#00000000",
    edge_color="#00880088",
)
DROPLETS_CSV_SUFFIX = ".droplets.csv"

CONSTRAINTS = {
    "scale": [1, 0.1],
    "tx": [0, 150],
    "ty": [0, 150],
    "angle": [0, 10],
}


class DetectWells(QWidget):
    "Finds the droplets using template"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_image = create_widget(label="BF - data", annotation=Image)
        self.select_template = create_widget(
            label="BF - template", annotation=Layer
        )
        self.select_centers = create_widget(
            label="centers-template", annotation=Points
        )
        self.container = Container(
            widgets=[
                self.select_image,
                self.select_template,
                self.select_centers,
            ]
        )
        self.btn = QPushButton("Detect!")
        self.btn.clicked.connect(self._detect)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        self.viewer.layers.events.inserted.connect(self.reset_choices)
        self.viewer.layers.events.removed.connect(self.reset_choices)
        self.reset_choices(self.viewer.layers.events.inserted)

        self.setLayout(self.layout)
        img, centers = _sample_data.make_template()
        self.viewer.add_image(img[0], **img[1])
        self.viewer.add_points(centers[0], **centers[1])
        self.reset_choices()

    def _detect(self):
        logger.info("Start detecting")
        data_layer = self.viewer.layers[self.select_image.current_choice]

        assert isinstance(path := get_path(data_layer), str)
        logger.debug(f"data path: {path}")

        temp_layer = self.viewer.layers[self.select_template.current_choice]
        temp = temp_layer.data
        temp_scale = temp_layer.scale[0]
        centers_layer = self.viewer.layers[self.select_centers.current_choice]
        centers = centers_layer.data
        centers_scale = centers_layer.scale[0]
        ccenters = (
            centers * centers_scale / temp_scale - np.array(temp.shape) / 2.0
        )

        data, downscale = get_data(data_layer)

        locate_fun = partial(locate_wells, template=temp, positions=ccenters)

        self.positions, self.tvecs = align_recursive(
            data=data,
            template=temp,
            positions=ccenters,
            aligning_function=locate_fun,
            upscale=downscale,
            progress=progress,
        )

        droplets_layer = show_droplet_layer(self.viewer, self.positions)

        self.viewer.layers[self.select_centers.current_choice].visible = False
        self.viewer.layers[self.select_template.current_choice].visible = False

        try:
            path = data_layer.source.path
            droplets_layer.save(ppp := os.path.join(path, DROPLETS_CSV_SUFFIX))
        except Exception as e:
            logger.debug(f"Unable to save detection inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")
            droplets_layer.save(
                ppp := os.path.join(path + DROPLETS_CSV_SUFFIX)
            )
        logger.info(f"Saving detections into {ppp}")

    def reset_choices(self, event=None):
        self.select_image.reset_choices(event)
        self.select_template.reset_choices(event)
        self.select_centers.reset_choices(event)


@log
def locate_wells(
    data: np.ndarray,
    template: np.ndarray,
    positions: list,
    pad_ratio=1.3,
    constraints=CONSTRAINTS,
):
    """
    Compares bf image with template to get rigit transform  zoom, rotation and shift.
    Then moves the positions (droplet coordinates of the template) accordingly
    to get them on top of the bf droplets.
    Parameters:
    -----------
    bf: np.array 2D
    template: np.array 2D
    positions: list [[y0,x0],...]
        centers - are the droplet centers,
        positions mean with the zero coordinates in the center of the image
    """
    try:
        tvec = align.get_transform(
            image=data,
            template=template,
            constraints=constraints,
            pad_ratio=pad_ratio,
        )
        logger.info(tvec)
        return move_centers(positions, tvec, data.shape), tvec
    except Exception as e:
        logger.error("Error locating wells: ", e)
        raise e


@log
def align_recursive(
    data: da.Array,
    template: np.ndarray,
    positions: list,
    index: list = [],
    progress=tqdm,
    aligning_function=locate_wells,
    upscale=8,
) -> list:
    """
    Recurcively aligning multi-dimentional arrays with the template
    by the chunks of 2d arrays.
    Parameters:
    -----------
    data: dask.array n-dimensional
    template: numpy array 2D
    positions: list 2D [[y0,x0], ...]
    upscale: int , default 8
        the data and template are normally downscaled 8 times to save time,
        so the coordinates should be upscaled the same to match with the image in the viewer.
    Return:
    --------
    positions, tvecs:

        positions: list of positions with the of the size (m,n) where
        m - is the total number of translated postitions
        n - number of dimesions of initial data.
        tvecs: list of tranform vectors (dicts)
    """
    logger.debug(f"align {data}")
    if data.ndim > 2:
        pos = []
        tvecs = []
        for i, d in enumerate(progress(data)):
            new_ind = index + [i]
            logger.debug(f"index {new_ind}")
            aligned_positions, tvec = align_recursive(
                data=d,
                template=template,
                positions=positions,
                index=new_ind,
                progress=progress,
                aligning_function=aligning_function,
            )
            pos += aligned_positions
            tvecs.append(tvec)
        return pos, tvecs
    else:
        data = data.compute() if isinstance(data, da.Array) else data
        coords, tvec = aligning_function(
            data=data, template=template, positions=positions
        )
        logger.debug(f"Finished aligning index {index}, tvec: {tvec}")
        pos = [index + list(o) for o in coords * upscale]
        logger.debug(f"Added index {index} to positions")

        tvec["timg"] = []
        return pos, tvec


@log
def get_data(data_layer: layers.Image, downscale=8):
    """
    Looks for 1/8 scaled array from multiscale dataset
    """
    if data_layer.multiscale:
        n_scales = len(data_layer.data)
        assert (
            n_scales >= 2
        ), "Weird multiscale, looking for 1/8 scale, or 1/4 at least"
        if isinstance(data_layer.data[n_scales - 1], da.Array):
            try:
                data = data_layer.data[3]
            except IndexError:
                data = data_layer.data[2][..., ::2, ::2]
        elif isinstance(data_layer.data[0], np.ndarray):
            try:
                data = data_layer.data[3]
            except IndexError:
                data = data_layer.data[2][..., ::2, ::2]
    else:
        data = data_layer.data[..., ::8, ::8]

    return data, downscale


@log
def get_path(data_layer):
    if (path := data_layer.source.path) is not None:
        return path
    elif (path := data_layer.metadata["path"]) is not None:
        return path
    else:
        raise ValueError("Unable to get path of the dataset")


@log
def show_droplet_layer(viewer, data):
    return viewer.add_points(data, **DROPLETS_LAYER_PROPS)


@log
def add_new_dim(centers, value):
    return np.concatenate(
        (np.ones((len(centers), 1)) * value, centers), axis=1
    )


@log
def rot(vec: np.ndarray, angle_deg: float, canvas_shape=(0, 0)):
    theta = angle_deg / 180.0 * np.pi
    pivot = np.array(canvas_shape) / 2.0

    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.array([[c, -s], [s, c]])

    return np.dot(vec - pivot, mat) + pivot


@log
def trans(vec: np.ndarray, tr_vec: np.ndarray):
    return vec + tr_vec


@log
def move_centers(centers, tvec: dict, figure_size: tuple):
    """
    applies tvec and moves the zero coodinates
    from the center to the edge of the figure.
    """
    return (
        rot(
            centers / tvec["scale"],
            tvec["angle"],
        )
        - tvec["tvec"]
        + np.array(figure_size) / 2.0
    )
