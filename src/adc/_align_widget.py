import logging
import os
from asyncio.log import logger
from functools import partial
from multiprocessing import Pool

import dask.array as da
import numpy as np
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image, Layer, Points
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from adc import _sample_data, align

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DROPLETS_LAYER_PROPS = dict(
    name="Droplets",
    size=300,
    face_color="#00000000",
    edge_color="#00880088",
)
DROPLETS_CSV_SUFFIX = ".droplets.csv"


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
        if data_layer.multiscale:
            n_scales = len(data_layer.data)
            assert (
                n_scales >= 2
            ), "Weird multiscale, looking for 1/8 scale, or 1/4 at least"
            if isinstance(data_layer.data[n_scales - 1], da.Array):
                try:
                    data = data_layer.data[3].compute()
                except IndexError:
                    data = data_layer.data[2][..., ::2, ::2].compute()
            elif isinstance(data_layer.data[0], np.ndarray):
                try:
                    data = data_layer.data[3]
                except IndexError:
                    data = data_layer.data[2][..., ::2, ::2]
        else:
            data = data_layer.data[..., ::8, ::8]
            if isinstance(data, da.Array):
                data = data.compute()

        temp = self.viewer.layers[self.select_template.current_choice].data
        centers = self.viewer.layers[self.select_centers.current_choice].data
        ccenters = centers / 8 - np.array(temp.shape) / 2.0

        if data.ndim == 2:
            data = (data,)
        p = Pool(cores := len(data))
        logger.info(f"Processing in parallel with {cores} cores")
        try:
            centers16 = p.map(
                partial(locate_wells, template=temp, ccenters=ccenters), data
            )
            self.aligned_centers = np.concatenate(
                [add_new_dim(c * 8, i) for i, c in enumerate(centers16)]
            )

            droplets_layer = show_droplet_layer(
                self.viewer, self.aligned_centers
            )

            self.viewer.layers[
                self.select_centers.current_choice
            ].visible = False
            self.viewer.layers[
                self.select_template.current_choice
            ].visible = False

        except Exception as e:
            logger.error(e)
        finally:
            p.close()

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


def get_path(data_layer):
    if (path := data_layer.source.path) is not None:
        return path
    elif (path := data_layer.metadata["path"]) is not None:
        return path
    else:
        raise ValueError("Unable to get path of the dataset")


def show_droplet_layer(viewer, data):
    return viewer.add_points(data, **DROPLETS_LAYER_PROPS)


def add_new_dim(centers, value):
    return np.concatenate(
        (np.ones((len(centers), 1)) * value, centers), axis=1
    )


def rot(vec: np.ndarray, angle_deg: float, canvas_shape=(0, 0)):
    theta = angle_deg / 180.0 * np.pi
    pivot = np.array(canvas_shape) / 2.0

    c = np.cos(theta)
    s = np.sin(theta)
    mat = np.array([[c, -s], [s, c]])

    return np.dot(vec - pivot, mat) + pivot


def trans(vec: np.ndarray, tr_vec: np.ndarray):
    return vec + tr_vec


def move_centers(centers, tvec: dict, figure_size):
    return (
        rot(
            centers / tvec["scale"],
            tvec["angle"],
        )
        - tvec["tvec"]
        + np.array(figure_size) / 2.0
    )


def locate_wells(bf, template, ccenters):
    try:
        tvec = align.get_transform(
            image=bf,
            template=template,
            constraints={
                "scale": [1, 0.1],
                "tx": [0, 150],
                "ty": [0, 150],
                "angle": [0, 10],
            },
            pad_ratio=1.3,
        )
        logger.info(tvec)
        return move_centers(ccenters, tvec, bf.shape)
    except Exception as e:
        logger.error("Error locating wells: ", e)
        return ccenters
