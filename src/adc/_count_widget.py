import json
import logging
import os
from asyncio.log import logger
from functools import partial, reduce
from operator import add

import dask.array as da
import numpy as np
import pandas as pd
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

from adc import count

COUNTS_LAYER_PROPS = dict(
    name="Counts",
    face_color="#ffffff00",
    edge_color="#ff007f00",
)
COUNTS_JSON_SUFFIX = ".counts.json"

DETECTION_LAYER_PROPS = dict(
    name="Detections",
    size=20,
    face_color="#ffffff00",
    edge_color="#ff007f88",
)
DETECTION_CSV_SUFFIX = ".detections.csv"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class CountCells(QWidget):
    "Detects cells in TRITC"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_TRITC = create_widget(
            annotation=Image,
            label="TRITC",
        )
        self.radius = 300
        self.select_centers = create_widget(label="centers", annotation=Points)
        self.container = Container(
            widgets=[self.select_TRITC, self.select_centers]
        )

        self.out_path = ""
        self.output_filename_widget = QLineEdit("path")
        self.btn = QPushButton("Localize!")
        self.btn.clicked.connect(self._update_detections)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        # self.viewer.layers.events.inserted.connect(self.reset_choices)
        # self.viewer.layers.events.removed.connect(self.reset_choices)
        # self.reset_choices(self.viewer.layers.events.inserted)

        self.setLayout(self.layout)

    def _update_detections(self):
        show_info("Loading the data")
        with progress(desc="Loading data") as prb:
            selected_layer = self.viewer.layers[
                self.select_TRITC.current_choice
            ]
            logger.debug(f"selected_layer: {selected_layer}")
            if selected_layer.multiscale:
                ddata = selected_layer.data[0]
                logger.debug(
                    f"multiscale data: selecting highest resolution: {ddata}"
                )
            else:
                ddata = selected_layer.data
                logger.debug(f"not multiscale data: {ddata}")
            if isinstance(ddata, da.Array):
                ddata = ddata.compute()
                logger.debug(f"compute dask array: {ddata}")
            if ddata.ndim == 2:
                ddata = np.reshape(ddata, (1, *ddata.shape))
                logger.debug(f"reshaping: {ddata}")
            else:
                logger.debug("Finished data loading")
            prb.close()

        centers = (
            centers_layer := self.viewer.layers[
                self.select_centers.current_choice
            ]
        ).data
        logger.debug(f"selected centers: {centers}")
        try:
            logger.debug(f"creating dataframe with columns ['chip', 'y', 'x']")
            self.df = pd.DataFrame(data=centers, columns=["chip", "y", "x"])
            logger.debug(f"created dataframe {self.df}")
        except ValueError as e:
            logger.debug(f"problem with dataframe {e}")
            show_error("Choose the right layer with actual localizations")
            return
        show_info("Data loaded. Counting")
        self.viewer.window._status_bar._toggle_activity_dock(True)
        peaks_raw = list(
            map(
                partial(
                    count.get_global_coordinates_from_well_coordinates,
                    fluo=ddata,
                    size=self.radius,
                ),
                progress(centers, desc="Localizing:"),
            )
        )
        show_info("Done localizing")
        n_peaks_per_well = list(map(len, peaks_raw))
        detections = reduce(add, peaks_raw)

        counts_layer = self.viewer.add_points(
            centers_layer.data, text=n_peaks_per_well, **COUNTS_LAYER_PROPS
        )

        detections_layer = self.viewer.add_points(
            detections, **DETECTION_LAYER_PROPS
        )
        try:
            path = selected_layer.source.path
            detections_layer.save(
                ppp := os.path.join(path, DETECTION_CSV_SUFFIX)
            )
            with open(
                pppc := os.path.join(path, COUNTS_JSON_SUFFIX), "w"
            ) as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        except Exception as e:
            logger.debug(f"Unable to save detections inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")
            detections_layer.save(
                ppp := os.path.join(path + DETECTION_CSV_SUFFIX)
            )
        logger.info(f"Saving detections into {ppp}")

        try:
            with open(
                ppp := os.path.join(path, COUNTS_JSON_SUFFIX), "w"
            ) as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        except Exception as e:
            logger.debug(f"Unable to save counts inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")

            with open(ppp := path + COUNTS_JSON_SUFFIX, "w") as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        logger.info(f"Saving counts into {ppp}")

    def show_counts(self, counts):
        self.counts = counts
        logger.debug(counts)

    def _update_path(self):
        BF = self.select_BF.current_choice
        TRITC = self.select_TRITC.current_choice
        maxz = "maxZ" if self.zmax_box.checkState() > 0 else ""
        self.out_path = "_".join((BF, TRITC, maxz)) + ".zarr"
        logger.debug(self.out_path)
        self.output_filename_widget.setText(self.out_path)
        self._combine(dry_run=True)

    def reset_choices(self, event=None):
        self.select_centers.reset_choices(event)
        self.select_TRITC.reset_choices(event)
