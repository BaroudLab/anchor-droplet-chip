import json
import logging
import os
from asyncio.log import logger

import dask.array as da
import numpy as np
import pandas as pd
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

from adc import count, _sample_data

from ._align_widget import DROPLETS_CSV_SUFFIX

TABLE_NAME = "table.csv"

COUNTS_LAYER_PROPS = dict(
    name="Counts", size=300, face_color="#00000000", edge_color="#00880088"
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
logger.setLevel(logging.INFO)


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
        self.out = []
        self.counts_layer = None
        self.detections_layer = None

        self.out_path = ""
        self.output_filename_widget = QLineEdit("path")
        self.btn = QPushButton("Localize!")
        self.btn.clicked.connect(self.process_stack)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        # self.viewer.layers.events.inserted.connect(self.reset_choices)
        # self.viewer.layers.events.removed.connect(self.reset_choices)
        # self.reset_choices(self.viewer.layers.events.inserted)

        self.setLayout(self.layout)

        if not "centers" in self.viewer.layers:
            centers = _sample_data.make_centers()[0]
            self.viewer.add_points(centers[0], **centers[1])
            self.reset_choices()
            
    def process_stack(self):
        self._pick_data_ref()
        self._pick_centers()

        show_info("Data loaded. Counting")

        self.viewer.window._status_bar._toggle_activity_dock(True)

        self._update_detections()

    def _pick_data_ref(self):
        "Get dask array to know the shape etc"
        self.selected_layer = self.viewer.layers[
            self.select_TRITC.current_choice
        ]
        logger.debug(f"selected_layer: {self.selected_layer}")
        if self.selected_layer.multiscale:
            self.ddata_ref = self.selected_layer.data[0]
            logger.debug(
                f"multiscale data: selecting highest resolution: {self.ddata_ref}"
            )
        else:
            self.ddata_ref = self.selected_layer.data
            logger.debug(f"not multiscale data: {self.ddata_ref}")

    def _pick_centers(self):
        self.centers_layer = self.viewer.layers[
            self.select_centers.current_choice
        ]
        self.centers = self.centers_layer.data
        logger.debug(f"selected centers: {len(self.centers)}")

    def _update_detections(self):
        logger.debug("Creating output layers")
        self.detections_layer = self.viewer.add_points(
            data=[[0] * self.ddata_ref.ndim], **DETECTION_LAYER_PROPS
        )
        self.counts_layer = self.viewer.add_points(
            data=[[0] * self.ddata_ref.ndim], text=[], **COUNTS_LAYER_PROPS
        )
        logger.debug("Creating worker")
        self.out = count.count_recursive(
            data=self.ddata_ref,
            positions=self.centers,
            size=self.radius,
            progress=progress,
        )
        self.save_results()

    def save_results(self):
        show_info("Done localizing ")

        locs, n_peaks_per_well, drops, table_df = self.out

        self.detections_layer.data = locs
        self.counts_layer.data = drops
        self.counts_layer.text = n_peaks_per_well

        try:
            path = self.selected_layer.source.path
            if path is None:
                try:
                    path = self.selected_layer.metadata["path"]
                except KeyError:
                    show_error("Unable to find the path")
                    return
            self.detections_layer.save(
                ppp := os.path.join(path, DETECTION_CSV_SUFFIX)
            )
        except Exception as e:
            logger.debug(f"Unable to save detections inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")
            self.detections_layer.save(
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

        try:
            ppp = os.path.join(path, DROPLETS_CSV_SUFFIX)
            droplets_df = pd.DataFrame(
                data=drops, columns=[f"axis-{i}" for i in range(len(drops[0]))]
            )
            droplets_df.to_csv(ppp)
        except Exception as e:
            logger.debug(f"Unable to save droplets inside the zarr: {e}")
            logger.debug(f"Saving in a separate file")

            droplets_df.to_csv(ppp := path + DROPLETS_CSV_SUFFIX)
        logger.info(f"Saving counts into {ppp}")

        try:
            ppp = os.path.join(os.path.dirname(path), TABLE_NAME)

            table_df.to_csv(ppp)
            logger.info(f"Saving table into {ppp}")
        except Exception as e:
            logger.error(f"Unable to save table into {ppp}: {e}")

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
