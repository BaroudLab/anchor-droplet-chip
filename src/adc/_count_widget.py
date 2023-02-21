import json
import os
from asyncio.log import logger
from functools import partial, reduce
from operator import add

import pandas as pd
from magicgui.widgets import Container, create_widget
from napari import Viewer
from napari.layers import Image, Points
from napari.utils import progress
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QLineEdit, QPushButton, QVBoxLayout, QWidget

from adc import count

DETECTION_LAYER_NAME = "Detections"
COUNTS_LAYER_NAME = "Counts"


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
            fluo = (
                self.viewer.layers[self.select_TRITC.current_choice]
                .data[0]  # maximum resilution from the piramide
                .compute()
            )  # max resolution
            prb.close()
        centers = (
            centers_layer := self.viewer.layers[
                self.select_centers.current_choice
            ]
        ).data
        try:
            self.df = pd.DataFrame(data=centers, columns=["chip", "y", "x"])
        except ValueError:
            show_error("Choose the right layer with actual localizations")
            return
        show_info("Data loaded. Counting")
        self.viewer.window._status_bar._toggle_activity_dock(True)
        peaks_raw = list(
            map(
                partial(
                    count.get_global_coordinates_from_well_coordinates,
                    fluo=fluo,
                    size=self.radius,
                ),
                progress(centers, desc="Localizing:"),
            )
        )
        show_info("Done localizing")
        n_peaks_per_well = list(map(len, peaks_raw))
        detections = reduce(add, peaks_raw)

        self.viewer.add_points(
            centers_layer.data,
            name=COUNTS_LAYER_NAME,
            text=n_peaks_per_well,
            face_color="#ffffff00",
            edge_color="#ff007f00",
        )

        self.viewer.add_points(
            detections,
            name=DETECTION_LAYER_NAME,
            size=20,
            face_color="#ffffff00",
            edge_color="#ff007f88",
        )
        try:
            path = self.viewer.layers[
                self.select_TRITC.current_choice
            ].metadata["path"]
            self.viewer.layers[DETECTION_LAYER_NAME].save(
                os.path.join(path, ".detections.csv")
            )
            with open(os.path.join(path, ".counts.json"), "w") as fp:
                json.dump(n_peaks_per_well, fp, indent=2)
        except Exception as e:
            logger.error(f"Saving detections failed: {e}")

    def show_counts(self, counts):
        self.counts = counts
        print(counts)

    def _update_path(self):
        BF = self.select_BF.current_choice
        TRITC = self.select_TRITC.current_choice
        maxz = "maxZ" if self.zmax_box.checkState() > 0 else ""
        self.out_path = "_".join((BF, TRITC, maxz)) + ".zarr"
        print(self.out_path)
        self.output_filename_widget.setText(self.out_path)
        self._combine(dry_run=True)

    def reset_choices(self, event=None):
        self.select_centers.reset_choices(event)
        self.select_TRITC.reset_choices(event)
