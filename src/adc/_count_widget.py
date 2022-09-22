from operator import add
import os
from threading import Thread

import dask.array as da
from magicgui.widgets import create_widget, Slider
from napari import Viewer
from napari.layers import Image, Points
from napari.utils.notifications import show_info
from napari.utils import progress
from napari.qt import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    
)

from adc import count
import pandas as pd
import numpy as np


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


        self.out_path = ""
        self.output_filename_widget = QLineEdit("path")
        self.btn = QPushButton("Localize!")
        self.btn.clicked.connect(self._update_detections)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.select_TRITC.native)
        self.layout.addWidget(self.select_centers.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        # self.viewer.layers.events.inserted.connect(self.reset_choices)
        # self.viewer.layers.events.removed.connect(self.reset_choices)
        # self.reset_choices(self.viewer.layers.events.inserted)

        self.setLayout(self.layout)

    
    def _update_detections(self):
        show_info('Loading the data')
        with progress(desc="Loading data") as prb:
            fluo = self.viewer.layers[self.select_TRITC.current_choice].data[0].compute() # max resolution
        centers = self.viewer.layers[self.select_centers.current_choice].data
        self.df = pd.DataFrame(data=centers, columns=["chip", "y", "x"])
        show_info('Data loaded. Counting')
        counts = []
        detections = []
        self.viewer.window._status_bar._toggle_activity_dock(True)
        for i, r in progress(self.df.iterrows(), total=2500, desc="wells"):
            out = count.get_peak_number(count.crop2d(fluo[int(r.chip)], (r.y,r.x), self.radius), return_pos=True)
            cnt, pos = out.values()
            counts.append(cnt)
            for yx in pos:
                global_yx = np.array(yx) + np.array((r.y,r.x)) - self.radius/2
                detections.append((int(r.chip), global_yx[0], global_yx[1]))
        self.df.loc[:, "counts"] = counts
        self.viewer.add_points(data=centers, properties=self.df, text="counts", size=self.radius)
        self.viewer.add_points(detections, size=20, face_color="#ffffff00", edge_color="#00ffff88")

    def show_counts(self, counts):
        self.counts  = counts
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

