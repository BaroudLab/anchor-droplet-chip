import logging
from functools import partial

import dask.array as da
import numpy as np
import pandas as pd
import torch
from cellpose import models
from magicgui.widgets import Container, SpinBox, create_widget
from napari import Viewer
from napari.layers import Image
from napari.qt.threading import thread_worker
from napari.utils import progress
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)


class SegmentYeast(QWidget):
    BTN_TEXT = "Segment!"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_image = create_widget(label="mCherry", annotation=Image)
        self.diam = SpinBox(label="diameter (px)", value=50)
        self.container = Container(widgets=[self.select_image, self.diam])
        self.btn = QPushButton(self.BTN_TEXT)
        self.btn.clicked.connect(self._detect)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.container.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()
        self.setLayout(self.layout)
        self.reset_choices()
        self.stop = False
        self.init_model()

    def _detect(self):
        self.layer = self.viewer.layers[self.select_image.current_choice]
        self.data = self.layer.metadata["dask_data"]
        self.path = self.layer.metadata["path"]
        logger.debug(f"detecting {self.data.shape}")
        self.p = progress(total=len(self.data), desc="segmenting")
        logger.debug(self.p)
        self.worker = self.segment()
        logger.debug(self.worker, "create")
        self.worker.yielded.connect(self.update_layer)
        logger.debug(self.worker, "yield")
        self.worker.finished.connect(self.close_progress)
        logger.debug(self.worker, "fin")
        self.worker.aborted.connect(self.close_progress)
        logger.debug(self.worker, "abort")
        self.worker.start()
        logger.debug(self.worker, "start")
        self.btn.clicked.disconnect()
        self.btn.clicked.connect(self.abort)
        self.btn.setText("STOP!")

    def init_model(self):
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_built()
            else torch.device("cuda")
        )
        self.model = models.Cellpose(
            model_type="cyto2", gpu=True, device=self.device
        )
        logger.debug(f"using {self.model.device.type}")
        self.op = partial(
            self.model.eval,
            channels=[0, 1],
            diameter=self.diam.value,
            batch_size=16,
        )

    def update_layer(self, data):
        logger.debug("update layer")
        labels, props = data
        self.labels = np.array(labels)
        logger.debug(self.labels.shape, len(props))
        self.df = pd.concat(pd.DataFrame(p, index=p["label"]) for p in props)
        self.df.loc[0] = [0] * len(self.df.columns)
        self.df = self.df.reset_index()
        try:
            self.viewer.layers[(layer_name := "cellpose")].data = self.labels
            self.viewer.layers["cellpose"].properties = self.df
        except KeyError:
            self.viewer.add_labels(
                self.labels, name=layer_name, properties=self.df
            )
        self.p.update()

    def abort(self):
        self.p.set_description("aborting")
        self.worker.quit()
        self.btn.setText("Aborting...")

    def close_progress(self):
        logger.debug("close progress")
        self.p.close()
        self.btn.clicked.disconnect()
        self.btn.clicked.connect(self._detect)
        self.btn.setText(self.BTN_TEXT)

    @thread_worker
    def segment(self):
        logger.debug("start segment")
        labels = []
        props = []
        for d in self.data:
            logger.debug(d.shape)
            if isinstance(d, da.Array):
                d = d.compute()
                logger.debug("compute dask array into memory")
            mask, _, _, _ = self.op(d)
            logger.debug(mask.shape, d[1].shape)
            prop = regionprops_table(
                label_image=mask,
                intensity_image=d[1],
                properties=("label", "area", "mean_intensity", "eccentricity"),
            )
            labels.append(mask)
            props.append(prop)
            logger.debug(f"yielding labels, props")
            yield (labels, props)

    def reset_choices(self, event=None):
        self.select_image.reset_choices(event)
