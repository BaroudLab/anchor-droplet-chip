import logging
import os
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

logging.basicConfig(level="DEBUG")
logger = logging.getLogger(__name__)


TIF_SUFFIX = "_CP_labels_BF_mCherry.tif"
CSV_SUFFIX = "_CP_labels_BF_mCherry.csv"


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

        logger.info(f"detecting  {self.data.shape} from {self.path}")
        save_path_tif = self.path.replace(".tif", TIF_SUFFIX)
        assert not os.path.exists(save_path_tif), f"{save_path_tif} exists"
        self.save_path_tif = save_path_tif
        save_path_csv = self.path.replace(".tif", CSV_SUFFIX)
        assert not os.path.exists(save_path_csv), f"{save_path_csv} exists"
        self.save_path_csv = save_path_csv

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
        self.status = "processing"

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
        logger.debug(f"Labels {self.labels.shape}, {len(props)} properties")
        self.df = pd.concat(pd.DataFrame(p, index=p["label"]) for p in props)
        self.df.loc[0] = [0] * len(self.df.columns)
        self.df = self.df.reset_index()
        try:
            self.out_layer.data = self.labels
            self.out_layer.properties = self.df
            self.out_layer.metadata["df"] = self.df
            self.out_layer.metadata["status"] = self.status
        except AttributeError:
            self.out_layer = self.viewer.add_labels(
                self.labels,
                name="cellpose",
                properties=self.df,
                metadata={"df": self.df, "status": self.status},
            )
        self.p.update()

    def abort(self):
        self.p.set_description("aborting")
        self.worker.quit()
        self.btn.setText("Aborting...")
        self.out_layer.metadata["status"] = "aborted"

    def close_progress(self):
        logger.debug("close progress")
        self.p.close()
        self.btn.clicked.disconnect()
        self.btn.clicked.connect(self._detect)
        self.btn.setText(self.BTN_TEXT)
        if self.status == "finished":
            self.out_layer.metadata["status"] = "finished"
        save = self.out_layer.save(self.save_path_tif)
        logger.info(f"saved {save}")
        self.df.to_csv(self.save_path_csv)
        logger.info(f"saved {self.save_path_csv}")

    @thread_worker
    def segment(self):
        logger.debug("start segment")
        labels = []
        props = []
        max_label = 0
        for frame, d in enumerate(self.data):
            logger.debug(d.shape)
            if isinstance(d, da.Array):
                d = d.compute()
                logger.debug("compute dask array into memory")
            bf, mCherry, GFP = d
            mask, _, _, _ = self.op(d)
            logger.debug(
                f"mask shape: {mask.shape}, mCheery shape: {mCherry.shape}"
            )

            mask = mask + max_label
            mask[mask == max_label] = 0
            max_label = mask.max()
            labels.append(mask)
            prop = regionprops_table(
                label_image=mask,
                intensity_image=mCherry,
                properties=(
                    "label",
                    "centroid",
                    "area",
                    "mean_intensity",
                    "eccentricity",
                    "solidity",
                ),
            )
            props.append({**prop, "frame": frame})
            logger.debug(f"yielding labels, props")
            yield (labels, props)
        self.status = "finished"

    def reset_choices(self, event=None):
        self.select_image.reset_choices(event)
