import os
from threading import Thread

import dask.array as da
from magicgui.widgets import (
    CheckBox,
    Container,
    Label,
    LineEdit,
    create_widget,
)
from napari import Viewer
from napari.layers import Image, Layer
from napari.layers._multiscale_data import MultiScaleData
from napari.utils import progress
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from zarr_tools.convert import to_zarr

from adc import _sample_data


class CombineStack(QWidget):
    "Combines brightfield with tritc with max projection"

    def __init__(self, napari_viewer: Viewer) -> None:
        super().__init__()
        self.viewer = napari_viewer
        self.select_BF = create_widget(
            annotation=Image,
            label="BF",
        )
        self.select_TRITC = create_widget(label="TRITC", annotation=Layer)

        self.select_BF.changed.connect(self._update_path)
        self.select_TRITC.changed.connect(self._update_path)

        self.out_path = ""
        self.output_filename_widget = LineEdit(
            label="path", bind=self.out_path
        )
        self.zmax_box = CheckBox(label="Z max project")
        self.zmax_box.changed.connect(self._update_path)
        self.btn = QPushButton("Combine!")
        self.btn.clicked.connect(self._combine)

        c = Container(
            widgets=[
                self.select_BF,
                self.select_TRITC,
                self.zmax_box,
                Label(label="Output filename"),
                self.output_filename_widget,
            ]
        )
        self.layout = QVBoxLayout()
        self.layout.addWidget(c.native)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()

        # self.viewer.layers.events.inserted.connect(self.reset_choices)
        # self.viewer.layers.events.removed.connect(self.reset_choices)
        # self.reset_choices(self.viewer.layers.events.inserted)

        self.setLayout(self.layout)
        _sample_data.make_template()

    def _update_path(self):
        BF = self.select_BF.current_choice
        TRITC = self.select_TRITC.current_choice
        maxz = "_maxZ" if self.zmax_box.value else "_2D"
        self.out_path = "_".join((BF, TRITC)) + maxz + ".zarr"
        print(self.out_path)
        self.output_filename_widget.value = self.out_path
        self._combine(dry_run=True)

    def _combine(self, dry_run=False):
        BF = self.viewer.layers[self.select_BF.current_choice]
        TRITC = self.viewer.layers[self.select_TRITC.current_choice]
        if any([isinstance(a.data, MultiScaleData) for a in [BF, TRITC]]):
            return
        print(
            f"""
        combining {(BF_path := BF.metadata["path"])}, \
            {(TRITC_path := TRITC.metadata["path"])},
            {(zmax := self.zmax_box.value)}
            """
        )

        if zmax:
            fd2d = TRITC.data.max(axis=1)
            self.viewer.add_image(
                data=[
                    fd2d[..., :: 2**i, :: 2**i]
                    for i in progress(
                        range(4), desc="Compute multiscale TRITC maxZ"
                    )
                ],
                name=self.select_TRITC.current_choice + "_maxZ",
            )
            self.viewer.layers[self.select_BF.current_choice].data = [
                BF.data[..., :: 2**i, :: 2**i]
                for i in progress(range(4), desc="Compute multiscale BF")
            ]
        else:
            fd2d = TRITC.data
        bd2d = da.stack([BF.data, fd2d], axis=1)
        show_info(f"Resulting stack: {bd2d}")
        self.layout.addWidget(QLabel(str(bd2d.shape)))
        zarr_path = os.path.join(
            os.path.commonpath((BF_path, TRITC_path)), self.out_path
        )
        show_info("file will be saved to:" + zarr_path)

        if not dry_run:
            show_info("start generating zarr")
            t = Thread(
                target=to_zarr,
                daemon=True,
                args=(bd2d, zarr_path),
                kwargs=dict(
                    steps=4,
                    name=["BF", "TRITC"],
                    colormap=["gray", "green"],
                    lut=((30000, 40000), (440, 600)),
                ),
            )
            t.start()
        return zarr_path

    def reset_choices(self, event=None):
        self.select_BF.reset_choices(event)
        self.select_TRITC.reset_choices(event)
