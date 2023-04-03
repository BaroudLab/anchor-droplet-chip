import napari
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ProgressBarWidget(QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer,
        total: int = 10,
        stop=lambda: print("STOP"),
        parent=None,
    ):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.total = total
        self.stop = stop
        self.initUI()

    def initUI(self):
        # Set widget properties
        self.setWindowTitle("Progress Bar Widget")
        self.setFont(QFont("Arial", 10))

        # Create widget components
        self.titleLabel = QLabel("Progress:", self)
        self.progressBar = QProgressBar(self)
        self.stopButton = QPushButton("Stop", self)

        # Set progress bar properties
        self.progressBar.setRange(0, self.total)
        self.progressBar.setValue(0)

        # Create vertical layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.progressBar)
        layout.addWidget(self.stopButton)

        # Connect stop button signal
        self.stopButton.clicked.connect(self.stop)

    def updateProgress(self, value):
        self.progressBar.setValue(value)

    def updateStatus(self, status):
        self.titleLabel.setText(f"Progress: {status}")

    def stop(self):
        self.stop()
        self.stopButton.setVisible(False)
