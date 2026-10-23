"""
main.py — Main Studio Window & Worker Thread orchestration for AI Image Hub.
"""
from __future__ import annotations

import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget,
    QVBoxLayout, QWidget,
)

import engine
import styles
from widgets.drop_zone import DropZone
from widgets.segmented_control import SegmentedControl
from widgets.slider_view import SliderView


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    rgba_img = img.convert("RGBA")
    data = rgba_img.tobytes("raw", "RGBA")
    qimg = QImage(data, rgba_img.width, rgba_img.height, rgba_img.width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class InferenceWorker(QThread):
    progressChanged = pyqtSignal(float, str)
    finishedImage = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(
        self,
        ai_engine: engine.AIEngine,
        image: Image.Image,
        mode: str,
        model_choice: Optional[str],
    ) -> None:
        super().__init__()
        self.ai_engine = ai_engine
        self.image = image
        self.mode = mode
        self.model_choice = model_choice
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            out = self.ai_engine.process(
                image=self.image,
                mode=self.mode,
                model_choice=self.model_choice,
                progress=lambda frac, msg: self.progressChanged.emit(frac, msg),
                cancel_event=self.cancel_event,
            )
            self.finishedImage.emit(out)
        except engine.ProcessingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    MODES = [engine.MODE_CLEAR, engine.MODE_UPSCALE, engine.MODE_REMBG]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Image Hub")
        self.setWindowIcon(styles.make_app_icon())
        self.resize(1120, 720)
        self.setMinimumSize(900, 600)

        self.ai_engine = engine.AIEngine(models_dir="models")
        self.input_path: Optional[Path] = None
        self.input_pil: Optional[Image.Image] = None
        self.output_pil: Optional[Image.Image] = None
        self.worker: Optional[InferenceWorker] = None

        self._build_ui()
        self._refresh_model_dropdown()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(20, 16, 20, 18)
        main_layout.setSpacing(12)

        # ── Top Header Bar ──
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        lbl_title = QLabel("AI Image Hub")
        lbl_title.setObjectName("AppTitle")
        lbl_sub = QLabel("Local GPU Image Restoration, Super-Resolution & Cutout Studio")
        lbl_sub.setObjectName("AppSubtitle")
        title_col.addWidget(lbl_title)
        title_col.addWidget(lbl_sub)
        header.addLayout(title_col)
        header.addStretch(1)

        badge = QLabel(self.ai_engine.device_info.label)
        badge.setObjectName("DeviceBadge")
        badge.setToolTip(self.ai_engine.device_info.tooltip)
        if self.ai_engine.device_info.backend != "cuda":
            badge.setProperty("cpu", "true")
        header.addWidget(badge)
        main_layout.addLayout(header)

        # ── Center Workspace (Viewport + Right Control Panel) ──
        workspace = QHBoxLayout()
        workspace.setSpacing(16)

        self.stack = QStackedWidget()
        self.drop_zone = DropZone()
        self.drop_zone.fileDropped.connect(self.load_file)
        self.slider_view = SliderView()
        self.stack.addWidget(self.drop_zone)
        self.stack.addWidget(self.slider_view)
        workspace.addWidget(self.stack, stretch=1)

        # Right Control Panel
        panel = QFrame()
        panel.setFixedWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 14, 12, 14)
        panel_layout.setSpacing(14)

        lbl_mode = QLabel("PROCESSING MODE")
        lbl_mode.setObjectName("Caption")
        panel_layout.addWidget(lbl_mode)

        self.seg_mode = SegmentedControl(["Clear 1x", "Upscale 4x", "Remove BG"])
        self.seg_mode.currentIndexChanged.connect(self._on_mode_changed)
        panel_layout.addWidget(self.seg_mode)

        lbl_model = QLabel("AI MODEL / ENGINE")
        lbl_model.setObjectName("Caption")
        panel_layout.addWidget(lbl_model)

        self.combo_model = QComboBox()
        panel_layout.addWidget(self.combo_model)

        self.btn_import_model = QPushButton("+ Import Custom Model...")
        self.btn_import_model.clicked.connect(self._import_custom_model)
        panel_layout.addWidget(self.btn_import_model)

        panel_layout.addSpacing(8)
        lbl_info = QLabel("IMAGE DETAILS")
        lbl_info.setObjectName("Caption")
        panel_layout.addWidget(lbl_info)

        self.lbl_status = QLabel("No image loaded")
        self.lbl_status.setObjectName("Status")
        self.lbl_detail = QLabel("Drop a file into the viewport to begin.")
        self.lbl_detail.setObjectName("StatusDetail")
        self.lbl_detail.setWordWrap(True)
        panel_layout.addWidget(self.lbl_status)
        panel_layout.addWidget(self.lbl_detail)

        self.progress = QProgressBar()
        self.progress.setObjectName("Thin")
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        panel_layout.addWidget(self.progress)

        panel_layout.addStretch(1)

        # Action Buttons
        self.btn_run = QPushButton("Enhance Image")
        self.btn_run.setProperty("accent", "true")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run_clicked)
        panel_layout.addWidget(self.btn_run)

        self.btn_new = QPushButton("Load Another Image...")
        self.btn_new.clicked.connect(self._browse_new_image)
        panel_layout.addWidget(self.btn_new)

        export_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Image")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_output)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self._copy_output)
        export_row.addWidget(self.btn_save)
        export_row.addWidget(self.btn_copy)
        panel_layout.addLayout(export_row)

        workspace.addWidget(panel)
        main_layout.addLayout(workspace, stretch=1)

    def _current_mode(self) -> str:
        return self.MODES[self.seg_mode.currentIndex()]

    def _on_mode_changed(self, _index: int) -> None:
        self._refresh_model_dropdown()

    def _refresh_model_dropdown(self) -> None:
        mode = self._current_mode()
        self.combo_model.clear()
        if mode == engine.MODE_REMBG:
            self.btn_import_model.setEnabled(False)
            for sid, label in engine.REMBG_SESSIONS:
                self.combo_model.addItem(label, userData=sid)
        else:
            self.btn_import_model.setEnabled(True)
            self.combo_model.addItem("Auto (Recommended)", userData="Auto (Recommended)")
            for mf in self.ai_engine.scan_models():
                self.combo_model.addItem(f"{mf.name}  ({mf.size_mb:.1f} MB)", userData=mf.name)

    def _import_custom_model(self) -> None:
        exts = " ".join(f"*{e}" for e in engine.MODEL_EXTS)
        path, _ = QFileDialog.getOpenFileName(
            self, "Import AI Model Weights", "", f"AI Models ({exts})"
        )
        if not path:
            return
        src = Path(path)
        dst = self.ai_engine.models_dir / src.name
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            self._refresh_model_dropdown()
            idx = self.combo_model.findData(dst.name)
            if idx >= 0:
                self.combo_model.setCurrentIndex(idx)
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Failed to import model:\n{exc}")

    def _browse_new_image(self) -> None:
        exts = " ".join(f"*{e}" for e in engine.SUPPORTED_IMAGE_EXTS)
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", f"Images ({exts})")
        if path:
            self.load_file(path)

    def load_file(self, path_str: str) -> None:
        try:
            img = engine.load_image(path_str)
        except Exception as exc:
            QMessageBox.warning(self, "Open Image Failed", str(exc))
            return

        self.input_path = Path(path_str)
        self.input_pil = img
        self.output_pil = None

        pm = pil_to_qpixmap(img)
        self.drop_zone.setPreview(pm, path_str)
        self.stack.setCurrentWidget(self.drop_zone)

        self.lbl_status.setText(self.input_path.name)
        self.lbl_detail.setText(f"Original: {img.width} × {img.height} px  ·  Ready")
        self.progress.setValue(0)
        self.btn_run.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_copy.setEnabled(False)

    def _on_run_clicked(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            return
        if not self.input_pil:
            return

        mode = self._current_mode()
        choice = self.combo_model.currentData()

        self.stack.setCurrentWidget(self.drop_zone)
        self.drop_zone.setScanning(True, "AI Scanning & Enhancing...")
        self.btn_run.setText("Cancel")
        self.btn_run.setProperty("accent", "false")
        self.btn_run.setProperty("danger", "true")
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

        self.worker = InferenceWorker(self.ai_engine, self.input_pil, mode, choice)
        self.worker.progressChanged.connect(self._on_worker_progress)
        self.worker.finishedImage.connect(self._on_worker_finished)
        self.worker.failed.connect(self._on_worker_failed)
        self.worker.cancelled.connect(self._on_worker_cancelled)
        self.worker.start()

    def _reset_run_button(self) -> None:
        self.drop_zone.setScanning(False)
        self.btn_run.setText("Enhance Image")
        self.btn_run.setProperty("danger", "false")
        self.btn_run.setProperty("accent", "true")
        self.btn_run.style().unpolish(self.btn_run)
        self.btn_run.style().polish(self.btn_run)

    def _on_worker_progress(self, frac: float, msg: str) -> None:
        self.progress.setValue(int(frac * 1000))
        self.lbl_detail.setText(msg)
        self.drop_zone.setScanning(True, msg)

    def _on_worker_finished(self, out_img: Image.Image) -> None:
        self._reset_run_button()
        self.output_pil = out_img
        self.progress.setValue(1000)

        assert self.input_pil is not None
        before_pm = pil_to_qpixmap(self.input_pil)
        after_pm = pil_to_qpixmap(out_img)
        self.slider_view.setImages(before_pm, after_pm, animate_reveal=True)
        self.stack.setCurrentWidget(self.slider_view)

        self.lbl_detail.setText(
            f"{self.input_pil.width}×{self.input_pil.height}  ➔  {out_img.width}×{out_img.height} px"
        )
        self.btn_save.setEnabled(True)
        self.btn_copy.setEnabled(True)

    def _on_worker_failed(self, err: str) -> None:
        self._reset_run_button()
        self.progress.setValue(0)
        self.lbl_detail.setText("Error during processing.")
        QMessageBox.critical(self, "Processing Error", err)

    def _on_worker_cancelled(self) -> None:
        self._reset_run_button()
        self.progress.setValue(0)
        self.lbl_detail.setText("Processing cancelled.")

    def _save_output(self) -> None:
        if not self.output_pil:
            return
        stem = self.input_path.stem if self.input_path else "image"
        mode = self._current_mode()
        default_name = f"{stem}_{mode}.png"
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Image", default_name, "PNG Image (*.png);;JPEG Image (*.jpg);;WebP (*.webp)"
        )
        if out_path:
            save_img = self.output_pil
            if out_path.lower().endswith((".jpg", ".jpeg")) and save_img.mode == "RGBA":
                save_img = save_img.convert("RGB")
            save_img.save(out_path)
            self.lbl_detail.setText(f"Saved to {Path(out_path).name}")

    def _copy_output(self) -> None:
        if not self.output_pil:
            return
        QGuiApplication.clipboard().setPixmap(pil_to_qpixmap(self.output_pil))
        self.lbl_detail.setText("Copied result to clipboard!")


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(styles.dark_palette())
    app.setStyleSheet(styles.build_stylesheet())

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()