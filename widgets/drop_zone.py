"""
widgets/drop_zone.py — Magnetic Aura Drop Zone + Bioluminescent Laser Scanner.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer, pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush, QColor, QConicalGradient, QDragEnterEvent, QDragLeaveEvent,
    QDropEvent, QFont, QLinearGradient, QMouseEvent, QPainter, QPainterPath,
    QPen, QPixmap,
)
from PyQt6.QtWidgets import QFileDialog, QWidget

import engine
import styles


class DropZone(QWidget):
    fileDropped = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._pixmap: Optional[QPixmap] = None
        self._image_path: Optional[str] = None
        self._scanning = False
        self._status_text = ""

        self._hover_aura = 0.0
        self._angle = 0.0
        self._laser_y = 0.0

        self._aura_anim = QPropertyAnimation(self, b"hoverAura", self)
        self._aura_anim.setDuration(240)
        self._aura_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # ~60 FPS
        self._timer.timeout.connect(self._tick)

    hoverAura = styles.animated_float_property("_hover_aura", 0.0, 1.0)

    def setPreview(self, pixmap: Optional[QPixmap], path: Optional[str] = None) -> None:
        self._pixmap = pixmap
        self._image_path = path
        self.update()

    def setScanning(self, active: bool, status: str = "Processing with AI...") -> None:
        self._scanning = active
        self._status_text = status
        if active:
            self._laser_y = 0.0
            if not self._timer.isActive():
                self._timer.start()
        elif self._hover_aura <= 0.01:
            self._timer.stop()
        self.update()

    def _tick(self) -> None:
        self._angle = (self._angle + 3.2) % 360.0
        if self._scanning:
            self._laser_y = (self._laser_y + 0.016) % 1.0
        if not self._scanning and self._hover_aura <= 0.01:
            self._timer.stop()
        self.update()

    def _animate_aura(self, target: float) -> None:
        self._aura_anim.stop()
        self._aura_anim.setStartValue(self._hover_aura)
        self._aura_anim.setEndValue(target)
        self._aura_anim.start()
        if target > 0 and not self._timer.isActive():
            self._timer.start()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in engine.SUPPORTED_IMAGE_EXTS:
                    event.acceptProposedAction()
                    self._animate_aura(1.0)
                    return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._animate_aura(0.0)

    def dropEvent(self, event: QDropEvent) -> None:
        self._animate_aura(0.0)
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if Path(local).suffix.lower() in engine.SUPPORTED_IMAGE_EXTS:
                self.fileDropped.emit(local)
                break

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._scanning:
            exts = " ".join(f"*{e}" for e in engine.SUPPORTED_IMAGE_EXTS)
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", f"Images ({exts})"
            )
            if path:
                self.fileDropped.emit(path)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        outer = QRectF(self.rect()).adjusted(14.0, 14.0, -14.0, -14.0)
        scale = 1.0 + 0.012 * self._hover_aura
        cx, cy = outer.center().x(), outer.center().y()
        w, h = outer.width() * scale, outer.height() * scale
        rect = QRectF(cx - w / 2, cy - h / 2, w, h)

        card_path = styles.squircle_path(rect, styles.RADIUS_PANEL)
        painter.fillPath(card_path, QBrush(QColor(styles.BG_PANEL)))

        if self._pixmap and not self._pixmap.isNull():
            painter.save()
            painter.setClipPath(card_path)
            inner = rect.adjusted(20, 20, -20, -20)
            pw, ph = self._pixmap.width(), self._pixmap.height()
            ratio = min(inner.width() / pw, inner.height() / ph)
            dw, dh = pw * ratio, ph * ratio
            img_rect = QRectF(
                inner.center().x() - dw / 2,
                inner.center().y() - dh / 2,
                dw,
                dh,
            )
            painter.fillRect(img_rect, styles.checkerboard_brush(10))
            painter.drawPixmap(img_rect.toRect(), self._pixmap)

            if self._scanning:
                self._paint_laser(painter, img_rect)
            painter.restore()
        else:
            self._paint_empty_prompt(painter, rect)

        # Border & rotating conical aura
        if self._hover_aura > 0.01 or self._scanning:
            grad = QConicalGradient(rect.center(), -self._angle)
            grad.setColorAt(0.0, QColor(styles.ACCENT_CYAN))
            grad.setColorAt(0.25, QColor(styles.ACCENT_AZURE))
            grad.setColorAt(0.5, styles.rgba(styles.ACCENT_CYAN, 0.15))
            grad.setColorAt(0.75, QColor(styles.ACCENT_AZURE))
            grad.setColorAt(1.0, QColor(styles.ACCENT_CYAN))
            strength = max(self._hover_aura, 0.85 if self._scanning else 0.0)
            styles.stroke_glow(painter, card_path, QBrush(grad), width=2.0, spread=14.0, strength=strength)
        else:
            painter.strokePath(card_path, QPen(QColor(styles.BORDER_STRONG), 1.2))

    def _paint_laser(self, painter: QPainter, img_rect: QRectF) -> None:
        painter.fillRect(img_rect, styles.rgba("#080A0E", 0.35))
        scan_y = img_rect.top() + self._laser_y * img_rect.height()

        band_h = 56.0
        trail = QRectF(img_rect.left(), scan_y - band_h, img_rect.width(), band_h)
        grad = QLinearGradient(trail.topLeft(), trail.bottomLeft())
        grad.setColorAt(0.0, styles.rgba(styles.ACCENT_CYAN, 0.0))
        grad.setColorAt(1.0, styles.rgba(styles.ACCENT_CYAN, 0.28))
        painter.fillRect(trail.intersected(img_rect), QBrush(grad))

        line_path = QPainterPath()
        line_path.moveTo(img_rect.left(), scan_y)
        line_path.lineTo(img_rect.right(), scan_y)
        styles.stroke_glow(painter, line_path, styles.ACCENT_CYAN, width=2.2, spread=12.0, strength=1.0)

        if self._status_text:
            f = styles.ui_font(9.5, QFont.Weight.DemiBold)
            pw = styles.pill_width(self._status_text, f, dot=True)
            pill_r = QRectF(img_rect.center().x() - pw / 2, img_rect.bottom() - 42, pw, 30)
            styles.draw_pill(painter, pill_r, self._status_text, f, dot=QColor(styles.ACCENT_CYAN))

    def _paint_empty_prompt(self, painter: QPainter, rect: QRectF) -> None:
        c = rect.center()
        icon_r = QRectF(c.x() - 34, c.y() - 68, 68, 68)
        styles.paint_logo(painter, icon_r)

        painter.setFont(styles.ui_font(13, QFont.Weight.DemiBold))
        painter.setPen(QColor(styles.TEXT_PRIMARY))
        painter.drawText(
            QRectF(rect.left(), c.y() + 14, rect.width(), 28),
            int(Qt.AlignmentFlag.AlignCenter),
            "Drop Image Here or Click to Browse",
        )

        painter.setFont(styles.ui_font(9.5))
        painter.setPen(QColor(styles.TEXT_SECONDARY))
        painter.drawText(
            QRectF(rect.left(), c.y() + 42, rect.width(), 24),
            int(Qt.AlignmentFlag.AlignCenter),
            "Supports PNG, JPG, WebP, TIFF · GPU FP16 Accelerated",
        )