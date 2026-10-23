"""
widgets/slider_view.py — Interactive Before/After Split Curtain with Pan & Zoom.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QMouseEvent, QPainter, QPainterPath, QPen, QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget

import styles


class SliderView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._before: Optional[QPixmap] = None
        self._after: Optional[QPixmap] = None
        self._split = 0.5
        self._dragging_split = False
        self._panning = False
        self._last_mouse = QPointF()
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)

        self.setMouseTracking(True)

        self._reveal_anim = QPropertyAnimation(self, b"splitRatio", self)
        self._reveal_anim.setDuration(680)
        self._reveal_anim.setEasingCurve(QEasingCurve.Type.OutBack)

    splitRatio = styles.animated_float_property("_split", 0.02, 0.98)

    def setImages(self, before: QPixmap, after: QPixmap, animate_reveal: bool = True) -> None:
        self._before = before
        self._after = after
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._reveal_anim.stop()
        if animate_reveal:
            self._split = 0.06
            self._reveal_anim.setStartValue(0.06)
            self._reveal_anim.setEndValue(0.50)
            self._reveal_anim.start()
        else:
            self._split = 0.50
        self.update()

    def resetView(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _target_rect(self, canvas: QRectF) -> QRectF:
        if not self._after or self._after.isNull():
            return canvas
        pw, ph = self._after.width(), self._after.height()
        base_scale = min(canvas.width() / pw, canvas.height() / ph)
        dw = pw * base_scale * self._zoom
        dh = ph * base_scale * self._zoom
        cx = canvas.center().x() + self._pan.x()
        cy = canvas.center().y() + self._pan.y()
        return QRectF(cx - dw / 2, cy - dh / 2, dw, dh)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom = styles.clamp(self._zoom * factor, 0.5, 12.0)
        if abs(self._zoom - 1.0) < 0.05:
            self._pan = QPointF(0.0, 0.0)
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            split_x = self._current_split_x()
            if abs(event.position().x() - split_x) <= 24 or self._zoom <= 1.05:
                self._dragging_split = True
                self._update_split_from_x(event.position().x())
            else:
                self._panning = True
                self._last_mouse = event.position()
        elif event.button() == Qt.MouseButton.RightButton:
            self.resetView()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_split:
            self._update_split_from_x(event.position().x())
        elif self._panning:
            delta = event.position() - self._last_mouse
            self._last_mouse = event.position()
            self._pan += delta
            self.update()
        else:
            split_x = self._current_split_x()
            if abs(event.position().x() - split_x) <= 18:
                self.setCursor(Qt.CursorShape.SplitHCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging_split = False
        self._panning = False

    def _current_split_x(self) -> float:
        outer = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        return outer.left() + self._split * outer.width()

    def _update_split_from_x(self, x: float) -> None:
        outer = QRectF(self.rect()).adjusted(14, 14, -14, -14)
        if outer.width() > 0:
            self._split = styles.clamp((x - outer.left()) / outer.width(), 0.02, 0.98)
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._zoom < 3.0)

        outer = QRectF(self.rect()).adjusted(14.0, 14.0, -14.0, -14.0)
        card_path = styles.squircle_path(outer, styles.RADIUS_PANEL)
        painter.fillPath(card_path, QBrush(QColor(styles.BG_PANEL)))

        if not self._before or not self._after:
            painter.strokePath(card_path, QPen(QColor(styles.BORDER), 1.0))
            return

        painter.save()
        painter.setClipPath(card_path)

        inner = outer.adjusted(18, 18, -18, -18)
        dest = self._target_rect(inner)
        split_x = outer.left() + self._split * outer.width()

        # Draw checkerboard under transparent output
        painter.fillRect(dest, styles.checkerboard_brush(10))

        # Right side: AFTER (AI Enhanced)
        painter.save()
        painter.setClipRect(QRectF(split_x, outer.top(), outer.right() - split_x, outer.height()))
        painter.drawPixmap(dest.toRect(), self._after)
        painter.restore()

        # Left side: BEFORE (Original)
        painter.save()
        painter.setClipRect(QRectF(outer.left(), outer.top(), split_x - outer.left(), outer.height()))
        painter.drawPixmap(dest.toRect(), self._before)
        painter.restore()

        # glowing divider line
        line = QPainterPath()
        line.moveTo(split_x, outer.top())
        line.lineTo(split_x, outer.bottom())
        styles.stroke_glow(painter, line, styles.ACCENT_CYAN, width=1.8, spread=10.0, strength=0.9)

        # Center handle knob
        cy = outer.center().y()
        knob_r = QRectF(split_x - 16, cy - 16, 32, 32)
        knob_path = styles.squircle_path(knob_r, 16, smoothing=0.0)
        painter.fillPath(knob_path, QBrush(styles.rgba(styles.BG_DEEP, 0.90)))
        painter.strokePath(knob_path, QPen(QColor(styles.ACCENT_CYAN), 1.6))

        painter.setPen(QPen(QColor(styles.TEXT_PRIMARY), 1.6))
        painter.drawLine(QPointF(split_x - 6, cy), QPointF(split_x - 2, cy - 4))
        painter.drawLine(QPointF(split_x - 6, cy), QPointF(split_x - 2, cy + 4))
        painter.drawLine(QPointF(split_x + 6, cy), QPointF(split_x + 2, cy - 4))
        painter.drawLine(QPointF(split_x + 6, cy), QPointF(split_x + 2, cy + 4))

        # Pills: BEFORE / AFTER / Zoom
        f_tag = styles.ui_font(8.5, QFont.Weight.Bold)
        w_bef = styles.pill_width("BEFORE", f_tag)
        w_aft = styles.pill_width("AFTER", f_tag, dot=True)
        styles.draw_pill(painter, QRectF(outer.left() + 16, outer.top() + 16, w_bef, 26), "BEFORE", f_tag)
        styles.draw_pill(
            painter,
            QRectF(outer.right() - w_aft - 16, outer.top() + 16, w_aft, 26),
            "AFTER",
            f_tag,
            dot=QColor(styles.ACCENT_CYAN),
        )

        if abs(self._zoom - 1.0) > 0.05:
            z_txt = f"{int(self._zoom * 100)}% (Right-click reset)"
            w_z = styles.pill_width(z_txt, f_tag)
            styles.draw_pill(painter, QRectF(outer.left() + 16, outer.bottom() - 42, w_z, 26), z_txt, f_tag)

        painter.restore()
        painter.strokePath(card_path, QPen(QColor(styles.BORDER_STRONG), 1.2))