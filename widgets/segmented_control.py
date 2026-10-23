"""
widgets/segmented_control.py — Animated iOS-style sliding segmented control.
"""
from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

import styles


class SegmentedControl(QWidget):
    currentIndexChanged = pyqtSignal(int)

    def __init__(self, items: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items = items
        self._index = 0
        self._hover_index = -1
        self._pill_pos = 0.0

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._anim = QPropertyAnimation(self, b"pillPos", self)
        self._anim.setDuration(260)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    pillPos = styles.animated_float_property("_pill_pos")

    def currentIndex(self) -> int:
        return self._index

    def setCurrentIndex(self, index: int, animate: bool = True) -> None:
        if not (0 <= index < len(self._items)) or index == self._index:
            return
        self._index = index
        self._anim.stop()
        if animate:
            self._anim.setStartValue(self._pill_pos)
            self._anim.setEndValue(float(index))
            self._anim.start()
        else:
            self._pill_pos = float(index)
            self.update()
        self.currentIndexChanged.emit(index)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        n = max(1, len(self._items))
        idx = int(event.position().x() // (self.width() / n))
        idx = max(0, min(n - 1, idx))
        if idx != self._hover_index:
            self._hover_index = idx
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover_index = -1
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._items:
            n = len(self._items)
            idx = int(event.position().x() // (self.width() / n))
            self.setCurrentIndex(max(0, min(n - 1, idx)), animate=True)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = QRectF(self.rect()).adjusted(1.0, 1.0, -1.0, -1.0)
        track = styles.squircle_path(rect, 11.0)
        painter.fillPath(track, QBrush(QColor(styles.BG_DEEP)))
        painter.strokePath(track, QPen(QColor(styles.BORDER), 1.0))

        n = len(self._items)
        if n == 0:
            return

        pad = 3.5
        inner_w = rect.width() - pad * 2
        seg_w = inner_w / n
        pill_x = rect.left() + pad + self._pill_pos * seg_w
        pill_rect = QRectF(pill_x, rect.top() + pad, seg_w, rect.height() - pad * 2)

        pill_path = styles.squircle_path(pill_rect, 8.5)
        painter.fillPath(pill_path, QBrush(QColor(styles.BG_HOVER)))
        painter.strokePath(pill_path, QPen(styles.rgba(styles.ACCENT_CYAN, 0.38), 1.0))

        font_active = styles.ui_font(9.5, QFont.Weight.DemiBold)
        font_idle = styles.ui_font(9.5, QFont.Weight.Medium)

        for i, label in enumerate(self._items):
            item_rect = QRectF(rect.left() + pad + i * seg_w, rect.top(), seg_w, rect.height())
            dist = abs(self._pill_pos - i)
            t = styles.clamp(1.0 - dist)
            painter.setFont(font_active if t > 0.5 else font_idle)
            color = styles.mix(styles.TEXT_SECONDARY, styles.TEXT_PRIMARY, t)
            if i == self._hover_index and t < 0.2:
                color = QColor(styles.TEXT_PRIMARY)
            painter.setPen(color)
            painter.drawText(item_rect, int(Qt.AlignmentFlag.AlignCenter), label)