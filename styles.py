"""
styles.py — Design tokens, QSS and custom QPainter helpers for AI Image Hub.

"Space Black" design language:
  • Deep near-black canvases (#0D0E11 / #16181D)
  • Frosted-glass containers with hairline #262930 borders
  • Titanium trim highlights, Electric Cyan (#64D2FF) / Deep Azure (#0A84FF) accents
  • Continuous-curvature "squircle" geometry everywhere
"""
from __future__ import annotations

from typing import Optional, Union

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtProperty
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QIcon, QLinearGradient, QPainter,
    QPainterPath, QPalette, QPen, QPixmap,
)

# ─────────────────────────────── Palette ────────────────────────────────
BG_DEEP = "#0D0E11"
BG_SURFACE = "#16181D"
BG_PANEL = "#111216"
BG_ELEVATED = "#1D2026"
BG_HOVER = "#24272E"
BORDER = "#262930"
BORDER_STRONG = "#363A43"
TITANIUM = "#8D9096"
TITANIUM_LIGHT = "#D4D6D9"
TEXT_PRIMARY = "#F2F3F5"
TEXT_SECONDARY = "#A1A5AD"
TEXT_TERTIARY = "#62666E"
ACCENT_CYAN = "#64D2FF"
ACCENT_AZURE = "#0A84FF"
DANGER = "#FF453A"
WARNING = "#FFD60A"
SUCCESS = "#30D158"
CHECKER_DARK = "#17191D"
CHECKER_LIGHT = "#212329"

RADIUS_WINDOW = 14.0
RADIUS_PANEL = 22.0

ColorLike = Union[str, QColor]


# ─────────────────────────────── Colour math ─────────────────────────────
def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *v* into [lo, hi]."""
    return lo if v < lo else hi if v > hi else v


def rgba(color: ColorLike, alpha: float) -> QColor:
    """Return *color* with a 0..1 alpha."""
    c = QColor(color)
    c.setAlphaF(clamp(alpha))
    return c


def mix(a: ColorLike, b: ColorLike, t: float) -> QColor:
    """Linear interpolation between two colours."""
    ca, cb = QColor(a), QColor(b)
    t = clamp(t)
    return QColor.fromRgbF(
        ca.redF() + (cb.redF() - ca.redF()) * t,
        ca.greenF() + (cb.greenF() - ca.greenF()) * t,
        ca.blueF() + (cb.blueF() - ca.blueF()) * t,
        ca.alphaF() + (cb.alphaF() - ca.alphaF()) * t,
    )


def ui_font(point_size: float, weight: QFont.Weight = QFont.Weight.Normal,
            letter_spacing: float = 0.0) -> QFont:
    """System UI font (Segoe UI Variable on Windows 11, Segoe UI on Windows 10)."""
    f = QFont()
    f.setFamilies(["Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", "Arial"])
    f.setPointSizeF(point_size)
    f.setWeight(weight)
    if letter_spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, letter_spacing)
    f.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return f


# ─────────────────────── Animation glue (Qt properties) ───────────────────
def animated_float_property(attr: str, lo: Optional[float] = None, hi: Optional[float] = None):
    """
    Create a float ``pyqtProperty`` backed by ``self.<attr>`` that repaints the
    widget whenever it changes. This lets QPropertyAnimation drive QPainter
    widgets directly at the animation timer's ~60 Hz cadence.
    """
    def fget(self) -> float:
        return getattr(self, attr)

    def fset(self, value: float) -> None:
        v = float(value)
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        if getattr(self, attr) != v:
            setattr(self, attr, v)
            self.update()

    return pyqtProperty(float, fget=fget, fset=fset)


# ─────────────────────────────── Geometry ────────────────────────────────
def squircle_path(rect: QRectF, radius: float, smoothing: float = 0.6) -> QPainterPath:
    """
    Continuous-curvature rounded rectangle (iOS-style "squircle" approximation).

    Each corner is one cubic Bézier that starts ``d = r·(1+smoothing)`` from the
    corner along each edge, with control points pulled close to the corner. The
    control distance is solved so the curve's midpoint sits at ~90 % of where
    a circular arc of radius *r* would put it, which keeps the shape visually
    comparable to a regular rounded rect while giving smoother shoulders.
    """
    path = QPainterPath()
    w, h = rect.width(), rect.height()
    if w <= 0 or h <= 0:
        return path
    r = max(0.0, min(radius, w / 2.0, h / 2.0))
    if r < 0.5:
        path.addRect(rect)
        return path
    d = min(r * (1.0 + smoothing), w / 2.0, h / 2.0)
    c = max(0.0, (0.264 * r - 0.125 * d) / 0.375)

    x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
    path.moveTo(x0 + d, y0)
    path.lineTo(x1 - d, y0)
    path.cubicTo(x1 - c, y0, x1, y0 + c, x1, y0 + d)
    path.lineTo(x1, y1 - d)
    path.cubicTo(x1, y1 - c, x1 - c, y1, x1 - d, y1)
    path.lineTo(x0 + d, y1)
    path.cubicTo(x0 + c, y1, x0, y1 - c, x0, y1 - d)
    path.lineTo(x0, y0 + d)
    path.cubicTo(x0, y0 + c, x0 + c, y0, x0 + d, y0)
    path.closeSubpath()
    return path


# ─────────────────────────────── Painting helpers ────────────────────────
def stroke_glow(painter: QPainter, path: QPainterPath, brush: Union[QBrush, ColorLike],
                width: float = 2.0, spread: float = 12.0, layers: int = 5,
                strength: float = 1.0) -> None:
    """Stroke *path* with a soft bloom: several wide, faint passes + one crisp pass."""
    if strength <= 0.001:
        return
    b = brush if isinstance(brush, QBrush) else QBrush(QColor(brush))
    painter.save()
    base = painter.opacity()
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for i in range(layers, 0, -1):
        t = i / layers
        pen = QPen(b, width + spread * t)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setOpacity(base * strength * (0.035 + 0.11 * (1.0 - t)))
        painter.strokePath(path, pen)
    pen = QPen(b, width)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setOpacity(base * strength)
    painter.strokePath(path, pen)
    painter.restore()


_checker_cache: dict[int, QPixmap] = {}


def checkerboard_brush(cell: int = 10) -> QBrush:
    """Subtle dark charcoal checkerboard used behind transparent results."""
    pm = _checker_cache.get(cell)
    if pm is None:
        pm = QPixmap(cell * 2, cell * 2)
        pm.fill(QColor(CHECKER_DARK))
        p = QPainter(pm)
        p.fillRect(0, 0, cell, cell, QColor(CHECKER_LIGHT))
        p.fillRect(cell, cell, cell, cell, QColor(CHECKER_LIGHT))
        p.end()
        _checker_cache[cell] = pm
    return QBrush(pm)


def pill_width(text: str, font: QFont, dot: bool = False) -> float:
    """Width needed by :func:`draw_pill` for *text*."""
    return QFontMetricsF(font).horizontalAdvance(text) + 24.0 + (12.0 if dot else 0.0)


def draw_pill(painter: QPainter, rect: QRectF, text: str, font: QFont,
              fg: ColorLike = TEXT_PRIMARY, bg: Optional[QColor] = None,
              border: Optional[QColor] = None, dot: Optional[QColor] = None) -> None:
    """Frosted capsule label (used for BEFORE/AFTER tags, zoom %, info chips)."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    path = squircle_path(rect, rect.height() / 2.0, smoothing=0.0)
    painter.fillPath(path, QBrush(bg or rgba("#0F1115", 0.80)))
    painter.strokePath(path, QPen(border or rgba("#FFFFFF", 0.08), 1.0))
    x = rect.left() + 12.0
    if dot is not None:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(QPointF(x + 3.0, rect.center().y()), 3.0, 3.0)
        x += 12.0
    painter.setFont(font)
    painter.setPen(QColor(fg))
    painter.drawText(QRectF(x, rect.top(), rect.right() - x - 10.0, rect.height()),
                     int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)
    painter.restore()


def paint_logo(painter: QPainter, rect: QRectF) -> None:
    """App mark: azure→cyan squircle with a white four-point sparkle."""
    painter.save()
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    r = rect.adjusted(0.5, 0.5, -0.5, -0.5)
    path = squircle_path(r, r.width() * 0.28)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0.0, QColor(ACCENT_AZURE))
    g.setColorAt(1.0, QColor(ACCENT_CYAN))
    painter.fillPath(path, QBrush(g))
    sheen = QLinearGradient(r.topLeft(), r.bottomLeft())
    sheen.setColorAt(0.0, rgba("#FFFFFF", 0.32))
    sheen.setColorAt(0.5, rgba("#FFFFFF", 0.0))
    painter.fillPath(path, QBrush(sheen))

    def sparkle(cx: float, cy: float, radius: float) -> QPainterPath:
        k = radius * 0.2
        s = QPainterPath()
        s.moveTo(cx, cy - radius)
        s.quadTo(cx + k, cy - k, cx + radius, cy)
        s.quadTo(cx + k, cy + k, cx, cy + radius)
        s.quadTo(cx - k, cy + k, cx - radius, cy)
        s.quadTo(cx - k, cy - k, cx, cy - radius)
        return s

    c = r.center()
    painter.fillPath(sparkle(c.x() - r.width() * 0.04, c.y() + r.height() * 0.04, r.width() * 0.30),
                     QBrush(QColor("#FFFFFF")))
    painter.fillPath(sparkle(c.x() + r.width() * 0.22, c.y() - r.height() * 0.22, r.width() * 0.11),
                     QBrush(rgba("#FFFFFF", 0.9)))
    painter.restore()


def make_app_icon() -> QIcon:
    """Multi-resolution window/taskbar icon rendered from :func:`paint_logo`."""
    icon = QIcon()
    for s in (16, 20, 24, 32, 40, 48, 64, 128, 256):
        pm = QPixmap(s, s)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        paint_logo(p, QRectF(0, 0, s, s))
        p.end()
        icon.addPixmap(pm)
    return icon


def dark_palette() -> QPalette:
    """Fusion palette so native-ish dialogs also follow the dark theme."""
    pal = QPalette()
    R = QPalette.ColorRole
    pal.setColor(R.Window, QColor(BG_SURFACE))
    pal.setColor(R.WindowText, QColor(TEXT_PRIMARY))
    pal.setColor(R.Base, QColor(BG_DEEP))
    pal.setColor(R.AlternateBase, QColor(BG_ELEVATED))
    pal.setColor(R.Text, QColor(TEXT_PRIMARY))
    pal.setColor(R.Button, QColor(BG_ELEVATED))
    pal.setColor(R.ButtonText, QColor(TEXT_PRIMARY))
    pal.setColor(R.Highlight, QColor(ACCENT_AZURE))
    pal.setColor(R.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(R.ToolTipBase, QColor(BG_ELEVATED))
    pal.setColor(R.ToolTipText, QColor(TEXT_PRIMARY))
    pal.setColor(R.PlaceholderText, QColor(TEXT_TERTIARY))
    pal.setColor(R.Link, QColor(ACCENT_CYAN))
    for role in (R.Text, R.WindowText, R.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(TEXT_TERTIARY))
    return pal


# ─────────────────────────────── Stylesheet ──────────────────────────────
_QSS_TEMPLATE = """
* { outline: none; }
QWidget {
    color: @TEXT_PRIMARY;
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif;
    font-size: 10pt;
}
QLabel { background: transparent; }
QLabel#AppTitle { font-size: 10.5pt; font-weight: 600; color: @TEXT_PRIMARY; }
QLabel#AppSubtitle { color: @TEXT_TERTIARY; font-size: 9pt; }
QLabel#Caption { color: @TEXT_TERTIARY; font-size: 8pt; font-weight: 700; }
QLabel#DeviceBadge {
    color: @ACCENT_CYAN; background: rgba(100, 210, 255, 0.08);
    border: 1px solid rgba(100, 210, 255, 0.24); border-radius: 10px;
    padding: 3px 10px; font-size: 8.5pt; font-weight: 600;
}
QLabel#DeviceBadge[cpu="true"] {
    color: @WARNING; background: rgba(255, 214, 10, 0.08);
    border: 1px solid rgba(255, 214, 10, 0.26);
}
QLabel#Status { color: @TEXT_PRIMARY; font-weight: 600; }
QLabel#StatusDetail { color: @TEXT_TERTIARY; font-size: 9pt; }

QPushButton {
    background: @BG_ELEVATED; border: 1px solid @BORDER; border-radius: 10px;
    padding: 7px 16px; color: @TEXT_PRIMARY; font-weight: 500;
}
QPushButton:hover { background: @BG_HOVER; border-color: @BORDER_STRONG; }
QPushButton:pressed { background: #191B20; }
QPushButton:disabled { color: @TEXT_TERTIARY; background: #14161A; border-color: #1E2025; }
QPushButton[accent="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0A84FF, stop:1 #38B6FF);
    border: 1px solid rgba(100, 210, 255, 0.55); color: #FFFFFF; font-weight: 600; padding: 7px 24px;
}
QPushButton[accent="true"]:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2391FF, stop:1 #5CC8FF);
}
QPushButton[accent="true"]:disabled {
    background: #182433; color: rgba(255, 255, 255, 0.35); border-color: #1F2C3B;
}
QPushButton[danger="true"] {
    background: rgba(255, 69, 58, 0.14); border: 1px solid rgba(255, 69, 58, 0.45);
    color: #FF6961; font-weight: 600; padding: 7px 24px;
}
QPushButton[danger="true"]:hover { background: rgba(255, 69, 58, 0.22); }

QComboBox {
    background: @BG_ELEVATED; border: 1px solid @BORDER; border-radius: 10px;
    padding: 6px 30px 6px 12px; min-height: 20px; color: @TEXT_PRIMARY;
}
QComboBox:hover { border-color: @BORDER_STRONG; }
QComboBox:focus { border-color: rgba(100, 210, 255, 0.55); }
QComboBox:disabled { color: @TEXT_TERTIARY; background: #14161A; border-color: #1E2025; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right; width: 26px; border: none;
}
QComboBox::down-arrow {
    image: none; width: 0px; height: 0px; margin-right: 10px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid @TEXT_SECONDARY;
}
QComboBox QAbstractItemView {
    background: #1A1C21; border: 1px solid @BORDER_STRONG; border-radius: 10px; padding: 4px;
    selection-background-color: rgba(10, 132, 255, 0.35); selection-color: #FFFFFF; outline: none;
}
QComboBox QAbstractItemView::item { min-height: 28px; padding: 0px 10px; border-radius: 6px; }

QProgressBar#Thin {
    background: #1C1E23; border: none; border-radius: 2px; min-height: 4px; max-height: 4px;
}
QProgressBar#Thin::chunk {
    border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0A84FF, stop:1 #64D2FF);
}
QToolTip {
    background-color: #1C1F25; color: @TEXT_PRIMARY; border: 1px solid @BORDER_STRONG;
    border-radius: 6px; padding: 5px 8px;
}
QMessageBox, QDialog { background-color: @BG_SURFACE; }
QMessageBox QLabel { color: @TEXT_PRIMARY; }
QMessageBox QPushButton { min-width: 84px; }
QTextEdit, QPlainTextEdit {
    background: @BG_DEEP; border: 1px solid @BORDER; border-radius: 8px; color: @TEXT_SECONDARY;
    font-family: "Cascadia Mono", "Consolas", monospace; font-size: 9pt;
}
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #3A3E47; border-radius: 4px; min-height: 24px; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: #3A3E47; border-radius: 4px; min-width: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }
QMenu {
    background: #1A1C21; border: 1px solid @BORDER_STRONG; border-radius: 8px; padding: 4px;
}
QMenu::item { padding: 6px 18px; border-radius: 6px; }
QMenu::item:selected { background: rgba(10, 132, 255, 0.35); }
"""


def build_stylesheet() -> str:
    """Resolve @TOKENS in the QSS template (longest names first to avoid prefix clashes)."""
    tokens = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, str) and v.startswith("#")}
    qss = _QSS_TEMPLATE
    for name in sorted(tokens, key=len, reverse=True):
        qss = qss.replace("@" + name, tokens[name])
    return qss