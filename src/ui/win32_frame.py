"""
Native Windows (Win32 + DWM) frameless window integration for PhotoGalleryMainWindow.

Why this exists:
`Qt.FramelessWindowHint` alone strips `WS_CAPTION`, `WS_THICKFRAME`, and related
frame styles from the HWND. Without those styles, Windows DWM disables:
  1. Windows Aero Snap (`Win+Left/Right/Up/Down` and drag-to-edge snapping),
  2. Native DWM drop shadows and Windows 11 rounded corners, and
  3. Native OS `WM_NCHITTEST` border/corner resizing.

This module restores the required Win32 styles (`WS_THICKFRAME | WS_CAPTION |
WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU`), enables DWM shadow framing,
suppresses the native white title bar via `WM_NCCALCSIZE` (with maximized work-area
compensation), and routes `WM_NCHITTEST` for generous DPI-scaled resize edges and
native `HTCAPTION` title-bar dragging.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import TYPE_CHECKING

from PySide6.QtCore import QByteArray, QPoint
from PySide6.QtWidgets import (
    QAbstractButton,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMenuBar,
    QTabBar,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QMainWindow

# Window styles
GWL_STYLE = -16
GCL_STYLE = -26
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_MINIMIZEBOX = 0x00020000
WS_MAXIMIZEBOX = 0x00010000
WS_SYSMENU = 0x00080000
CS_DBLCLKS = 0x0008

# SetWindowPos flags
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020

# Window messages & hit-test codes
WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084

HTCLIENT = 1
HTCAPTION = 2
HTLEFT = 10
HTRIGHT = 11
HTTOP = 12
HTTOPLEFT = 13
HTTOPRIGHT = 14
HTBOTTOM = 15
HTBOTTOMLEFT = 16
HTBOTTOMRIGHT = 17

# System metrics
SM_CXSIZEFRAME = 32
SM_CYSIZEFRAME = 33
SM_CXPADDEDBORDER = 92

# Monitor & taskbar constants
MONITOR_DEFAULTTONEAREST = 2
ABM_GETSTATE = 0x00000004
ABM_GETTASKBARPOS = 0x00000005
ABS_AUTOHIDE = 0x0000001
ABE_LEFT = 0
ABE_TOP = 1
ABE_RIGHT = 2
ABE_BOTTOM = 3

# DWM attributes
DWMWA_NCRENDERING_POLICY = 2
DWMNCRP_ENABLED = 2
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2


class _MARGINS(ctypes.Structure):
    _fields_ = [
        ("cxLeftWidth", ctypes.c_int),
        ("cxRightWidth", ctypes.c_int),
        ("cyTopHeight", ctypes.c_int),
        ("cyBottomHeight", ctypes.c_int),
    ]


class _NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [
        ("rgrc", wintypes.RECT * 3),
        ("lppos", ctypes.c_void_p),
    ]


class _MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


class _APPBARDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uCallbackMessage", wintypes.UINT),
        ("uEdge", wintypes.UINT),
        ("rc", wintypes.RECT),
        ("lParam", wintypes.LPARAM),
    ]


def is_win32_frameless_supported() -> bool:
    """Return True when running on desktop Windows (not Qt offscreen platform)."""
    return sys.platform == "win32" and os.environ.get("QT_QPA_PLATFORM", "").lower() != "offscreen"


def _unpack_lparam_signed_xy(lparam: int) -> tuple[int, int]:
    """Extract signed 16-bit screen (x, y) coordinates from a Win32 LPARAM."""
    x = ctypes.c_short(lparam & 0xFFFF).value
    y = ctypes.c_short((lparam >> 16) & 0xFFFF).value
    return int(x), int(y)


def compute_resize_border_hit(
    gx: int,
    gy: int,
    rect: tuple[int, int, int, int],
    side_bw: int,
    top_bw: int,
    bottom_bw: int,
    is_maximized: bool,
    corner_bw: int = 0,
) -> int | None:
    """
    Return the Win32 `HT*` resize hit-test code if `(gx, gy)` lies on a resize edge
    or corner of `rect = (left, top, right, bottom)`, or `None` otherwise.
    """
    if is_maximized:
        return None
    left, top, right, bottom = rect
    if not (left <= gx < right and top <= gy < bottom):
        return None

    on_left = gx < left + side_bw
    on_right = gx >= right - side_bw
    on_top = gy < top + top_bw
    on_bottom = gy >= bottom - bottom_bw

    c_side = max(side_bw, corner_bw)
    c_bottom = max(bottom_bw, corner_bw)
    in_left_corner = gx < left + c_side
    in_right_corner = gx >= right - c_side
    in_bottom_corner = gy >= bottom - c_bottom

    # Corners (slightly wider target zone than straight edges for effortless diagonal resizing)
    if (on_top and in_left_corner) or (on_left and gy < top + c_side):
        return HTTOPLEFT
    if (on_top and in_right_corner) or (on_right and gy < top + c_side):
        return HTTOPRIGHT
    if (on_bottom and in_left_corner) or (in_left_corner and in_bottom_corner):
        return HTBOTTOMLEFT
    if (on_bottom and in_right_corner) or (in_right_corner and in_bottom_corner):
        return HTBOTTOMRIGHT

    if on_left:
        return HTLEFT
    if on_right:
        return HTRIGHT
    if on_top:
        return HTTOP
    if on_bottom:
        return HTBOTTOM
    return None


def adjust_maximized_client_rect(
    left: int,
    top: int,
    right: int,
    bottom: int,
    pad_x: int,
    pad_y: int,
    autohide_edge: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Shrink a maximized `WS_THICKFRAME` window rectangle inward so its invisible
    frame overhang does not clip the titlebar/edges off-screen.
    """
    c_left = left + max(0, pad_x)
    c_top = top + max(0, pad_y)
    c_right = right - max(0, pad_x)
    c_bottom = bottom - max(0, pad_y)

    if autohide_edge == ABE_BOTTOM:
        c_bottom -= 1
    elif autohide_edge == ABE_TOP:
        c_top += 1
    elif autohide_edge == ABE_LEFT:
        c_left += 1
    elif autohide_edge == ABE_RIGHT:
        c_right -= 1

    return c_left, c_top, c_right, c_bottom


def is_titlebar_caption_hit(title_bar: QWidget, local_pos: QPoint) -> bool:
    """
    Return True when `local_pos` (in `title_bar` logical coordinates) hits empty
    title-bar surface/labels rather than interactive controls (QAT or window buttons).
    """
    if not title_bar.isVisible():
        return False
    if not (0 <= local_pos.x() < title_bar.width() and 0 <= local_pos.y() < title_bar.height()):
        return False

    interactive_types = (
        QAbstractButton,
        QLineEdit,
        QComboBox,
        QAbstractSpinBox,
        QAbstractSlider,
        QTabBar,
        QMenuBar,
    )
    widget: QWidget | None = title_bar.childAt(local_pos)
    while widget is not None and widget is not title_bar:
        if isinstance(widget, interactive_types):
            return False
        widget = widget.parentWidget()
    return True


def _get_dpi_frame_padding(hwnd: int, rect: wintypes.RECT) -> tuple[int, int, int | None]:
    """Compute `(pad_x, pad_y, autohide_edge)` in physical pixels for the maximized monitor."""
    user32 = ctypes.windll.user32
    shell32 = ctypes.windll.shell32

    dpi = 96
    get_dpi_for_window = getattr(user32, "GetDpiForWindow", None)
    if get_dpi_for_window is not None:
        try:
            detected = int(get_dpi_for_window(hwnd))
            if detected > 0:
                dpi = detected
        except Exception:
            pass

    get_metrics_dpi = getattr(user32, "GetSystemMetricsForDpi", None)
    if get_metrics_dpi is not None:
        try:
            padded = int(get_metrics_dpi(SM_CXPADDEDBORDER, dpi))
            pad_x = int(get_metrics_dpi(SM_CXSIZEFRAME, dpi)) + padded
            pad_y = int(get_metrics_dpi(SM_CYSIZEFRAME, dpi)) + padded
        except Exception:
            pad_x = pad_y = 0
    else:
        padded = int(user32.GetSystemMetrics(SM_CXPADDEDBORDER))
        pad_x = int(user32.GetSystemMetrics(SM_CXSIZEFRAME)) + padded
        pad_y = int(user32.GetSystemMetrics(SM_CYSIZEFRAME)) + padded

    if pad_x <= 0:
        padded = int(user32.GetSystemMetrics(SM_CXPADDEDBORDER))
        pad_x = int(user32.GetSystemMetrics(SM_CXSIZEFRAME)) + padded
    if pad_y <= 0:
        padded = int(user32.GetSystemMetrics(SM_CXPADDEDBORDER))
        pad_y = int(user32.GetSystemMetrics(SM_CYSIZEFRAME)) + padded

    autohide_edge: int | None = None
    try:
        abd = _APPBARDATA()
        abd.cbSize = ctypes.sizeof(_APPBARDATA)
        state = shell32.SHAppBarMessage(ABM_GETSTATE, ctypes.byref(abd))
        if state & ABS_AUTOHIDE:
            hmon = user32.MonitorFromRect(ctypes.byref(rect), MONITOR_DEFAULTTONEAREST)
            mi = _MONITORINFO()
            mi.cbSize = ctypes.sizeof(_MONITORINFO)
            if hmon and user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                w, m = mi.rcWork, mi.rcMonitor
                if (w.left, w.top, w.right, w.bottom) == (m.left, m.top, m.right, m.bottom):
                    pos_abd = _APPBARDATA()
                    pos_abd.cbSize = ctypes.sizeof(_APPBARDATA)
                    if shell32.SHAppBarMessage(ABM_GETTASKBARPOS, ctypes.byref(pos_abd)):
                        autohide_edge = int(pos_abd.uEdge)
    except Exception:
        autohide_edge = None

    return pad_x, pad_y, autohide_edge


def apply_win32_frameless_styles(hwnd: int) -> bool:
    """
    Enable native Aero Snap (`WS_THICKFRAME | WS_CAPTION | WS_MAXIMIZEBOX | ...`) and
    DWM drop shadow framing on `hwnd` without showing the native OS title bar.
    Returns True if applied successfully.
    """
    if not is_win32_frameless_supported() or not hwnd:
        return False
    try:
        user32 = ctypes.windll.user32
        dwmapi = ctypes.windll.dwmapi

        desired_bits = WS_THICKFRAME | WS_CAPTION | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SYSMENU
        cur_style = int(user32.GetWindowLongPtrW(hwnd, GWL_STYLE))
        style_changed = (cur_style & desired_bits) != desired_bits
        if style_changed:
            user32.SetWindowLongPtrW(hwnd, GWL_STYLE, cur_style | desired_bits)

        cur_class_style = int(user32.GetClassLongPtrW(hwnd, GCL_STYLE))
        if not (cur_class_style & CS_DBLCLKS):
            user32.SetClassLongPtrW(hwnd, GCL_STYLE, cur_class_style | CS_DBLCLKS)

        if style_changed:
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )

        policy = ctypes.c_int(DWMNCRP_ENABLED)
        dwmapi.DwmSetWindowAttribute(hwnd, DWMWA_NCRENDERING_POLICY, ctypes.byref(policy), ctypes.sizeof(policy))

        margins = _MARGINS(1, 1, 1, 1)
        dwmapi.DwmExtendFrameIntoClientArea(hwnd, ctypes.byref(margins))

        # Windows 11 native rounded corners (ignored cleanly on Windows 10)
        corner_pref = ctypes.c_int(DWMWCP_ROUND)
        dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(corner_pref), ctypes.sizeof(corner_pref)
        )
        return True
    except Exception:
        return False


def handle_win32_native_event(
    window: "QMainWindow",
    event_type: QByteArray | bytes,
    message: int,
) -> tuple[bool, int] | None:
    """
    Process `WM_NCCALCSIZE` and `WM_NCHITTEST` for a frameless `QMainWindow` on Windows.
    Returns `(handled, lresult)` or `None` to fall back to Qt's default handling.
    """
    if not is_win32_frameless_supported() or message is None:
        return None
    if bytes(event_type) != b"windows_generic_MSG":
        return None

    try:
        msg_addr = int(message)
        if not msg_addr:
            return None
        msg = wintypes.MSG.from_address(msg_addr)
        user32 = ctypes.windll.user32

        if msg.message == WM_NCCALCSIZE:
            if not msg.wParam:
                return True, 0
            if user32.IsZoomed(msg.hWnd):
                params = _NCCALCSIZE_PARAMS.from_address(msg.lParam)
                r0 = params.rgrc[0]
                pad_x, pad_y, autohide_edge = _get_dpi_frame_padding(msg.hWnd, r0)
                r0.left, r0.top, r0.right, r0.bottom = adjust_maximized_client_rect(
                    r0.left, r0.top, r0.right, r0.bottom, pad_x, pad_y, autohide_edge
                )
            return True, 0

        if msg.message == WM_NCHITTEST:
            gx, gy = _unpack_lparam_signed_xy(int(msg.lParam))
            rect = wintypes.RECT()
            if not user32.GetWindowRect(msg.hWnd, ctypes.byref(rect)):
                return None

            dpr = max(1.0, float(window.devicePixelRatioF()))
            side_bw = max(8, int(round(8 * dpr)))
            bottom_bw = max(8, int(round(8 * dpr)))
            # Keep top resize border tight so near-top-right Close/Max buttons remain easy to click
            top_bw = max(5, int(round(5 * dpr)))
            corner_bw = max(12, int(round(12 * dpr)))
            is_max = bool(user32.IsZoomed(msg.hWnd))

            edge_hit = compute_resize_border_hit(
                gx,
                gy,
                (rect.left, rect.top, rect.right, rect.bottom),
                side_bw=side_bw,
                top_bw=top_bw,
                bottom_bw=bottom_bw,
                is_maximized=is_max,
                corner_bw=corner_bw,
            )
            if edge_hit is not None:
                return True, edge_hit

            ribbon = getattr(window, "ribbon", None)
            top_header = getattr(ribbon, "top_header", None) if ribbon is not None else None
            if isinstance(top_header, QWidget) and top_header.isVisible():
                pt = wintypes.POINT(gx, gy)
                if user32.ScreenToClient(msg.hWnd, ctypes.byref(pt)):
                    local_win_pt = QPoint(int(round(pt.x / dpr)), int(round(pt.y / dpr)))
                    local_tb_pt = top_header.mapFrom(window, local_win_pt)
                    if is_titlebar_caption_hit(top_header, local_tb_pt):
                        return True, HTCAPTION

            return True, HTCLIENT
    except Exception:
        return None

    return None
