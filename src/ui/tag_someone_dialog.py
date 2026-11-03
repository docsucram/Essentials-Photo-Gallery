"""
Tag Someone Dialog for PhotoGallery64.
Modeled on Windows Live Photo Gallery ('SPEC.md' Section 5, media_1792308377814_eb3d7e6d.png).

Provides an authentic WLPG popup dialog when tagging faces:
- Search box with 'Type a Name' placeholder and live filtering
- Scrollable list of existing people with 28x28 avatars
- Instant click or Enter selection
- Allows typing a new person's name
- Esc or [X] button to dismiss
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from PySide6.QtCore import QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QKeyEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.ribbon_toolbar import make_person_avatar_pixmap

logger = logging.getLogger(__name__)


class TagSomeoneDialog(QDialog):
    """
    Authentic WLPG 'Tag someone' floating popup dialog.
    """

    person_selected = Signal(str, int)  # (person_name, person_id)

    def __init__(
        self,
        people: Sequence[dict[str, Any]],
        parent: QWidget | None = None,
        title: str = "Tag someone",
    ) -> None:
        super().__init__(parent, Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFixedSize(270, 340)

        self._all_people = list(people)
        self._selected_name: str | None = None
        self._selected_id: int | None = None

        self._build_ui(title)
        self._populate_list(self._all_people)
        self.txt_search.setFocus(Qt.ActiveWindowFocusReason)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.txt_search.setFocus(Qt.ActiveWindowFocusReason)
        self.txt_search.selectAll()
        QTimer.singleShot(0, lambda: (self.txt_search.setFocus(Qt.ActiveWindowFocusReason), self.txt_search.selectAll()))

    def _build_ui(self, title: str) -> None:
        self.setObjectName("TagSomeoneDialog")
        self.setStyleSheet(
            """
            QDialog#TagSomeoneDialog {
                background-color: #FFFFFF;
                border: 1px solid #70C0E7;
                border-radius: 4px;
            }
            """
        )

        main_vbox = QVBoxLayout(self)
        main_vbox.setContentsMargins(6, 6, 6, 6)
        main_vbox.setSpacing(6)

        # 1. Header (Title + [X] button)
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 2, 2, 2)
        header_layout.setSpacing(4)

        lbl_title = QLabel(title, self)
        lbl_title.setStyleSheet("font-family: 'Segoe UI'; font-size: 9.5pt; font-weight: bold; color: #1E395B;")
        header_layout.addWidget(lbl_title)
        header_layout.addStretch(1)

        btn_close = QPushButton("✕", self)
        btn_close.setFixedSize(18, 18)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet(
            """
            QPushButton {
                border: none;
                background: transparent;
                color: #555555;
                font-size: 8pt;
                font-family: 'Segoe UI';
                border-radius: 2px;
            }
            QPushButton:hover {
                background: #E81123;
                color: #FFFFFF;
            }
            """
        )
        btn_close.clicked.connect(self.reject)
        header_layout.addWidget(btn_close)
        main_vbox.addLayout(header_layout)

        # 2. Search Input Box ('Type a Name')
        self.txt_search = QLineEdit(self)
        self.txt_search.setObjectName("TagSomeoneSearchInput")
        self.txt_search.setPlaceholderText("Type a Name")
        self.txt_search.setStyleSheet(
            """
            QLineEdit#TagSomeoneSearchInput {
                border: 1px solid #3399FF;
                border-radius: 2px;
                padding: 4px 6px;
                font-family: 'Segoe UI';
                font-size: 9pt;
                color: #000000;
                background-color: #FFFFFF;
            }
            QLineEdit#TagSomeoneSearchInput:focus {
                border: 1px solid #0066CC;
            }
            """
        )
        self.txt_search.textChanged.connect(self._on_search_text_changed)
        self.txt_search.returnPressed.connect(self._on_return_pressed)
        main_vbox.addWidget(self.txt_search)

        # 3. People List Widget
        self.list_widget = QListWidget(self)
        self.list_widget.setObjectName("TagSomeoneListWidget")
        self.list_widget.setIconSize(QSize(28, 28))
        self.list_widget.setStyleSheet(
            """
            QListWidget#TagSomeoneListWidget {
                border: 1px solid #D0DBE5;
                border-radius: 2px;
                background-color: #FFFFFF;
                font-family: 'Segoe UI';
                font-size: 9pt;
                outline: none;
            }
            QListWidget#TagSomeoneListWidget::item {
                padding: 3px 6px;
                color: #1E395B;
                border-bottom: 1px solid #F0F4F8;
            }
            QListWidget#TagSomeoneListWidget::item:hover {
                background-color: #E5F3FF;
                border: 1px solid #CCE8FF;
                border-radius: 2px;
            }
            QListWidget#TagSomeoneListWidget::item:selected {
                background-color: #CCE8FF;
                color: #000000;
                border: 1px solid #99D1FF;
                border-radius: 2px;
            }
            """
        )
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        main_vbox.addWidget(self.list_widget, 1)

    def _populate_list(self, people: Sequence[dict[str, Any]]) -> None:
        self.list_widget.clear()
        for p in people:
            pname = str(p.get("name", "")).strip()
            if not pname:
                continue
            pid = int(p.get("id", 0))
            blob = p.get("thumbnail_blob")
            pix = make_person_avatar_pixmap(pname, blob, size=28, circular=False)

            item = QListWidgetItem(QIcon(pix), pname)
            item.setData(Qt.UserRole, pid)
            item.setData(Qt.UserRole + 1, pname)
            item.setSizeHint(QSize(220, 34))
            self.list_widget.addItem(item)

        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _on_search_text_changed(self, query: str) -> None:
        q = query.strip().lower()
        if not q:
            self._populate_list(self._all_people)
            return

        filtered = [
            p for p in self._all_people
            if q in str(p.get("name", "")).lower()
        ]
        self._populate_list(filtered)

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.UserRole)
        pname = item.data(Qt.UserRole + 1)
        self._selected_id = pid
        self._selected_name = pname
        self.person_selected.emit(pname, pid)
        self.accept()

    def _on_return_pressed(self) -> None:
        # If an item is selected in the list, use that
        current_item = self.list_widget.currentItem()
        entered_text = self.txt_search.text().strip()

        if current_item and (
            not entered_text
            or current_item.text().lower() == entered_text.lower()
            or current_item.text().lower().startswith(entered_text.lower())
        ):
            self._on_item_clicked(current_item)
            return

        if entered_text:
            # Check if entered text exactly matches any person
            for p in self._all_people:
                if str(p.get("name", "")).strip().lower() == entered_text.lower():
                    pid = int(p.get("id", 0))
                    pname = str(p.get("name", "")).strip()
                    self._selected_id = pid
                    self._selected_name = pname
                    self.person_selected.emit(pname, pid)
                    self.accept()
                    return

            # Brand new name typed
            self._selected_id = -1
            self._selected_name = entered_text
            self.person_selected.emit(entered_text, -1)
            self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Down:
            curr = self.list_widget.currentRow()
            if curr < self.list_widget.count() - 1:
                self.list_widget.setCurrentRow(curr + 1)
            event.accept()
            return
        elif event.key() == Qt.Key_Up:
            curr = self.list_widget.currentRow()
            if curr > 0:
                self.list_widget.setCurrentRow(curr - 1)
            event.accept()
            return
        elif event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    @classmethod
    def prompt(
        cls,
        people: Sequence[dict[str, Any]],
        parent: QWidget | None = None,
        pos: QPoint | None = None,
        title: str = "Tag someone",
    ) -> tuple[str | None, int | None]:
        """
        Displays the Tag someone dialog and returns (person_name, person_id) or (None, None).
        If person_id is -1, it represents a brand-new person name to be created.
        """
        dlg = cls(people, parent, title=title)
        if pos is not None:
            # Clamp to screen bounds
            scr = QApplication.primaryScreen().geometry()
            x = min(max(0, pos.x()), scr.width() - dlg.width() - 10)
            y = min(max(0, pos.y()), scr.height() - dlg.height() - 10)
            dlg.move(x, y)
        elif parent is not None:
            p_geom = parent.frameGeometry()
            dlg.move(p_geom.center() - dlg.rect().center())
        if dlg.exec() == QDialog.Accepted:
            return dlg._selected_name, dlg._selected_id
        return None, None
