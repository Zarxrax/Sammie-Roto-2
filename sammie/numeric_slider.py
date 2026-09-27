"""Editable value field for integer sliders and scaled decimal sliders."""

from decimal import Decimal, InvalidOperation

from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator, QIntValidator
from PySide6.QtWidgets import QLineEdit


class NumericSliderValue(QLineEdit):
    def __init__(self, slider, decimals=0, parent=None):
        super().__init__(parent)
        self.slider = slider
        self.decimals = decimals
        self.scale = 10 ** decimals
        self.setAlignment(Qt.AlignCenter)
        self.setFixedWidth(65 if decimals else 48)
        if decimals:
            validator = QDoubleValidator(slider.minimum() / self.scale,
                                         slider.maximum() / self.scale, decimals, self)
            validator.setNotation(QDoubleValidator.StandardNotation)
        else:
            validator = QIntValidator(slider.minimum(), slider.maximum(), self)
        self.setValidator(validator)
        slider.valueChanged.connect(self._show_value)
        self.editingFinished.connect(self._commit)
        self._show_value(slider.value())

    def _format(self, value):
        return f"{value / self.scale:.{self.decimals}f}" if self.decimals else str(value)

    def _show_value(self, value):
        if self.hasFocus() and self.isModified():
            return
        cursor = self.cursorPosition()
        self.setText(self._format(value))
        self.setCursorPosition(min(cursor, len(self.text())))

    def _commit(self):
        try:
            value = int(Decimal(self.text()) * self.scale)
        except (InvalidOperation, ValueError):
            value = self.slider.value()
        self.slider.setValue(value)
        self.setModified(False)
        self._show_value(self.slider.value())

    def keyPressEvent(self, event):
        if event.key() not in (Qt.Key_Up, Qt.Key_Down):
            return super().keyPressEvent(event)
        text = self.text()
        cursor = self.cursorPosition()
        digit = cursor if cursor < len(text) and text[cursor].isdigit() else cursor - 1
        if digit < 0 or not text[digit].isdigit():
            return
        decimal = text.find(".")
        integer_end = decimal if decimal >= 0 else len(text)
        place = integer_end - digit - 1 if digit < integer_end else integer_end - digit
        increment = 10 ** (place + self.decimals)
        direction = 1 if event.key() == Qt.Key_Up else -1
        self._commit()
        self.slider.setValue(self.slider.value() + direction * increment)
        self.setModified(False)
        self._show_value(self.slider.value())
        self.setCursorPosition(min(cursor, len(self.text())))
