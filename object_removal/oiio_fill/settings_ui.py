"""Controls for OpenImageIO's push-pull fill engine."""
from PySide6.QtWidgets import QLabel


def create_settings(tab):
    hint = QLabel("Fills removed regions from surrounding pixels using OpenImageIO.\nBest for small or simple areas.")
    hint.setWordWrap(True)
    return hint


def load_settings(tab):
    pass


def connect_preview(tab, callback):
    pass
