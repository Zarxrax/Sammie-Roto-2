"""
model_downloader.py

A reusable dialog for downloading model files with:
- Progress bar per file + overall progress
- MD5 checksum verification
- .part extension during download, renamed on success
- On-demand or batch usage via shared and engine-owned model registries

Engine folders can declare downloads in their own ``downloads.py`` module.

Usage (on-demand, single model by key):
    from model_downloader import ensure_models

    if not ensure_models("matanyone", parent=self):
        return  # user cancelled or download failed

Usage (multiple models by key):
    if not ensure_models(["sam2_large", "sam2_base_plus"], parent=self):
        return

Usage (all models at once, e.g. on first install):
    if not ensure_models("all", parent=self, title="Downloading Models"):
        return

Usage (ad-hoc spec, bypassing the registry):
    from model_downloader import ensure_models, DownloadSpec

    if not ensure_models(DownloadSpec(url=..., md5=..., dest_dir=...), parent=self):
        return
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from importlib.util import find_spec
import os
from pkgutil import iter_modules
import sys
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from PySide6.QtCore import (
    QObject, QThread, Signal, Slot
)
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QProgressBar, QVBoxLayout, QWidget
)


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class DownloadSpec:
    """Describes a single file to download."""
    url: str
    md5: str
    dest_dir: str

    @property
    def filename(self) -> str:
        return self.url.split("/")[-1]

    @property
    def final_path(self) -> Path:
        return Path(self.dest_dir) / self.filename

    @property
    def part_path(self) -> Path:
        return Path(self.dest_dir) / (self.filename + ".part")

    def already_downloaded(self) -> bool:
        return self.final_path.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Worker (runs in a background QThread)
# ---------------------------------------------------------------------------

class _DownloadWorker(QObject):
    # file-level signals
    file_started = Signal(int, str)                 # (index, filename)
    file_progress = Signal(int, float, float, int)  # (index, mb_done, mb_total, percent)
    file_done = Signal(int)                         # (index,)
    file_skipped = Signal(int, str)                 # (index, filename)
    file_error = Signal(int, str)                   # (index, error_message)

    # overall
    overall_progress = Signal(int, int)      # (files_done, files_total)
    all_done = Signal()
    cancelled = Signal()

    def __init__(self, specs: List[DownloadSpec], parent: QObject | None = None):
        super().__init__(parent)
        self._specs = specs
        self._abort = False
        self._current_response = None  # set during active download
    
        self._session = requests.Session()
        self._session.headers.update({
            "Accept-Encoding": "identity"
        })

    @Slot()
    def run(self) -> None:
        total = len(self._specs)
        done = 0

        for idx, spec in enumerate(self._specs):
            if self._abort:
                self.cancelled.emit()
                return

            # Already on disk with correct checksum?
            if spec.already_downloaded():
                self.file_skipped.emit(idx, spec.filename)
                done += 1
                self.overall_progress.emit(done, total)
                continue

            # Ensure destination directory exists
            Path(spec.dest_dir).mkdir(parents=True, exist_ok=True)

            self.file_started.emit(idx, spec.filename)

            try:
                self._download_one(idx, spec)
            except Exception as exc:
                # Clean up partial file
                if spec.part_path.exists():
                    spec.part_path.unlink(missing_ok=True)
                if self._abort:
                    self.cancelled.emit()
                else:
                    self.file_error.emit(idx, str(exc))
                return  # stop processing further files on error

            done += 1
            self.overall_progress.emit(done, total)
            self.file_done.emit(idx)

        self.all_done.emit()

    def _download_one(self, idx: int, spec: DownloadSpec) -> None:
        self._current_response = self._session.get(spec.url, stream=True, timeout=30)
        response = self._current_response
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        last_emit = 0
        emit_interval = 1 * 1024 * 1024  # emit every 1 MB

        with open(spec.part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if self._abort:
                    raise RuntimeError("Download cancelled by user.")
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if downloaded - last_emit >= emit_interval or downloaded == total_size:
                        mb_done = downloaded / 1_048_576
                        mb_total = total_size / 1_048_576 if total_size else 0
                        pct = int(downloaded / total_size * 100) if total_size else 0

                        self.file_progress.emit(idx, mb_done, mb_total, pct)
                        last_emit = downloaded

        # download finished — no active network socket anymore
        self._current_response = None

        # Verify checksum
        actual_md5 = _md5(spec.part_path)
        if actual_md5 != spec.md5:
            spec.part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {spec.filename}.\n"
                f"  Expected : {spec.md5}\n"
                f"  Got      : {actual_md5}"
            )

        # Atomic rename
        spec.part_path.rename(spec.final_path)

    def abort(self) -> None:
        self._abort = True
        # Close the socket so iter_content() unblocks immediately
        if self._current_response is not None:
            try:
                self._current_response.raw.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ModelDownloadDialog(QDialog):
    """
    Modal dialog that downloads one or more model files in the background.

    Parameters
    ----------
    specs : list of DownloadSpec
        Files to download (already-present files are silently skipped).
    parent : QWidget, optional
    title : str
        Window title.
    """

    def __init__(
        self,
        specs: List[DownloadSpec],
        parent: QWidget | None = None,
        title: str = "Downloading Models",
    ):
        super().__init__(parent)
        self._specs = specs
        self._worker: _DownloadWorker | None = None
        self._thread: QThread | None = None
        self._success = False

        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setModal(True)

        self._build_ui()
        self._start_downloads()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Status label (current file name)
        self._status_label = QLabel("Preparing…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Per-file progress bar
        self._file_bar = QProgressBar()
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(0)
        self._file_bar.setTextVisible(True)
        layout.addWidget(self._file_bar)

        # Overall label + bar (only shown for multi-file downloads)
        self._overall_label = QLabel("Overall progress:")
        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, len(self._specs))
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("%v / %m files")

        if len(self._specs) > 1:
            layout.addWidget(self._overall_label)
            layout.addWidget(self._overall_bar)
        else:
            self._overall_label.hide()
            self._overall_bar.hide()

        # Error label (hidden until needed)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: red;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        # Button box: Cancel only while downloading; Close after
        self._button_box = QDialogButtonBox()
        self._cancel_btn = self._button_box.addButton(QDialogButtonBox.Cancel)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------
    # Download orchestration
    # ------------------------------------------------------------------

    def _start_downloads(self) -> None:
        self._worker = _DownloadWorker(self._specs)
        self._thread = QThread(self)

        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)

        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_skipped.connect(self._on_file_skipped)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_error.connect(self._on_file_error)
        self._worker.overall_progress.connect(self._on_overall_progress)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(int, str)
    def _on_file_started(self, idx: int, filename: str) -> None:
        self._status_label.setText(f"Downloading  {filename}…")
        self._file_bar.setValue(0)
        self._file_bar.setFormat("%p%")

    @Slot(int, float, float, int)
    def _on_file_progress(self, idx: int, mb_done: float, mb_total: float, pct: int) -> None:
        if mb_total > 0:
            self._file_bar.setValue(pct)
            self._file_bar.setFormat(f"{pct}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
        else:
            self._file_bar.setRange(0, 0)

    @Slot(int, str)
    def _on_file_skipped(self, idx: int, filename: str) -> None:
        self._status_label.setText(f"✓  {filename}  (already downloaded)")
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(100)
        self._file_bar.setFormat("Already downloaded")

    @Slot(int)
    def _on_file_done(self, idx: int) -> None:
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(100)
        self._file_bar.setFormat("Verified ✓")

    @Slot(int, str)
    def _on_file_error(self, idx: int, message: str) -> None:
        self._thread.quit()
        spec = self._specs[idx]
        self._status_label.setText(f"Failed to download  {spec.filename}")
        self._error_label.setText(f"Error: {message}")
        self._error_label.show()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    @Slot(int, int)
    def _on_overall_progress(self, done: int, total: int) -> None:
        self._overall_bar.setValue(done)

    @Slot()
    def _on_all_done(self) -> None:
        self._thread.quit()
        self._success = True
        self.accept() # auto-close
        #self._status_label.setText("All models downloaded successfully.")
        #self._cancel_btn.setText("Close")
        #self._cancel_btn.clicked.disconnect()
        #self._cancel_btn.clicked.connect(self.accept)

    @Slot()
    def _on_cancelled(self) -> None:
        self._thread.quit()
        self.reject()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.abort()
        # Dialog will close when the worker emits cancelled or we force-close
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling…")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.abort()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)

    @property
    def succeeded(self) -> bool:
        """True if all downloads completed successfully."""
        return self._success


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def ensure_models(
    models: "str | DownloadSpec | List[str | DownloadSpec]",
    parent: "QWidget | None" = None,
    title: str = "Downloading Models",
) -> bool:
    """
    Ensure one or more models are present, downloading any that are missing.

    ``models`` can be:
    - A registry key string:           "matanyone"
    - The special string "all":        downloads every entry in MODEL_REGISTRY
    - A list of registry key strings:  ["sam2_large", "matanyone"]
    - A DownloadSpec instance:         for ad-hoc specs not in the registry
    - A list mixing keys and specs

    Returns True if all models are present (or were successfully downloaded),
    False if the user cancelled or a download failed.
    """
    registry = get_model_registry()
    # Normalise to a flat list of DownloadSpec
    if isinstance(models, str) and models == "all":
        specs = list(registry.values())
    else:
        if not isinstance(models, list):
            models = [models]
        specs = []
        for item in models:
            if isinstance(item, str):
                if item not in registry:
                    raise KeyError(
                        f"Unknown model key {item!r}. "
                        f"Available keys: {list(registry)}"
                    )
                specs.append(registry[item])
            elif isinstance(item, DownloadSpec):
                specs.append(item)
            else:
                raise TypeError(f"Expected str or DownloadSpec, got {type(item)}")

    needed = [s for s in specs if not s.already_downloaded()]
    if not needed:
        return True

    dlg = ModelDownloadDialog(needed, parent=parent, title=title)
    accepted = dlg.exec() == QDialog.Accepted
    return accepted and dlg.succeeded


# ---------------------------------------------------------------------------
# Shared model registry. Matting engines declare their own downloads.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: "dict[str, DownloadSpec]" = {}


def get_model_registry() -> dict[str, DownloadSpec]:
    """Include checkpoint declarations from discovered engine folders."""
    registry = dict(MODEL_REGISTRY)
    for category in ("matting", "segmentation", "object_removal"):
        package = import_module(category)
        for module in iter_modules(package.__path__):
            if not module.ispkg:
                continue
            module_name = f"{category}.{module.name}.downloads"
            if find_spec(module_name) is None:
                continue
            for key, spec in import_module(module_name).DOWNLOADS.items():
                if key in registry:
                    raise ValueError(f"Duplicate model download key: {key}")
                registry[key] = spec
    return registry

# ---------------------------------------------------------------------------
# CLI entrypoint  —  python model_downloader.py [KEY …]
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    script_dir = str(Path(__file__).resolve().parent)
    project_dir = str(Path(__file__).resolve().parent.parent)
    sys.path[:] = [project_dir] + [path for path in sys.path if path not in (script_dir, project_dir)]
    registry = get_model_registry()
    keys = sys.argv[1:]
    if keys:
        unknown = [k for k in keys if k not in registry]
        if unknown:
            print(f"Unknown model key(s): {unknown}", file=sys.stderr)
            print(f"Available: {list(registry)}", file=sys.stderr)
            sys.exit(1)
        specs = [registry[k] for k in keys]
    else:
        specs = list(registry.values())  # all models

    for spec in specs:
        if spec.already_downloaded():
            print(f"{spec.filename} already downloaded.")
            continue
 
        os.makedirs(spec.dest_dir, exist_ok=True)
        print(f"Downloading {spec.filename} to {spec.dest_dir}...")
 
        r = requests.get(spec.url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(spec.part_path, "wb") as f:
            for data in r.iter_content(1024 * 1024):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            spec.part_path.unlink(missing_ok=True)
            sys.exit(f"Error while downloading {spec.filename}")
 
        actual_md5 = _md5(spec.part_path)
        if actual_md5 != spec.md5:
            spec.part_path.unlink(missing_ok=True)
            sys.exit(
                f"Checksum mismatch for {spec.filename}.\n"
                f"  Expected : {spec.md5}\n"
                f"  Got      : {actual_md5}"
            )
 
        spec.part_path.rename(spec.final_path)
