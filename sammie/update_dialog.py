# sammie/update_dialog.py
import os
import sys
import subprocess
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QMessageBox, QWidget, QFrame, QApplication
)
from PySide6.QtGui import QPixmap, QIcon, QCursor
from PySide6.QtCore import Qt, Signal, Slot


class UpdateBannerWidget(QFrame):
    """A subtle, non-intrusive notification banner displayed at the top of the main window."""
    update_clicked = Signal(dict)
    dismissed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_info: Dict[str, Any] = {}
        self.setObjectName("updateBanner")
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QFrame#updateBanner {
                background-color: #1e3a5f;
                border-bottom: 1px solid #2b5282;
            }
            QFrame#updateBanner QLabel {
                color: #ffffff;
                font-size: 13px;
            }
            QFrame#updateBanner QPushButton#viewBtn {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                border-radius: 3px;
                padding: 4px 12px;
                font-weight: bold;
                font-size: 12px;
            }
            QFrame#updateBanner QPushButton#viewBtn:hover {
                background-color: #106ebe;
            }
            QFrame#updateBanner QPushButton#closeBtn {
                background-color: transparent;
                color: #cccccc;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 2px 6px;
            }
            QFrame#updateBanner QPushButton#closeBtn:hover {
                color: #ffffff;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 12, 0)
        layout.setSpacing(12)

        self.icon_label = QLabel("🔔")
        layout.addWidget(self.icon_label)

        self.text_label = QLabel("A new version of Sammie-Roto is available!")
        layout.addWidget(self.text_label)

        layout.addStretch()

        self.action_btn = QPushButton("View Changes & Update")
        self.action_btn.setObjectName("viewBtn")
        self.action_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.action_btn.clicked.connect(self._on_action_clicked)
        layout.addWidget(self.action_btn)

        self.close_btn = QPushButton("✕")
        self.close_btn.setObjectName("closeBtn")
        self.close_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.close_btn.setToolTip("Dismiss for this session")
        self.close_btn.clicked.connect(self.dismiss)
        layout.addWidget(self.close_btn)

        self.hide()

    def show_update(self, update_info: Dict[str, Any]):
        """Displays the banner with updated version information."""
        self.update_info = update_info
        latest = update_info.get("latest_version", "")
        self.text_label.setText(f"A new version of Sammie-Roto (v{latest}) is available!")
        self.show()

    def dismiss(self):
        """Hides the banner."""
        self.hide()
        self.dismissed.emit()

    def _on_action_clicked(self):
        self.update_clicked.emit(self.update_info)


class UpdateDialog(QDialog):
    """Modal dialog displaying release notes for uninstalled updates and update trigger."""

    def __init__(self, parent, update_info: Dict[str, Any]):
        super().__init__(parent)
        self.update_info = update_info
        self.setWindowTitle("Sammie-Roto Update Available")
        self.setMinimumSize(640, 680)
        self.resize(680, 720)
        self.setWindowFlags((self.windowFlags() & ~Qt.WindowContextHelpButtonHint) | Qt.WindowCloseButtonHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Header title
        current_ver = update_info.get("current_version", "unknown")
        latest_ver = update_info.get("latest_version", "unknown")
        header = QLabel(f"Version {latest_ver} is available! (Current: v{current_ver})")
        header.setStyleSheet("font-size: 17px; font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(header)

        # Optional release image banner
        image_bytes = update_info.get("image_bytes")
        if image_bytes:
            pixmap = QPixmap()
            if pixmap.loadFromData(image_bytes):
                img_label = QLabel()
                scaled_pixmap = pixmap.scaled(
                    640, 200,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                img_label.setPixmap(scaled_pixmap)
                img_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(img_label)

        # Markdown release notes browser
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        notes = update_info.get("notes", "No release notes available.")
        self.text_browser.setMarkdown(notes)
        layout.addWidget(self.text_browser, stretch=1)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        self.update_btn = QPushButton("Install Update Now")
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        self.update_btn.clicked.connect(self._confirm_and_update)
        btn_layout.addWidget(self.update_btn)

        self.github_btn = QPushButton("View on GitHub")
        self.github_btn.clicked.connect(self._open_github)
        btn_layout.addWidget(self.github_btn)

        btn_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        layout.addLayout(btn_layout)

        if parent:
            geo = parent.geometry()
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2
            )

    def _open_github(self):
        url = self.update_info.get("url", "https://github.com/Zarxrax/Sammie-Roto-2/releases")
        webbrowser.open_new(url)

    def _confirm_and_update(self):
        reply = QMessageBox.question(
            self,
            "Confirm Update",
            "Sammie-Roto will save your session, close, and launch the updater "
            "to install the latest changes.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        if reply == QMessageBox.Yes:
            self.accept()
            launch_updater_and_exit(parent_window=self.parent())


def launch_updater_and_exit(parent_window=None):
    """Safely saves session state, unlocks instance lock, launches the updater script, and exits."""
    script_dir = Path(__file__).resolve().parent.parent

    # 1. First trigger window close event to ensure points/session settings are saved
    if parent_window:
        try:
            parent_window.close()
        except Exception as e:
            print(f"Warning: Exception during parent window close: {e}")

    # 2. Release single instance lock if held
    app = QApplication.instance()
    if app and hasattr(app, "lock_file"):
        try:
            app.lock_file.unlock()
        except Exception as e:
            print(f"Warning: Exception unlocking lock file: {e}")

    # 3. Launch platform-appropriate update script with --update flag
    try:
        if sys.platform == "win32":
            install_script = script_dir / "install.bat"
            if not install_script.exists():
                QMessageBox.critical(None, "Update Error", f"Install script not found:\n{install_script}")
                return
            # Launch in a visible Command Prompt window with title
            cmd_args = ["cmd.exe", "/c", "start", "Sammie-Roto Updater", str(install_script), "--update"]
            subprocess.Popen(
                cmd_args,
                cwd=str(script_dir),
                shell=True
            )

        elif sys.platform == "darwin":
            install_script = script_dir / "install.sh"
            if not install_script.exists():
                QMessageBox.critical(None, "Update Error", f"Install script not found:\n{install_script}")
                return
            install_script.chmod(install_script.stat().st_mode | 0o755)
            # Launch Terminal.app running the install script with --update
            escaped_script = str(install_script).replace('"', '\\"')
            osa_script = f'tell application "Terminal" to do script "\"{escaped_script}\" --update"'
            subprocess.Popen(["osascript", "-e", osa_script])

        else:
            # Linux terminal dispatch
            install_script = script_dir / "install.sh"
            if not install_script.exists():
                QMessageBox.critical(None, "Update Error", f"Install script not found:\n{install_script}")
                return
            install_script.chmod(install_script.stat().st_mode | 0o755)

            launched = False
            for term in ["x-terminal-emulator", "gnome-terminal", "konsole", "xfce4-terminal", "xterm"]:
                if subprocess.run(["which", term], capture_output=True).returncode == 0:
                    if term == "gnome-terminal":
                        subprocess.Popen([term, "--", str(install_script), "--update"], cwd=str(script_dir))
                    else:
                        subprocess.Popen([term, "-e", f"{install_script} --update"], cwd=str(script_dir))
                    launched = True
                    break

            if not launched:
                cmd_to_run = f"{install_script} --update"
                if app:
                    app.clipboard().setText(cmd_to_run)
                QMessageBox.information(
                    None,
                    "Manual Update Required",
                    f"The update command has been copied to your clipboard:\n\n{cmd_to_run}\n\n"
                    "Please open a terminal and run it to complete the update."
                )

    except Exception as e:
        QMessageBox.critical(None, "Update Error", f"Failed to launch update script:\n{e}")
        return

    # 4. Quit application
    if app:
        app.quit()
    sys.exit(0)
