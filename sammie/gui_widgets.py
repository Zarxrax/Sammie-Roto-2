"""
GUI utility widgets for Sammie-Roto application.

This module contains reusable UI components including:
- ConsoleRedirect: Redirects stdout/stderr to GUI console
- ColorDisplayWidget: Displays object colors
- UpdateChecker: Checks for application updates
- ClickableLabel: QLabel with double-click support
- HotkeysHelpDialog: Displays keyboard shortcuts
- PointTable: Table widget for displaying segmentation points
- ImageViewer: Custom graphics view for image display with zoom/pan
- FrameSlider: Custom QSlider with visual in/out point indicators and range highlighting
"""

import os
import shutil
import threading
import requests
from packaging import version
from datetime import datetime
from PySide6.QtWidgets import (
    QLabel, QTableWidget, QTableWidgetItem, QAbstractItemView, 
    QHeaderView, QPushButton, QWidget, QHBoxLayout, QVBoxLayout,
    QDialog, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QColorDialog, QSlider, QStyleOptionSlider, QStyle
)
from PySide6.QtGui import (
    QPixmap, QMouseEvent, QWheelEvent, QPainter, QColor, QIcon,
    QPen, QPolygon, QPalette
)
from PySide6.QtCore import (
    Qt, QPointF, QObject, Signal, QRect, QPoint
)

from sammie import sammie
from sammie.settings_manager import get_settings_manager

# ==================== CONSOLE REDIRECT ====================

class ConsoleRedirect(QObject):
    """Redirects stdout/stderr to GUI console with support for progress updates"""
    text_written = Signal(str, bool)  # text, is_carriage_return

    def __init__(self):
        super().__init__()
        # Create log file
        self.log_file = open("sammie_debug.log", "w", encoding="utf-8")
        self.log_file.write(f"\n=== Started: {datetime.now()} ===\n")
        self.log_file.flush()
        
    def write(self, text):
        if not text:
            return
        # Write to log file
        self.log_file.write(text)
        self.log_file.flush()
        
        # Detect carriage return without newline (tqdm-like progress)
        if "\r" in text and not text.endswith("\n"):
            clean = text.split("\r")[-1]  # take the last part after \r
            self.text_written.emit(clean, True)
        else:
            self.text_written.emit(text, False)

    def flush(self):
        pass
        
    def close(self):
        self.log_file.write(f"=== Ended: {datetime.now()} ===\n")
        self.log_file.close()


# ==================== COLOR DISPLAY ====================

class ColorDisplayWidget(QLabel):
    """Widget to display a colored rectangle representing object colors"""
    
    def __init__(self, color_rgb, size=(20, 16)):
        super().__init__()
        self.color_rgb = color_rgb
        self.size = size
        self.setFixedSize(*size)
        self.update_color()
    
    def update_color(self):
        """Renders the color rectangle with border"""
        pixmap = QPixmap(*self.size)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        
        # Fill with the specified color
        if self.color_rgb is None:
            r, g, b = (0, 255, 0)  # fallback default
        else:
            r, g, b = self.color_rgb
        color = QColor(r, g, b)
        painter.fillRect(1, 1, self.size[0]-2, self.size[1]-2, color)
        
        # Draw border
        painter.setPen(QColor(100, 100, 100))
        painter.drawRect(0, 0, self.size[0]-1, self.size[1]-1)
        painter.end()
        
        self.setPixmap(pixmap)
    
    def set_color(self, color_rgb):
        """Update the displayed color"""
        self.color_rgb = color_rgb
        self.update_color()

class ColorPickerWidget(ColorDisplayWidget):
    """Clickable color display widget that opens a color picker dialog    
    Extends ColorDisplayWidget to add click functionality for choosing colors.
    Used for selecting background colors.
    """

    color_changed = Signal(tuple)  # Emits (r, g, b) tuple
    
    def __init__(self, color_rgb=(0, 255, 0), size=(20, 16)):
        super().__init__(color_rgb, size)
        self.setToolTip("Click to choose background color")
        # Make it look clickable
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        """Handle mouse click to open color picker"""
        if event.button() == Qt.LeftButton:
            self._open_color_dialog()
        super().mousePressEvent(event)
    
    def _open_color_dialog(self):
        """Open color picker dialog"""
        
        current_color = QColor(*self.color_rgb)
        color = QColorDialog.getColor(
            current_color,
            self,
            "Choose Greenscreen Color",
            QColorDialog.DontUseNativeDialog  # Use Qt dialog for consistency
        )
        
        if color.isValid():
            new_color = (color.red(), color.green(), color.blue())
            self.set_color(new_color)
            self.color_changed.emit(new_color)

# ==================== UPDATE CHECKER ====================

class UpdateChecker(QObject):
    """Checks for application updates on GitHub in background thread"""
    update_available = Signal(str, str)  # current_version, latest_version
    
    def __init__(self):
        super().__init__()
    
    def check_for_updates(self, repo="Zarxrax/Sammie-Roto", timeout=5):
        """Check for updates in a background thread"""
        def background_check():
            try:
                url = f"https://api.github.com/repos/{repo}/releases/latest"
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    latest = response.json().get("tag_name", "").lstrip("v")
                    from sammie_main import __version__
                    if version.parse(latest) > version.parse(__version__):
                        # Emit signal to main thread
                        self.update_available.emit(__version__, latest)
            except Exception as e:
                # Optionally log the error
                print(f"Update check failed silently: {e}")
        
        threading.Thread(target=background_check, daemon=True).start()


# ==================== CLICKABLE LABEL ====================

class ClickableLabel(QLabel):
    """A QLabel that emits a signal when double-clicked"""
    doubleClicked = Signal()
    
    def __init__(self, text=""):
        super().__init__(text)
        self.setStyleSheet("QLabel:hover { color: #0078d4; }")  # Visual feedback on hover
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click events"""
        if event.button() == Qt.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


# ==================== HOTKEYS HELP DIALOG ====================

class HotkeysHelpDialog(QDialog):
    """Dialog displaying all keyboard shortcuts"""
    
    def __init__(self, shortcuts, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.resize(500, 600)

        layout = QVBoxLayout(self)

        # Create table
        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Shortcut", "Description"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.NoSelection)

        # Fill table
        table.setRowCount(len(shortcuts))
        for row, (key, desc) in enumerate(shortcuts):
            table.setItem(row, 0, QTableWidgetItem(key))
            table.setItem(row, 1, QTableWidgetItem(desc))

        layout.addWidget(table)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


# ==================== POINT TABLE ====================

class PointTable(QTableWidget):
    """Custom table widget for displaying point data with delete functionality"""
    
    point_selected = Signal(list)  # Emits list of selected points data: {frame, object_id, x, y}

    def __init__(self):
        super().__init__()
        self.current_frame = 0  # Track current frame for show_all_points feature
        self._setup_table()
        self.itemSelectionChanged.connect(self._on_selection_changed)
    
    def _setup_table(self):
        """Initialize table structure and appearance"""
        # Set up columns (added one more for delete button)
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["Frame", "Object ID", "Type", "X", "Y", "Action"])
        
        # Configure table behavior
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSortingEnabled(False)
        self.setAlternatingRowColors(True)
        
        # Configure column widths
        header = self.horizontalHeader()
        self.setColumnWidth(0, 60)    # Frame
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # Object ID auto-resize
        self.setColumnWidth(2, 60)    # Type
        self.setColumnWidth(3, 60)    # X
        self.setColumnWidth(4, 60)    # Y
        self.setColumnWidth(5, 80)    # Action (Delete button)
    
    def _create_colored_object_id_widget(self, object_id):
        """Create a widget with colored square and object ID number"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)
        
        # Create colored square
        if 0 <= object_id < len(sammie.PALETTE):
            color_widget = ColorDisplayWidget(sammie.PALETTE[object_id], size=(16, 12))
        else:
            # Fallback for out-of-range object IDs
            color_widget = ColorDisplayWidget((128, 128, 128), size=(16, 12))
        
        # Create label with object ID number
        id_label = QLabel(str(object_id))
        id_label.setAlignment(Qt.AlignCenter)
        
        # Get object name from settings
        settings_mgr = get_settings_manager()
        object_names = settings_mgr.get_session_setting("object_names", {})
        object_name = object_names.get(str(object_id), "")
        
        # Add to layout
        layout.addWidget(color_widget)
        layout.addWidget(id_label)

        # Add name label if there's a name
        if object_name:
            name_label = QLabel(f"({object_name})")
            name_label.setStyleSheet("color: gray; font-style: italic;")
            layout.addWidget(name_label)
        
        layout.addStretch()  # Push everything to the left

        return widget
    
    def add_point(self, frame, object_id, positive, x, y):
        """Add a new point to the table"""
        # Check if show_all_points is enabled and if frame matches current_frame
        settings_mgr = get_settings_manager()
        show_all_points = settings_mgr.get_session_setting("show_all_points", True)
        
        if not show_all_points and frame != self.current_frame:
            # Don't add the point if show_all_points is disabled and frame doesn't match
            return
        
        row_count = self.rowCount()
        self.insertRow(row_count)

        # Create regular items for most columns
        items = [
            (0, QTableWidgetItem(str(frame))),
            #(2, QTableWidgetItem("➕" if positive else "➖")),
            (3, QTableWidgetItem(str(x))),
            (4, QTableWidgetItem(str(y)))
        ]
        
        # Set regular items
        for col, item in items:
            self.setItem(row_count, col, item)

        # Set icon for positive/negative column
        type_item = QTableWidgetItem()
        if positive:
            type_item.setIcon(QIcon(":/icons/plus.png"))
        else:
            type_item.setIcon(QIcon(":/icons/minus.png"))
        self.setItem(row_count, 2, type_item)

        # Store the boolean value in the item's user data
        #type_item.setData(Qt.UserRole, positive)
        #self.setItem(row_count, 2, type_item)
        
        # Set colored widget for Object ID column
        colored_widget = self._create_colored_object_id_widget(object_id)
        self.setCellWidget(row_count, 1, colored_widget)
        
        # Add delete button in the last column
        self._add_delete_button(row_count)
        
        self.scrollToBottom()
    
    def set_current_frame(self, frame):
        """Update current frame and refresh table if show_all_points is enabled"""
        self.current_frame = frame
        settings_mgr = get_settings_manager()
        show_all_points = settings_mgr.get_session_setting("show_all_points", True)
        
        if not show_all_points:
            # Refresh table to show only points for current frame
            # This will be called by parent window with all points
            pass  # The refresh will be handled by parent window's _refresh_table method
    
    def _add_delete_button(self, row):
        """Add a delete button to the specified row"""
        delete_btn = QPushButton("Delete")
        delete_btn.setMaximumWidth(70)
        
        # Connect the button to delete this specific row
        delete_btn.clicked.connect(lambda _, r=row: self.delete_selected_row(r))
        
        # Create a widget container for centering the button
        widget_container = QWidget()
        layout = QHBoxLayout(widget_container)
        layout.addWidget(delete_btn)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setAlignment(Qt.AlignCenter)
        
        self.setCellWidget(row, 5, widget_container)
    
    def delete_selected_row(self, single_row):
        """Delete the currently selected rows or a single specified row"""
        selected_rows = self.selectionModel().selectedRows()
        
        # If there are selected rows AND the clicked row is in the selection, delete all selected
        if selected_rows and single_row in [idx.row() for idx in selected_rows]:
            rows = sorted([idx.row() for idx in selected_rows], reverse=True)
        # Otherwise, delete only the single clicked row
        else:
            rows = [single_row] if single_row is not None and single_row < self.rowCount() else []

        if not rows:
            return
        
        # Block signals to prevent selection changes during deletion
        self.blockSignals(True)

        # Batch remove all points first
        removed_points = []
        affected_frames = set()  # Track which frames need mask regeneration
        for row in rows:
            if row >= self.rowCount():
                continue
            
            # Get point data from row
            frame_item = self.item(row, 0)
            if not frame_item:
                continue
            
            frame = int(frame_item.text())
            affected_frames.add(frame) # This frame will be regenerated
            
            # Get object ID from the colored widget
            object_id = 0
            widget = self.cellWidget(row, 1)
            if widget:
                labels = widget.findChildren(QLabel)
                for label in labels:
                    try:
                        object_id = int(label.text())
                        break
                    except ValueError:
                        continue
            
            # Get coordinates
            x_item = self.item(row, 3)
            y_item = self.item(row, 4)
            if not x_item or not y_item:
                continue
            
            x, y = int(x_item.text()), int(y_item.text())
            
            # Remove point from manager
            if hasattr(self, 'parent_window') and hasattr(self.parent_window, 'point_manager'):
                removed_point = self.parent_window.point_manager.remove_point(frame, object_id, x, y)
                if removed_point:
                    removed_points.append(removed_point)
                    print(f"Deleted point: Frame {frame}, Object {object_id}, Position ({x}, {y})")
                elif len(rows) == 1:  # Single deletion - print warning
                    print(f"Warning: Point not found in manager: Frame {frame}, Object {object_id}, Position ({x}, {y})")
        
        # Remove all rows from table
        for row in rows:
            self.removeRow(row)
        self.clearSelection()
        self._update_delete_buttons()

        # Unblock signals after deletion is complete
        self.blockSignals(False)
        
        # Delete masks only for affected frames, then replay points to regenerate them
        if removed_points and hasattr(self, 'parent_window'):
            # Delete mask directories for affected frames only
            for frame in affected_frames:
                frame_mask_dir = os.path.join(sammie.mask_dir, f"{frame:05d}")
                if os.path.exists(frame_mask_dir):
                    shutil.rmtree(frame_mask_dir)

            # Replay points to regenerate masks for affected frames 
            points = self.parent_window.point_manager.get_all_points()
            self.parent_window.sam_manager.propagated = False
            self.parent_window.matany_manager.propagated = False
            self.parent_window.update_tracking_status()
            self.parent_window.sam_manager.replay_points(points)

    def _update_delete_buttons(self):
        """Update all delete button connections after a row is removed"""
        for row in range(self.rowCount()):
            widget_container = self.cellWidget(row, 5)
            if widget_container:
                # Find the delete button in the container
                delete_btn = widget_container.findChild(QPushButton)
                if delete_btn:
                    # Disconnect all previous connections and reconnect with correct row
                    delete_btn.clicked.disconnect()
                    delete_btn.clicked.connect(lambda _, r=row: self.delete_selected_row(r))
    
    def clear_points(self):
        """Remove all points from the table"""
        self.setRowCount(0)
    
    def remove_last_point(self):
        """Remove the most recently added point"""
        row_count = self.rowCount()
        if row_count > 0:
            self.removeRow(row_count - 1)

    def _on_selection_changed(self):
        """Handle row selection changes"""
        selected_rows = self.selectionModel().selectedRows()
        point_data = []
        if selected_rows:
            for row in selected_rows:
                i_row = row.row()
                point_data.append(self._get_point_from_row(i_row))
            if point_data:
                self.point_selected.emit(point_data)
        else:
            # No selection - emit None or empty dict
            self.point_selected.emit([])
    
    def _get_point_from_row(self, row):
        """Extract point data from a table row"""
        if row >= self.rowCount():
            return None
        
        # Get frame
        frame_item = self.item(row, 0)
        if not frame_item:
            return None
        frame = int(frame_item.text())
        
        # Get object ID
        object_id = 0
        widget = self.cellWidget(row, 1)
        if widget:
            labels = widget.findChildren(QLabel)
            for label in labels:
                try:
                    object_id = int(label.text())
                    break
                except ValueError:
                    continue
        
        # Get coordinates
        x_item = self.item(row, 3)
        y_item = self.item(row, 4)
        if not x_item or not y_item:
            return None
        
        x = int(x_item.text())
        y = int(y_item.text())
        
        return {
            'frame': frame,
            'object_id': object_id,
            'x': x,
            'y': y
        }


# ==================== IMAGE VIEWER ====================

class ImageViewer(QGraphicsView):
    """Custom graphics view for image display with zoom and pan functionality"""
    
    # Add signal for point clicks
    point_clicked = Signal(int, int, bool)  # x, y coordinates, is_positive
    # Add signal for file drops
    file_dropped = Signal(str)  # file path
    
    def __init__(self, status_callback=None, parent_window=None):
        super().__init__()
        self._setup_graphics_view()
        self._init_variables()
        self.status_callback = status_callback
        self.parent_window = parent_window

        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def _setup_graphics_view(self):
        """Initialize the graphics view and scene"""
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        # Rendering hints for higher quality scaling
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)
    
    def _init_variables(self):
        """Initialize state variables"""
        self._is_panning = False
        self._pan_start = QPointF()
        self.point_editing_enabled = True
        self.original_pixmap = None
        self.fit_scale = 1.0
        self.min_scale = 1.0
        self.max_scale = 8.0
        self.current_scale = 1.0
        self.has_been_initialized = False  # Track if we've done initial zoom


    def load_image(self, image, preserve_zoom=True):
        """Load and display an image from file path"""
        pixmap = image
        if pixmap:
            # Store current view state if preserving zoom
            if preserve_zoom and self.original_pixmap and self.has_been_initialized:
                # Save scrollbar positions directly (more reliable than centerOn)
                h_value = self.horizontalScrollBar().value()
                v_value = self.verticalScrollBar().value()
                current_zoom = self.current_scale
                should_restore = True
            else:
                h_value = 0
                v_value = 0
                current_zoom = None
                should_restore = False
            
            # Update the pixmap
            self.original_pixmap = pixmap
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            
            # Only update scene rect if dimensions changed
            if self.sceneRect() != pixmap.rect():
                self.setSceneRect(pixmap.rect())
                self._update_fit_scale()
            
            # Apply zoom based on preserve_zoom setting
            if should_restore and current_zoom is not None:
                # Keep the same zoom level
                if abs(self.current_scale - current_zoom) > 0.001:
                    self.resetTransform()
                    self.scale(current_zoom, current_zoom)
                    self.current_scale = current_zoom
                
                # Restore exact scrollbar positions
                self.horizontalScrollBar().setValue(h_value)
                self.verticalScrollBar().setValue(v_value)
                self._update_status_text()
            else:
                # First time loading or explicitly requested reset
                self.set_zoom(self.fit_scale)
                self.has_been_initialized = True
            
            return True
        return False

    def update_image(self, image):
        """Update image while preserving zoom/pan - convenience method"""
        return self.load_image(image, preserve_zoom=True)
    
    def load_image_reset_zoom(self, image):
        """Load image and reset zoom to fit - convenience method"""
        return self.load_image(image, preserve_zoom=False)

    def clear_image(self):
        """Clear the currently displayed image and reset viewer state"""
        # Clear the pixmap from the graphics item
        self.pixmap_item.setPixmap(QPixmap())
        
        # Reset the scene rectangle
        self.setSceneRect(0, 0, 0, 0)
        
        # Clear stored references
        self.original_pixmap = None
        
        # Reset scale values
        self.fit_scale = 0.1
        self.min_scale = 0.1
        self.current_scale = 1.0
        
        # Reset the transform
        self.resetTransform()
    
    def _update_fit_scale(self):
        """Calculate the scale factor needed to fit image in viewport"""
        if not self.original_pixmap:
            return
        
        viewport_size = self.viewport().size()
        image_size = self.original_pixmap.size()
        
        scale_w = viewport_size.width() / image_size.width()
        scale_h = viewport_size.height() / image_size.height()
        
        self.fit_scale = min(scale_w, scale_h)
        self.min_scale = min(1.0, self.fit_scale)
    
    def set_zoom(self, scale_factor):
        """Set the zoom level to a specific scale factor"""
        if not self.original_pixmap:
            return
        
        # Clamp scale factor to valid range
        scale_factor = max(self.min_scale, min(self.max_scale, scale_factor))
        
        # Preserve center point during zoom
        center_before = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        
        self.scale(scale_factor, scale_factor)
        self.centerOn(center_before)
        
        self.current_scale = scale_factor
        self._update_status_text()
    
    def zoom_to_fit(self):
        """Zoom to fit the entire image in the viewport"""
        self._update_fit_scale()
        self.set_zoom(self.fit_scale)
    
    def zoom_to_100(self):
        """Zoom to 100% (1:1 pixel ratio)"""
        self.set_zoom(1.0)
    
    def dragEnterEvent(self, event):
        """Handle drag enter events"""
        # Check if the dragged data contains URLs (files)
        if event.mimeData().hasUrls():
            # Get the first URL
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                # Check if it's a supported file type
                supported_extensions = [
                    '.mp4', '.m4v', '.mkv', '.mov', '.avi', '.webm',
                    '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'
                ]
                if any(file_path.lower().endswith(ext) for ext in supported_extensions):
                    event.acceptProposedAction()
                    return
        event.ignore()
    
    def dragMoveEvent(self, event):
        """Handle drag move events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        """Handle drop events"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                # Emit signal to notify parent window
                self.file_dropped.emit(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    # ==================== EVENT HANDLERS ====================
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel zoom"""
        if not self.original_pixmap:
            return
        
        zoom_factor = 1.5 if event.angleDelta().y() > 0 else 1 / 1.5
        new_scale = self.current_scale * zoom_factor
        self.set_zoom(new_scale)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events for clicking and panning"""
        if event.button() == Qt.MiddleButton:
            self._is_panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            super().mousePressEvent(event)
            return
    
        if self.point_editing_enabled:
            if event.button() == Qt.LeftButton:
                # Left click = positive point
                mouse_pos = self.mapToScene(event.position().toPoint())
                x, y = int(mouse_pos.x()), int(mouse_pos.y())
                
                # Check if click is within image bounds
                if (self.original_pixmap and 
                    0 <= x < self.original_pixmap.width() and 
                    0 <= y < self.original_pixmap.height()):
                    
                    # Emit signal for positive point addition
                    self.point_clicked.emit(x, y, True)
                    return  # Don't call super() to prevent other handling
            
            elif event.button() == Qt.RightButton:
                # Right click = negative point
                mouse_pos = self.mapToScene(event.position().toPoint())
                x, y = int(mouse_pos.x()), int(mouse_pos.y())
                
                # Check if click is within image bounds
                if (self.original_pixmap and 
                    0 <= x < self.original_pixmap.width() and 
                    0 <= y < self.original_pixmap.height()):
                    
                    # Emit signal for negative point addition
                    self.point_clicked.emit(x, y, False)
                    return  # Don't call super() to prevent context menu
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement for panning and coordinate display"""
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            self._update_mouse_status(event.position().toPoint())
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        
        super().mouseReleaseEvent(event)
    
    def resizeEvent(self, event):
        """Handle viewport resize events"""
        super().resizeEvent(event)
        
        if self.original_pixmap:
            previous_fit_scale = self.fit_scale
            self._update_fit_scale()
            
            # If currently at fit scale, maintain fit on resize
            if abs(self.current_scale - previous_fit_scale) < 0.001:
                self.set_zoom(self.fit_scale)
    
    # ==================== STATUS UPDATES ====================
    
    def _update_mouse_status(self, pos):
        """Update status bar with current mouse coordinates"""
        if not self.original_pixmap:
            return
        
        scene_pos = self.mapToScene(pos)
        x, y = int(scene_pos.x()), int(scene_pos.y())
        
        if (0 <= x < self.original_pixmap.width() and 
            0 <= y < self.original_pixmap.height()):
            self._update_status_text(x, y)
    
    def _update_status_text(self, x=None, y=None):
        """Update the status text with coordinates and zoom level"""
        if self.status_callback:
            zoom_text = f"Zoom: {int(self.current_scale * 100)}%"
            if x is not None and y is not None:
                self.status_callback(f"Pos: ({x}, {y})   {zoom_text}")
            else:
                self.status_callback(f"{zoom_text}")


class FrameSlider(QSlider):
    """
    Custom slider with visual feedback for in/out points.
    
    Features:
    - Triangular markers for in/out points
    - Highlighted range between in/out points
    - Gray/unhighlighted areas outside the range
    """
    
    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._in_point = None
        self._out_point = None
        
        # Colors for the range highlight
        self._highlight_color = None  # Will use palette highlight color
        self._background_color = None  # Will use palette mid color
        
    def set_in_point(self, frame):
        """Set the in point to the specified frame"""
        if frame is not None and (frame < self.minimum() or frame > self.maximum()):
            return
        self._in_point = frame
        if self.isVisible():  # Only update if widget is visible and ready
            self.update()
    
    def set_out_point(self, frame):
        """Set the out point to the specified frame"""
        if frame is not None and (frame < self.minimum() or frame > self.maximum()):
            return
        self._out_point = frame
        if self.isVisible():  # Only update if widget is visible and ready
            self.update()
    
    def get_in_point(self):
        """Get the current in point"""
        return self._in_point
    
    def get_out_point(self):
        """Get the current out point"""
        return self._out_point
    
    def clear_in_out_points(self):
        """Clear both in and out points"""
        self._in_point = None
        self._out_point = None
        self.update()
    
    def _frame_to_pixel(self, frame):
        """Convert a frame number to pixel position in the groove"""
        if self.maximum() == self.minimum():
            return 0

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        # Get groove rect
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )

        # Get handle rect for width calculation
        handle = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
        )

        # Use Qt's built-in position calculation for consistency
        # This matches exactly how Qt calculates the handle position
        slider_pos = QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            frame,
            groove.width() - handle.width(),  # Available span
            opt.upsideDown
        )
        
        # The position is relative to the start of the usable range
        # Add groove left and half handle width to get absolute pixel position
        pixel = groove.left() + handle.width() // 2 + slider_pos
        
        return pixel
    
    def paintEvent(self, event):
        """Custom paint event to draw range highlight and markers"""
        try:
            # Safety check - ensure widget is properly initialized
            if not self.isVisible() or self.width() <= 0 or self.height() <= 0:
                super().paintEvent(event)
                return
                
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            # Get style option for the slider
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            
            # Get the groove rectangle
            groove_rect = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
            )
            
            # Draw the highlighted range if both in and out points are set
            if self._in_point is not None and self._out_point is not None:
                try:
                    self._draw_range_highlight(painter, groove_rect)
                except Exception as e:
                    print(f"Error drawing range highlight: {e}")
            
            # Let the default slider draw itself
            painter.end()
            super().paintEvent(event)
            
            # Draw the in/out point markers on top
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            if self._in_point is not None:
                try:
                    self._draw_marker(painter, self._in_point, is_in_point=True)
                except Exception as e:
                    print(f"Error drawing in point marker: {e}")
            
            if self._out_point is not None:
                try:
                    self._draw_marker(painter, self._out_point, is_in_point=False)
                except Exception as e:
                    print(f"Error drawing out point marker: {e}")
                    
            painter.end()
            
        except Exception as e:
            print(f"Error in paintEvent: {e}")
            import traceback
            traceback.print_exc()
            super().paintEvent(event)
    
    def _draw_range_highlight(self, painter, groove_rect):
        """Draw the highlighted range between in and out points"""
        start_frame = min(self._in_point, self._out_point)
        end_frame = max(self._in_point, self._out_point)
        
        start_pixel = self._frame_to_pixel(start_frame)
        end_pixel = self._frame_to_pixel(end_frame)
        
        # Get colors from palette
        palette = self.palette()
        highlight_color = palette.color(QPalette.Highlight)
        highlight_color.setAlpha(60)  # Semi-transparent
        
        # Draw the highlighted rectangle over the groove
        if self.orientation() == Qt.Horizontal:
            highlight_rect = QRect(
                start_pixel,
                groove_rect.top(),
                end_pixel - start_pixel,
                groove_rect.height()
            )
        else:
            highlight_rect = QRect(
                groove_rect.left(),
                start_pixel,
                groove_rect.width(),
                end_pixel - start_pixel
            )
        
        painter.fillRect(highlight_rect, highlight_color)
    
    def _draw_marker(self, painter, frame, is_in_point):
        """Draw a bracket marker for an in or out point"""
        pixel_pos = self._frame_to_pixel(frame)
        
        # Get groove rectangle for positioning
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
        )
        
        # Use solid black for both markers
        pen_color = QColor(0, 0, 0)
        painter.setPen(QPen(pen_color, 2))  # 2px line thickness
        
        # Bracket centered vertically on the groove
        bracket_height = 15
        bracket_width = 4
        y_center = groove_rect.top() + groove_rect.height() // 2
        y_top = y_center - bracket_height // 2
        y_bottom = y_center + bracket_height // 2
        
        if is_in_point:
            # In point: left bracket [
            painter.drawLine(pixel_pos, y_top, pixel_pos, y_bottom)  # Vertical line
            painter.drawLine(pixel_pos, y_top, pixel_pos + bracket_width, y_top)  # Top horizontal
            painter.drawLine(pixel_pos, y_bottom, pixel_pos + bracket_width, y_bottom)  # Bottom horizontal
        else:
            # Out point: right bracket ]
            painter.drawLine(pixel_pos, y_top, pixel_pos, y_bottom)  # Vertical line
            painter.drawLine(pixel_pos - bracket_width, y_top, pixel_pos, y_top)  # Top horizontal
            painter.drawLine(pixel_pos - bracket_width, y_bottom, pixel_pos, y_bottom)  # Bottom horizontal