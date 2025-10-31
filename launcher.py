from PySide6.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import Qt, QLockFile, QDir
import sys
import argparse
import os
import traceback

def show_splash(app):
    splash_pix = QPixmap("sammie/splash.webp")
    #splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash = QSplashScreen(splash_pix, Qt.SplashScreen)
    splash.showMessage("Loading Sammie-Roto...", Qt.AlignCenter | Qt.AlignBottom, Qt.white)
    splash.show()
    app.processEvents()
    return splash

def check_single_instance():
    """
    Check if another instance is already running.
    Returns (lock_file, is_first_instance)
    """
    # Use a lock file in the temp directory
    lock_file_path = QDir.tempPath() + "/sammie-roto.lock"
    lock_file = QLockFile(lock_file_path)
    
    # Try to lock - returns True if successful (first instance)
    if lock_file.tryLock(100):  # 100ms timeout
        return lock_file, True
    else:
        return lock_file, False
    
def show_error_dialog(app, error_message, detailed_error=""):
    """Show an error dialog with the crash information"""
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setWindowTitle("Sammie-Roto Startup Error")
    msg_box.setText("Sammie-Roto encountered an error during startup and cannot continue.")
    msg_box.setInformativeText(error_message)
    
    if detailed_error:
        msg_box.setDetailedText(detailed_error)
    
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.setDefaultButton(QMessageBox.Ok)
    
    # Make sure the dialog appears on top
    msg_box.setWindowFlags(msg_box.windowFlags() | Qt.WindowStaysOnTopHint)
    
    try:
        with open("sammie_debug.log", "w", encoding="utf-8") as f:
            f.write(error_message + '\n' + detailed_error)
    except:
        pass

    return msg_box.exec()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Sammie-Roto: Video Segmentation and Matting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python launcher.py                          # Start with GUI file picker
    python launcher.py video.mp4               # Load specific video file
    python launcher.py image.jpg               # Load specific image file
    python launcher.py --file path/to/video.mp4 # Load using --file flag
            """
    )
    
    # Positional argument for file path (most common usage)
    parser.add_argument(
        'file', 
        nargs='?', 
        help='Path to video or image file to load'
    )
    
    # Alternative --file flag (for explicit usage)
    parser.add_argument(
        '--file', '-f',
        dest='file_flag',
        help='Path to video or image file to load'
    )
    
    args = parser.parse_args()
    
    # Use either positional argument or --file flag
    file_path = args.file or args.file_flag
    
    # Validate file exists if provided
    if file_path:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
    
    return file_path

if __name__ == "__main__":
    # Parse command line arguments first
    file_to_load = parse_arguments()

    try:
        if os.name == 'nt':
            import ctypes
            myappid = 'Sammie-Roto.2'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('sammie/icon.ico'))
    
    # Check for single instance
    lock_file, is_first = check_single_instance()
    if not is_first:
        QMessageBox.warning(
            None,
            "Already Running",
            "Sammie-Roto is already running.\nOnly one instance can run at a time."
        )
        sys.exit(0)

    splash = show_splash(app)

    try:
        # Import heavy stuff after splash
        from sammie_main import MainWindow

        # Create window and pass file path if provided
        window = MainWindow(initial_file=file_to_load)
        window.show()
        splash.finish(window)

        sys.exit(app.exec())
    
    except ImportError as e:
        splash.close()
        error_msg = f"Failed to import required modules: {str(e)}"
        detailed_error = traceback.format_exc()
        show_error_dialog(app, error_msg, detailed_error)
        print(f"Import Error: {error_msg}")
        print(detailed_error)
        sys.exit(1)
        
    except Exception as e:
        splash.close()
        error_msg = f"Unexpected error during startup: {str(e)}"
        detailed_error = traceback.format_exc()
        show_error_dialog(app, error_msg, detailed_error)
        print(f"Startup Error: {error_msg}")
        print(detailed_error)
        sys.exit(1)