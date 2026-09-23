"""Shared state and output cleanup for object removal engines."""
import os
import shutil
from sammie import core

class RemovalManager:
    """Manager for object removal operations"""

    def __init__(self):
        self.pipe = None
        self.propagated = False  # whether removal has been completed
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for removal events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")

    def clear_removal(self):
        """Clear removal data"""
        if os.path.exists(core.removal_dir):
            shutil.rmtree(core.removal_dir)
        os.makedirs(core.removal_dir)
        self.propagated = False
        print("Object removal data cleared")
