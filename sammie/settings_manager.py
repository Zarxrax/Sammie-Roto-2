# sammie/settings_manager.py
import json
import os
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path

@dataclass
class ApplicationSettings:
    """Global application settings that persist across sessions"""
    # UI Layout
    main_splitter_sizes: list = field(default_factory=lambda: [1000, 300])
    vertical_splitter_sizes: list = field(default_factory=lambda: [600, 208])
    bottom_splitter_sizes: list = field(default_factory=lambda: [450, 550])
    
    # Window state
    window_width: int = 1400
    window_height: int = 900
    window_maximized: bool = False
    
    # View defaults
    default_view_mode: str = "Segmentation-Edit"
    default_show_masks: bool = True
    default_show_outlines: bool = True
    default_antialias: bool = False
    default_show_removal_mask: bool = True
    default_bgcolor: tuple = (0, 255, 0)  # RGB green
    
    # Segmentation defaults
    default_object_id: int = 0
    
    # Segmentation Processing defaults
    default_holes: int = 0
    default_dots: int = 0
    default_border_fix: int = 0
    default_grow: int = 0
    default_show_all_points: bool = True  # If True, show points from all frames
    
    # MatAnyone Processing defaults
    default_matany_grow: int = 0
    default_matany_gamma: float = 1.0
    default_matany_res: int = 720

    # Object Removal Processing defaults
    default_removal_method: str = "Minimax-Remover"
    default_inpaint_method: str = "Telea"
    default_inpaint_radius: int = 3
    default_inpaint_grow: int = 5
    default_minimax_steps: int = 6
    default_minimax_resolution: int = 480
    default_minimax_vae_tiling: bool = False
    
    # Playback
    playback_fps: int = 24  # frames per second for playback
    
    # Performance
    sam_model: str = "Base"
    force_cpu: bool = False    
    frame_format: str = "png"
    display_update_frequency: int = 5
    
    # Deduplication
    dedupe_threshold: float = 0.8
    
    # Export dialog defaults
    export_codec: str = "prores"
    export_output_type: str = "Matte" 
    export_use_input_folder: bool = True
    export_filename_template: str = "{input_name}-{output_type}"
    export_antialias: bool = False
    export_quantizer: int = 14
    export_include_original: bool = False
    export_multiple: bool = False
    export_folder_path: str = ""

@dataclass  
class SessionSettings:
    """Session-specific settings that are saved with each video/project"""
    # Video information
    video_file_path: str = ""
    frame_format: str = "png"
    video_width: int = 0
    video_height: int = 0
    video_fps: float = 0.0
    total_frames: int = 0
    
    # Current state
    current_frame: int = 0
    current_view_mode: str = "Segmentation-Edit"
    
    # View options
    show_masks: bool = True
    show_outlines: bool = True
    antialias: bool = False
    show_removal_mask: bool = True
    bgcolor: tuple = (0, 255, 0)  # RGB green
    show_all_points: bool = True  # If True, show points from all frames
    
    # Segmentation parameters
    holes: int = 0
    dots: int = 0
    border_fix: int = 0
    grow: int = 0
    
    # MatAnyone parameters
    matany_grow: int = 0
    matany_gamma: float = 1.0
    matany_res: int = 720

    # Object removal parameters
    inpaint_method: str = "Telea"
    inpaint_radius: int = 3
    inpaint_grow: int = 0
    minimax_steps: int = 6
    minimax_resolution: int = 480
    minimax_vae_tiling: bool = False
    
    # Selected objects
    selected_object_id: int = 0
    
    # Object names
    object_names: dict = field(default_factory=dict)
    
    # Tracking state
    is_propagated: bool = False
    is_deduplicated: bool = False
    is_matted: bool = False
    is_removed: bool = False
    
    # Session metadata
    created_timestamp: str = ""
    modified_timestamp: str = ""


class SettingsManager:
    """Centralized settings management for the application"""
    
    def __init__(self, temp_dir: str = "temp"):
        self.temp_dir = temp_dir

        # Rename .json settings files to .conf for backward compatibility.
        # This will be removed in a future version
        try:
            if os.path.exists("sammie_settings.json") :
                shutil.move("sammie_settings.json", "sammie_settings.conf")
                if os.path.exists(os.path.join(temp_dir, "session_settings.json")) :
                    shutil.move(os.path.join(temp_dir, "session_settings.json"), os.path.join(temp_dir, "session_settings.conf"))
        except Exception as e:
            print(f"Warning: Failed to rename settings files: {e}")

        self.app_settings_file = "sammie_settings.conf"
        self.session_settings_file = os.path.join(temp_dir, "session_settings.conf")
        self.points_file = os.path.join(temp_dir, "points.json")
        
        # Initialize settings
        self.app_settings = ApplicationSettings()
        self.session_settings = SessionSettings()
        
        # Load existing settings
        self.load_app_settings()
    
    # ==================== APPLICATION SETTINGS ====================
    
    def load_app_settings(self) -> bool:
        """Load application settings from file"""
        try:
            if os.path.exists(self.app_settings_file):
                with open(self.app_settings_file, 'r') as f:
                    data = json.load(f)
                    # Update existing settings with loaded data
                    for key, value in data.items():
                        if hasattr(self.app_settings, key):
                            setattr(self.app_settings, key, value)
                #print("Application settings loaded")
                return True
        except Exception as e:
            print(f"Error loading application settings: {e}")
        return False
    
    def save_app_settings(self) -> bool:
        """Save application settings to file"""
        try:
            with open(self.app_settings_file, 'w') as f:
                json.dump(asdict(self.app_settings), f, indent=2)
            # print("Application settings saved - internal")
            return True
        except Exception as e:
            print(f"Error saving application settings: {e}")
            return False
    
    def get_app_setting(self, key: str, default: Any = None) -> Any:
        """Get an application setting value"""
        return getattr(self.app_settings, key, default)
    
    def set_app_setting(self, key: str, value: Any) -> bool:
        """Set an application setting value"""
        if hasattr(self.app_settings, key):
            setattr(self.app_settings, key, value)
            return True
        return False
    
    # ==================== SESSION SETTINGS ====================
    
    def load_session_settings(self) -> bool:
        """Load session settings from temp directory"""
        try:
            if os.path.exists(self.session_settings_file):
                with open(self.session_settings_file, 'r') as f:
                    data = json.load(f)
                    # Update existing settings with loaded data
                    for key, value in data.items():
                        if hasattr(self.session_settings, key):
                            setattr(self.session_settings, key, value)
                return True
        except Exception as e:
            print(f"Error loading session settings: {e}")
        return False
    
    def save_session_settings(self) -> bool:
        """Save session settings to temp directory"""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Update timestamp
            from datetime import datetime
            self.session_settings.modified_timestamp = datetime.now().isoformat()
            
            with open(self.session_settings_file, 'w') as f:
                json.dump(asdict(self.session_settings), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session settings: {e}")
            return False
    
    def get_session_setting(self, key: str, default: Any = None) -> Any:
        """Get a session setting value"""
        return getattr(self.session_settings, key, default)
    
    def set_session_setting(self, key: str, value: Any) -> bool:
        """Set a session setting value"""
        if hasattr(self.session_settings, key):
            setattr(self.session_settings, key, value)
            return True
        return False
    
    def create_new_session(self, video_path: str = "") -> None:
        """Create a new session with default values"""
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        # Reset to defaults, incorporating app setting defaults
        self.session_settings = SessionSettings(
            video_file_path=video_path,
            frame_format=self.app_settings.frame_format,
            current_view_mode=self.app_settings.default_view_mode,
            show_masks=self.app_settings.default_show_masks,
            show_outlines=self.app_settings.default_show_outlines,
            antialias=self.app_settings.default_antialias,
            show_removal_mask=self.app_settings.default_show_removal_mask,
            bgcolor=self.app_settings.default_bgcolor,
            show_all_points=self.app_settings.default_show_all_points,
            holes=self.app_settings.default_holes,
            dots=self.app_settings.default_dots,
            border_fix=self.app_settings.default_border_fix,
            grow=self.app_settings.default_grow,
            matany_grow=self.app_settings.default_matany_grow,
            matany_gamma=self.app_settings.default_matany_gamma,
            matany_res = self.app_settings.default_matany_res,
            inpaint_method=self.app_settings.default_inpaint_method,
            inpaint_radius=self.app_settings.default_inpaint_radius,
            inpaint_grow=self.app_settings.default_inpaint_grow,
            minimax_steps=self.app_settings.default_minimax_steps,
            minimax_resolution=self.app_settings.default_minimax_resolution,
            minimax_vae_tiling=self.app_settings.default_minimax_vae_tiling,
            selected_object_id=self.app_settings.default_object_id,
            created_timestamp=timestamp,
            modified_timestamp=timestamp
        )
    
    # ==================== POINTS MANAGEMENT ====================
    
    def save_points(self, points_list: list) -> bool:
        """Save points to session directory"""
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
            with open(self.points_file, 'w') as f:
                json.dump(points_list, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving points: {e}")
            return False
    
    def load_points(self) -> Optional[list]:
        """Load points from session directory"""
        try:
            if os.path.exists(self.points_file):
                with open(self.points_file, 'r') as f:
                    points = json.load(f)
                if len(points) > 0: # Only print a message if there are points to load
                    print(f"Loaded {len(points)} points from session")
                return points
        except Exception as e:
            print(f"Error loading points: {e}")
        return None
    
    # ==================== CONVENIENCE METHODS ====================
    
    def update_video_info(self, width: int, height: int, fps: float, total_frames: int, video_path: str = ""):
        """Update video information in session settings"""
        self.session_settings.video_width = width
        self.session_settings.video_height = height
        self.session_settings.video_fps = fps
        self.session_settings.total_frames = total_frames
        if video_path:
            self.session_settings.video_file_path = video_path
    
    def get_view_options(self) -> Dict[str, Any]:
        """Get current view options as a dictionary"""
        return {
            'view_mode': self.session_settings.current_view_mode,
            'show_masks': self.session_settings.show_masks,
            'show_outlines': self.session_settings.show_outlines,
            'antialias': self.session_settings.antialias,
            'bgcolor': self.session_settings.bgcolor,
            'show_removal_mask': self.session_settings.show_removal_mask
        }
    
    def get_segmentation_params(self) -> Dict[str, Any]:
        """Get current segmentation parameters"""
        return {
            'holes': self.session_settings.holes,
            'dots': self.session_settings.dots,
            'border_fix': self.session_settings.border_fix,
            'grow': self.session_settings.grow
        }
    
    def get_matting_params(self) -> Dict[str, Any]:
        """Get current matting parameters"""
        return {
            'matany_gamma': self.session_settings.matany_gamma,
            'matany_grow': self.session_settings.matany_grow,
            'matany_res': self.session_settings.matany_res
        }
    
    def get_inpainting_params(self) -> Dict[str, Any]:
        """Get current inpainting parameters"""
        return {
            'inpaint_method': self.session_settings.inpaint_method,
            'inpaint_radius': self.session_settings.inpaint_radius,
            'inpaint_grow': self.session_settings.inpaint_grow,
            'minimax_steps': self.session_settings.minimax_steps,
            'minimax_resolution': self.session_settings.minimax_resolution,
            'minimax_vae_tiling': self.session_settings.minimax_vae_tiling
        }

    def session_exists(self) -> bool:
        """Check if a session exists in temp directory"""
        return (os.path.exists(self.session_settings_file) or 
                os.path.exists(self.points_file) or
                (os.path.exists(self.temp_dir) and os.path.exists(os.path.join(self.temp_dir, "frames"))))
    
    def clear_session(self) -> bool:
        """Clear current session data"""
        try:
            # Remove session files
            files_to_remove = [self.session_settings_file, self.points_file]
            for file_path in files_to_remove:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Reset session settings to defaults
            self.create_new_session()
            print("Session cleared")
            return True
        except Exception as e:
            print(f"Error clearing session: {e}")
            return False
    
    # ==================== IMPORT/EXPORT ====================
    
    def export_session(self, export_path: str) -> bool:
        """Export current session (settings + points) to a file"""
        try:
            export_data = {
                'session_settings': asdict(self.session_settings),
                'points': self.load_points() or []
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"Session exported to {export_path}")
            return True
        except Exception as e:
            print(f"Error exporting session: {e}")
            return False
    
    def import_session(self, import_path: str) -> bool:
        """Import session (settings + points) from a file"""
        try:
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            # Load session settings
            if 'session_settings' in data:
                for key, value in data['session_settings'].items():
                    if hasattr(self.session_settings, key):
                        setattr(self.session_settings, key, value)
            
            # Save points separately
            if 'points' in data:
                self.save_points(data['points'])
            
            # Save the imported session
            self.save_session_settings()
            print(f"Session imported from {import_path}")
            return True
        except Exception as e:
            print(f"Error importing session: {e}")
            return False


# ==================== GLOBAL SETTINGS INSTANCE ====================

# Global settings manager instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

def initialize_settings(temp_dir: str = "temp") -> SettingsManager:
    """Initialize the global settings manager"""
    global _settings_manager
    _settings_manager = SettingsManager(temp_dir)
    return _settings_manager

# ==================== CONVENIENCE FUNCTIONS ====================

def get_app_setting(key: str, default: Any = None) -> Any:
    """Convenience function to get application setting"""
    return get_settings_manager().get_app_setting(key, default)

def set_app_setting(key: str, value: Any) -> bool:
    """Convenience function to set application setting"""
    return get_settings_manager().set_app_setting(key, value)

def get_session_setting(key: str, default: Any = None) -> Any:
    """Convenience function to get session setting"""
    return get_settings_manager().get_session_setting(key, default)

def set_session_setting(key: str, value: Any) -> bool:
    """Convenience function to set session setting"""
    return get_settings_manager().set_session_setting(key, value)

def save_app_settings() -> bool:
    """Convenience function to save application settings"""
    return get_settings_manager().save_app_settings()

def save_session_settings() -> bool:
    """Convenience function to save session settings"""
    return get_settings_manager().save_session_settings()
