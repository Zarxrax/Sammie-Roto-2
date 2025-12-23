# sammie/export_formats.py
"""
Export format definitions and configuration.
Each format class encapsulates format-specific behavior.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os


@dataclass
class ExportSettings:
    """Unified export settings structure"""
    format_id: str
    output_dir: str
    filename_template: str
    output_type: str
    object_id: int  # -1 for all objects
    antialias: bool
    quality: int  # CRF/CQ value for lossy codecs
    use_inout: bool
    in_point: Optional[int]
    out_point: Optional[int]
    include_original: bool = False  # EXR only
    export_multiple: bool = False  # Video only


class ExportFormat(ABC):
    """Base class for export formats"""
    
    @property
    @abstractmethod
    def format_id(self) -> str:
        """Unique identifier for this format"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Display name in UI"""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension including dot"""
        pass
    
    @property
    @abstractmethod
    def supports_alpha(self) -> bool:
        """Whether format supports alpha channel"""
        pass
    
    @property
    @abstractmethod
    def supports_quality_setting(self) -> bool:
        """Whether format supports quality/CRF setting"""
        pass
    
    @property
    @abstractmethod
    def supports_multiple_export(self) -> bool:
        """Whether format supports exporting multiple objects as separate files"""
        pass
    
    @property
    @abstractmethod
    def supports_include_original(self) -> bool:
        """Whether format supports including original frames"""
        pass
    
    @property
    @abstractmethod
    def is_sequence(self) -> bool:
        """Whether this format exports frame sequences"""
        pass
    
    @abstractmethod
    def get_available_output_types(self) -> List[str]:
        """Get list of valid output types for this format"""
        pass
    
    @abstractmethod
    def get_codec_name(self) -> Optional[str]:
        """Get codec name for video formats (None for sequences)"""
        pass
    
    @abstractmethod
    def get_pixel_format(self, has_alpha: bool) -> str:
        """Get pixel format for encoding"""
        pass
    
    @abstractmethod
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        """Get codec-specific options"""
        pass
    
    def get_quality_label(self) -> str:
        """Get label for quality setting"""
        return "Quality (CRF):"
    
    def get_quality_range(self) -> tuple:
        """Get min/max range for quality setting"""
        return (0, 51)
    
    def get_default_quality(self) -> int:
        """Get default quality value"""
        return 14


# === Video Formats ===

class ProResFormat(ExportFormat):
    @property
    def format_id(self) -> str:
        return "prores"
    
    @property
    def display_name(self) -> str:
        return "ProRes Video"
    
    @property
    def file_extension(self) -> str:
        return ".mov"
    
    @property
    def supports_alpha(self) -> bool:
        return True
    
    @property
    def supports_quality_setting(self) -> bool:
        return False
    
    @property
    def supports_multiple_export(self) -> bool:
        return True
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return False
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> str:
        return "prores_ks"
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return 'yuva444p10le' if has_alpha else 'yuv422p10le'
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {'profile': '4' if self.supports_alpha else '3'}


class FFV1Format(ExportFormat):
    @property
    def format_id(self) -> str:
        return "ffv1"
    
    @property
    def display_name(self) -> str:
        return "FFV1 Video"
    
    @property
    def file_extension(self) -> str:
        return ".mkv"
    
    @property
    def supports_alpha(self) -> bool:
        return True
    
    @property
    def supports_quality_setting(self) -> bool:
        return False
    
    @property
    def supports_multiple_export(self) -> bool:
        return True
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return False
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> str:
        return "ffv1"
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return 'bgra' if has_alpha else 'bgr0'
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {'level': '3'}


class H264Format(ExportFormat):
    @property
    def format_id(self) -> str:
        return "x264"
    
    @property
    def display_name(self) -> str:
        return "H.264 Video"
    
    @property
    def file_extension(self) -> str:
        return ".mp4"
    
    @property
    def supports_alpha(self) -> bool:
        return False
    
    @property
    def supports_quality_setting(self) -> bool:
        return True
    
    @property
    def supports_multiple_export(self) -> bool:
        return True
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return False
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> str:
        return "libx264"
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return 'yuv420p'
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {'crf': str(quality), 'preset': 'medium'}


class H265Format(ExportFormat):
    @property
    def format_id(self) -> str:
        return "x265"
    
    @property
    def display_name(self) -> str:
        return "H.265 Video"
    
    @property
    def file_extension(self) -> str:
        return ".mp4"
    
    @property
    def supports_alpha(self) -> bool:
        return False
    
    @property
    def supports_quality_setting(self) -> bool:
        return True
    
    @property
    def supports_multiple_export(self) -> bool:
        return True
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return False
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> str:
        return "libx265"
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return 'yuv420p'
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {'crf': str(quality), 'preset': 'medium'}


class VP9Format(ExportFormat):
    @property
    def format_id(self) -> str:
        return "vp9"
    
    @property
    def display_name(self) -> str:
        return "VP9 Video"
    
    @property
    def file_extension(self) -> str:
        return ".webm"
    
    @property
    def supports_alpha(self) -> bool:
        return True
    
    @property
    def supports_quality_setting(self) -> bool:
        return True
    
    @property
    def supports_multiple_export(self) -> bool:
        return True
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return False
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> str:
        return "libvpx-vp9"
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return 'yuva420p' if has_alpha else 'yuv420p'
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {
            'crf': str(quality),
            'b': '0',
            'row-mt': '1'
        }
    
    def get_quality_label(self) -> str:
        return "Quality (CQ):"
    
    def get_quality_range(self) -> tuple:
        return (0, 63)


# === Sequence Formats ===

class EXRSequenceFormat(ExportFormat):
    @property
    def format_id(self) -> str:
        return "exr"
    
    @property
    def display_name(self) -> str:
        return "EXR Sequence"
    
    @property
    def file_extension(self) -> str:
        return ".####.exr"
    
    @property
    def supports_alpha(self) -> bool:
        return False  # EXR uses separate layers instead
    
    @property
    def supports_quality_setting(self) -> bool:
        return False
    
    @property
    def supports_multiple_export(self) -> bool:
        return False  # EXR exports all objects as layers
    
    @property
    def supports_include_original(self) -> bool:
        return True
    
    @property
    def is_sequence(self) -> bool:
        return True
    
    def get_available_output_types(self) -> List[str]:
        return ['Segmentation-Matte', 'Matting-Matte']
    
    def get_codec_name(self) -> Optional[str]:
        return None
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return ''  # Not used for EXR
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {}


class PNGSequenceFormat(ExportFormat):
    @property
    def format_id(self) -> str:
        return "png"
    
    @property
    def display_name(self) -> str:
        return "PNG Sequence"
    
    @property
    def file_extension(self) -> str:
        return ".####.png"
    
    @property
    def supports_alpha(self) -> bool:
        return True
    
    @property
    def supports_quality_setting(self) -> bool:
        return False
    
    @property
    def supports_multiple_export(self) -> bool:
        return False
    
    @property
    def supports_include_original(self) -> bool:
        return False
    
    @property
    def is_sequence(self) -> bool:
        return True
    
    def get_available_output_types(self) -> List[str]:
        return [
            'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
        ]
    
    def get_codec_name(self) -> Optional[str]:
        return None
    
    def get_pixel_format(self, has_alpha: bool) -> str:
        return ''  # Not used for PNG
    
    def get_codec_options(self, quality: int) -> Dict[str, str]:
        return {}


# === Format Registry ===

class FormatRegistry:
    """Central registry for all export formats"""
    
    _formats = {
        'prores': ProResFormat(),
        'ffv1': FFV1Format(),
        'x264': H264Format(),
        'x265': H265Format(),
        'vp9': VP9Format(),
        'exr': EXRSequenceFormat(),
        'png': PNGSequenceFormat(),
    }
    
    @classmethod
    def get_format(cls, format_id: str) -> ExportFormat:
        """Get format by ID"""
        return cls._formats.get(format_id)
    
    @classmethod
    def get_all_formats(cls) -> List[ExportFormat]:
        """Get all available formats in display order"""
        return [
            cls._formats['prores'],
            cls._formats['ffv1'],
            cls._formats['x264'],
            cls._formats['x265'],
            cls._formats['vp9'],
            cls._formats['exr'],
            cls._formats['png'],
        ]
    
    @classmethod
    def get_format_ids(cls) -> List[str]:
        """Get all format IDs"""
        return list(cls._formats.keys())
