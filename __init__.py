from .nodes import Contrast_Brightness
from .nodes import Levels_Adjustment
from .nodes import Saturation_Vibrance
from .nodes import Tint
from .nodes import Noise
from .nodes import Bloom
from .nodes import Vignette_Effect
from .nodes import Chromatic_Aberration
from .nodes import Lens_Distortion
from .nodes import Depth_of_Field
from .nodes import Sensor_Dust
from .nodes import Lens_Dirt
from .nodes import Physically_Accurate_Lens_Dirt
from .nodes import Bloom_Lens_Flares
from .nodes import Halation
from .nodes import Sharpen_Simple
from .nodes import Sharpen_Unsharp_Mask
from .nodes import Manga_Toner
from .nodes import Monitor_Filter
from .nodes import VHS_Degrade
from .nodes import Watermark
from .nodes import Get_Watermark
from .nodes import Multi_Scale_Contrast
from .nodes import Contrast_Adaptive_Sharpening
from .nodes import VHS_Chroma_Smear
from .nodes import NTSC_Filter
from .nodes import Pixel_Art
from .nodes import Lut

NODE_CLASS_MAPPINGS = {
    "Contrast Brightness": Contrast_Brightness,
    "Levels Adjustment": Levels_Adjustment,
    "Saturation Vibrance": Saturation_Vibrance,
    "Tint": Tint,
    "Noise": Noise,
    "Bloom": Bloom,
    "Vignette Effect": Vignette_Effect,
    "Chromatic Aberration": Chromatic_Aberration,
    "Lens Distortion": Lens_Distortion,
    "Depth of Field": Depth_of_Field,
    "Sensor Dust": Sensor_Dust,
    "Lens Dirt": Lens_Dirt,
    "Physically Accurate Lens Dirt": Physically_Accurate_Lens_Dirt,
    "Bloom Lens Flares": Bloom_Lens_Flares,
    "Halation": Halation,
    "Sharpen Simple": Sharpen_Simple,
    "Sharpen Unsharp Mask": Sharpen_Unsharp_Mask,
    "Manga Toner": Manga_Toner,
    "Monitor Filter": Monitor_Filter,
    "VHS Degrade": VHS_Degrade,
    "Watermark": Watermark,
    "Get Watermark": Get_Watermark,
    "Multi Scale Contrast": Multi_Scale_Contrast,
    "Contrast Adaptive Sharpening": Contrast_Adaptive_Sharpening,
    "VHS Chroma Smear": VHS_Chroma_Smear,
    "NTSC Filter": NTSC_Filter,
    "Pixel Art": Pixel_Art,
    "Lut": Lut
}

NODE_DISPLAY_NAMES_MAPPINGS = {
    "Contrast Brightness": "Simple Contrast and Brightness adjustments",
    "Levels Adjustment": "Simple Levels adjustments",
    "Saturation Vibrance": "Simple Saturation and Vibrance adjustments",
    "Tint": "Simple Tint adjustments",
    "Noise": "Simple Noise adjustments",
    "Bloom": "Simple Bloom effects",
    "Vignette Effect": "Simple Vignette effects",
    "Chromatic Aberration": "Chromatic Aberration effects",
    "Lens Distortion": "Lens Distortion effects",
    "Depth of Field": "Depth of Field Effects",
    "Sensor Dust": "Sensor Dust effects",
    "Lens Dirt": "Lens Dirt effects",
    "Physically Accurate Lens Dirt": "Physically Accurate Lens Dirt Effects",
    "Bloom Lens Flares": "Bloom Lens Flares Effects",
    "Halation": "Halation Effects",
    "Sharpen Simple": "Sharpen effects",
    "Sharpen Unsharp Mask": "Sharpen Unsharp Mask Effects",
    "Manga Toner": "Manga Toner Effects",
    "Monitor Filter": "Monitor Filter Effect",
    "VHS Degrade": "VHS Degrade effects",
    "Watermark": "Watermark Effects",
    "Get Watermark": "Get Watermark from image",
    "Multi Scale Contrast": "Multi Scale Contrast Effects",
    "Contrast Adaptive Sharpening": "Contrast Adaptive Sharpening Effect",
    "VHS Chroma Smear": "VHS Chroma Smear Effect",
    "NTSC Filter": "NTSC Filter Effect",
    "Pixel Art": "Pixel Art Effects",
    "Lut": "Lut Effects"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']