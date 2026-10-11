"""
Scale calibration and ultrasound probe depth metrology module.
Automatically detects pixel-to-millimeter ratio from depth tick marks
and ultrasound image borders to avoid machine-specific scale drift.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import cv2
from scipy.signal import find_peaks


class UltrasoundScaleDetector:
    """
    Detects scale ticks along lateral borders of B-mode ultrasound frames.
    Calculates dynamic pixel-to-millimeter (px_to_mm) and millimeter-to-pixel conversion ratios.
    """
    
    def __init__(
        self,
        default_tick_interval_mm: float = 10.0, # Most linear probes place ticks every 10 mm (or 5 mm)
        default_px_per_mm: float = 10.0, # Baseline fallback: 50mm depth across 500px height
        margin_pct: float = 0.08, # Look within lateral 8% margins
    ):
        self.default_tick_interval_mm = default_tick_interval_mm
        self.default_px_per_mm = default_px_per_mm
        self.margin_pct = margin_pct

    def detect_scale_from_image(
        self,
        image: np.ndarray,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Analyzes image margins for periodic tick marks.
        
        Args:
            image: Grayscale or BGR ultrasound image.
            return_debug: Whether to return peak detection debug arrays.
            
        Returns:
            Dictionary with:
                - px_per_mm: float (pixels per mm)
                - mm_per_px: float (mm per pixel)
                - is_calibrated: bool (True if periodic ticks detected with high confidence)
                - tick_period_px: float
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape
        margin_w = max(10, int(w * self.margin_pct))
        
        # Check right lateral column margin (standard for most ultrasound machines like GE/Siemens)
        right_margin = gray[:, -margin_w:]
        left_margin = gray[:, :margin_w]
        
        # Compute horizontal projection profile (sum along rows)
        right_proj = np.mean(right_margin, axis=1)
        left_proj = np.mean(left_margin, axis=1)
        
        # Test which margin has stronger periodic tick variance
        use_right = np.std(right_proj) >= np.std(left_proj)
        signal = right_proj if use_right else left_proj
        
        # Normalize signal and compute first derivative (edge peaks)
        grad = np.abs(np.diff(signal))
        if np.max(grad) > 0:
            grad = grad / np.max(grad)
            
        # Detect peaks corresponding to tick marks
        # Ticks are separated by at least 15 pixels in typical clinical resolutions
        peaks, properties = find_peaks(grad, height=0.25, distance=15)
        
        if len(peaks) >= 3:
            # Measure inter-peak distances
            diffs = np.diff(peaks)
            median_diff = float(np.median(diffs))
            std_diff = float(np.std(diffs))
            
            # If standard deviation is small, spacing is regular and periodic
            if std_diff / max(1.0, median_diff) < 0.25 and median_diff > 10:
                px_per_mm = median_diff / self.default_tick_interval_mm
                return {
                    "px_per_mm": px_per_mm,
                    "mm_per_px": 1.0 / px_per_mm,
                    "is_calibrated": True,
                    "tick_period_px": median_diff,
                    "confidence": 1.0 - (std_diff / median_diff),
                }
                
        # Fallback to default estimated calibration
        return {
            "px_per_mm": self.default_px_per_mm,
            "mm_per_px": 1.0 / self.default_px_per_mm,
            "is_calibrated": False,
            "tick_period_px": self.default_px_per_mm * self.default_tick_interval_mm,
            "confidence": 0.0,
        }

    def convert_pixels_to_mm(self, pixels: float, scale_info: Dict[str, Any]) -> float:
        """Converts pixel measurement to physical millimeters."""
        return float(pixels * scale_info.get("mm_per_px", 1.0 / self.default_px_per_mm))

    def convert_mm_to_pixels(self, mm: float, scale_info: Dict[str, Any]) -> float:
        """Converts millimeters to pixel coordinates."""
        return float(mm * scale_info.get("px_per_mm", self.default_px_per_mm))
