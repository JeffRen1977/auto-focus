"""
Laser Detection Auto Focus (LDAF)
===================================

This module implements Laser Detection Auto Focus, a distance-based focusing
method using laser time-of-flight or structured light measurements.

Class:
- LDAFProcessor: Converts distance measurements to lens positions

Key Concepts:
- Uses laser ranging to measure distance to subject
- Directly converts distance to lens position
- Very fast but limited range and accuracy
- Useful for initial positioning

Learning Order:
Study this after cdaf.py. LDAF is less commonly used than PDAF/CDAF.
"""

import numpy as np
from typing import Dict

from .config import AFConfig


class LDAFProcessor:
    """
    Laser Detection Auto Focus (LDAF) Processor.
    
    LDAF is a distance-based autofocus method that uses laser ranging
    (time-of-flight or structured light) to directly measure the distance
    to the subject. The distance is then converted to a lens position.
    
    How LDAF Works:
    1. Laser emits light pulse or structured light pattern
    2. Sensor measures time-of-flight or pattern distortion
    3. Distance to subject is calculated
    4. Distance is converted to lens position using calibration table
    5. Lens is moved to calculated position
    
    Advantages:
    - Very fast (single measurement, no iterative search)
    - Direct distance measurement (no ambiguity)
    - Good for initial positioning
    - Works well for close subjects
    
    Limitations:
    - Limited range (typically 0.1-10 meters)
    - Requires line-of-sight to subject
    - Less accurate than PDAF/CDAF
    - Additional hardware cost (laser emitter/sensor)
    - May fail with transparent or reflective surfaces
    
    Usage:
    LDAF is typically used as a supplementary method in hybrid AF systems,
    providing fast initial positioning before fine-tuning with PDAF or CDAF.
    
    Comparison with Other Methods:
    - PDAF: Fast, requires texture, hardware-assisted
    - CDAF: Accurate, slow, iterative search
    - LDAF: Very fast, limited range, direct measurement
    """
    
    def __init__(self, config: AFConfig):
        """
        Initialize LDAF processor with configuration.
        
        Args:
            config: AFConfig object containing LDAF parameters
                - ldaf_confidence_threshold: Minimum confidence (default: 0.8)
                - ldaf_distance_range: Valid distance range in meters (default: (0.1, 10.0))
        
        Distance-to-Position Table:
        The processor maintains a lookup table that maps distance measurements
        to lens positions. This table is created based on lens calibration
        data. In production, this would be loaded from device-specific
        calibration files.
        """
        self.config = config
        
        # Distance to lens position lookup table (would be calibrated)
        # In production, this would be loaded from device calibration data
        self.distance_to_position_table = self._create_distance_table()
    
    def process_ldaf(self, distance: float, confidence: float) -> Dict:
        """
        Process LDAF distance measurement and convert to lens position.
        
        This method validates the distance measurement and converts it to
        a suggested lens position using the calibration lookup table.
        
        Args:
            distance: Measured distance in meters (float)
                     - Typical range: 0.1-10.0 meters
                     - Must be within ldaf_distance_range
            
            confidence: Measurement confidence (float, 0.0-1.0)
                       - Higher = more reliable measurement
                       - Must be >= ldaf_confidence_threshold to be used
                       - Based on signal quality and measurement consistency
        
        Returns:
            Dictionary containing:
            - 'suggested_position': Suggested lens position (float or None)
              - None if measurement is invalid
              - Typical range: 0-1023 (10-bit control)
            - 'confidence': Confidence score (float, 0.0-1.0)
              - 0.0 if measurement invalid
              - Input confidence if valid
            - 'distance': Input distance measurement (float)
            - 'reason': Reason for result (str)
              - 'success': Valid measurement, position calculated
              - 'low_confidence': Confidence below threshold
              - 'out_of_range': Distance outside valid range
        
        Validation:
        The measurement is validated against:
        1. Confidence threshold (must be >= ldaf_confidence_threshold)
        2. Distance range (must be within ldaf_distance_range)
        
        If validation fails, suggested_position is None and confidence is 0.0.
        
        Example:
            processor = LDAFProcessor(config)
            
            # Valid measurement
            result = processor.process_ldaf(distance=2.5, confidence=0.9)
            # Result: suggested_position=400, confidence=0.9, reason='success'
            
            # Low confidence (rejected)
            result = processor.process_ldaf(distance=2.5, confidence=0.5)
            # Result: suggested_position=None, confidence=0.0, reason='low_confidence'
            
            # Out of range (rejected)
            result = processor.process_ldaf(distance=20.0, confidence=0.9)
            # Result: suggested_position=None, confidence=0.0, reason='out_of_range'
        """
        # Check confidence threshold
        if confidence < self.config.ldaf_confidence_threshold:
            return {
                'suggested_position': None,
                'confidence': 0.0,
                'distance': distance,
                'reason': 'low_confidence'
            }
        
        # Check distance range
        min_dist, max_dist = self.config.ldaf_distance_range
        if not (min_dist <= distance <= max_dist):
            return {
                'suggested_position': None,
                'confidence': 0.0,
                'distance': distance,
                'reason': 'out_of_range'
            }
        
        # Convert distance to lens position using lookup table
        suggested_position = self._distance_to_lens_position(distance)
        
        return {
            'suggested_position': suggested_position,
            'confidence': confidence,
            'distance': distance,
            'reason': 'success'
        }
    
    def _create_distance_table(self) -> Dict[float, float]:
        """
        Create distance to lens position lookup table.
        
        This method generates a calibration table that maps distance
        measurements to lens positions. In production, this table would
        be loaded from device-specific calibration data obtained through
        lens characterization tests.
        
        Returns:
            Dictionary mapping distance (meters) to lens position (0-1023)
            - Keys: Distance values rounded to 0.1m precision
            - Values: Corresponding lens positions
        
        Calibration:
        The relationship between distance and lens position depends on:
        - Lens focal length
        - Sensor size
        - Lens movement mechanism
        - Object distance range
        
        This is a simplified implementation. Real calibration tables:
        - Are more complex (non-linear relationships)
        - Include temperature compensation
        - Account for lens manufacturing variations
        - Are specific to each lens-sensor combination
        
        The simplified relationship used here:
        position ≈ (1000 / (distance + 0.1)) * 100
        
        This gives:
        - Closer objects → higher position values
        - Farther objects → lower position values
        - Clamped to [0, 1023] range
        """
        table = {}
        # Create entries for distances from 0.1m to 10.0m in 0.1m steps
        for d in np.arange(0.1, 10.1, 0.1):
            # Simplified relationship: closer = higher position value
            # Real calibration would use measured lens characteristics
            position = 1000.0 / (d + 0.1) * 100
            # Clamp to valid range [0, 1023]
            position = min(1023, max(0, position))
            table[round(d, 1)] = position
        return table
    
    def _distance_to_lens_position(self, distance: float) -> float:
        """
        Convert distance to lens position using lookup table.
        
        This method finds the closest matching distance in the lookup table
        and returns the corresponding lens position.
        
        Args:
            distance: Distance in meters (float)
                     - Should be within valid range
        
        Returns:
            Lens position (float, 0-1023)
            - Corresponds to closest matching distance in table
        
        Algorithm:
        Uses nearest neighbor lookup - finds the table entry with
        minimum absolute difference from input distance.
        
        For production systems, more sophisticated interpolation could be used:
        - Linear interpolation between table entries
        - Cubic spline interpolation
        - Polynomial fitting
        """
        # Find closest distance in table
        distances = list(self.distance_to_position_table.keys())
        closest_distance = min(distances, key=lambda x: abs(x - distance))
        return self.distance_to_position_table[closest_distance]







