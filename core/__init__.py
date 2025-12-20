"""
Core Autofocus System Components
=================================

This package contains the core components of the autofocus system, organized
by functionality for easy understanding and learning.

Modules:
- config: Configuration classes and enumerations (AFMethod, AFState, AFConfig, etc.)
- roi_selection: Region of Interest selection for focusing
- pdaf: Phase Detection Auto Focus algorithm
- cdaf: Contrast Detection Auto Focus algorithm
- ldaf: Laser Detection Auto Focus algorithm
- autofocus_system: Main system integration and controller

For book chapter learning progression:
1. Start with config.py to understand system parameters
2. Learn roi_selection.py for understanding focus regions
3. Study pdaf.py for fast hardware-assisted focusing
4. Study cdaf.py for accurate contrast-based focusing
5. Finally, see autofocus_system.py for complete integration
"""

from .config import (
    AFMethod,
    AFState,
    AFConfig,
    ROIData,
    AFResult
)

from .roi_selection import ROISelector

from .pdaf import PDAFProcessor

from .cdaf import CDAFProcessor

from .ldaf import LDAFProcessor

from .autofocus_system import AutoFocusSystem

# Test utilities (optional, for examples and testing)
from .test_utils import create_test_image, create_test_pd_data

__all__ = [
    'AFMethod',
    'AFState',
    'AFConfig',
    'ROIData',
    'AFResult',
    'ROISelector',
    'PDAFProcessor',
    'CDAFProcessor',
    'LDAFProcessor',
    'AutoFocusSystem',
    # Test utilities
    'create_test_image',
    'create_test_pd_data',
]

