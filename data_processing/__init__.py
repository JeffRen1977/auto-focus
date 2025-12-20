"""
Data Processing Module
======================

This package contains utilities for reading and processing raw sensor data.

Modules:
- raw_reader: RAW10 format reading and PD image splitting

This is a foundational module - understanding raw data formats is essential
before learning autofocus algorithms.
"""

from .raw_reader import (
    unpack_raw10,
    split_pd_2x1_blocks,
    read_pd_raw,
    read_bayer_raw,
    demosaic_bayer,
    save_image_8bit
)

__all__ = [
    'unpack_raw10',
    'split_pd_2x1_blocks',
    'read_pd_raw',
    'read_bayer_raw',
    'demosaic_bayer',
    'save_image_8bit',
]

