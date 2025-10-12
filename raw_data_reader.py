"""
Raw Data Reader for Autofocus System

This module provides functions to read and process:
1. PD (Phase Detection) RAW10 images
2. Bayer RAW10 images

Both are used as input for PDAF and CDAF calculations.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import os


def unpack_raw10(data: bytes, width: int, height: int, stride: int) -> np.ndarray:
    """
    Unpacks raw10 image data.

    Reference: https://developer.android.com/reference/android/graphics/ImageFormat#RAW10

    Args:
        data: packed byte image data
        width: image width
        height: image height
        stride: ISP byte stride

    Returns:
        Unpacked uint16 numpy image

    Raises:
        ValueError: if data is insufficient
    """
    if len(data) < 5:
        raise ValueError("Packed data must be at least 5 bytes.")
    if len(data) < stride * height:
        raise ValueError(
            f"len(data): {len(data)}, stride: {stride}, height: {height}. "
            "len(data) must be greater than or equal to stride * height"
        )
    
    unpacked_width = width * 10 // 8
    packed_img = (
        np.frombuffer(data, dtype=np.uint8)
        .astype(np.uint16)
        .reshape(height, stride)[:, :unpacked_width]
        .reshape(unpacked_width * height // 5, 5)
    )
    
    unpacked_img = np.zeros((packed_img.shape[0], 4), dtype=np.uint16)
    fifth_byte_data = packed_img[:, 4]
    unpacked_img[:, 0] = (packed_img[:, 0] << 2) | (fifth_byte_data & 0x0003)
    unpacked_img[:, 1] = (packed_img[:, 1] << 2) | (fifth_byte_data >> 2 & 0x0003)
    unpacked_img[:, 2] = (packed_img[:, 2] << 2) | (fifth_byte_data >> 4 & 0x0003)
    unpacked_img[:, 3] = (packed_img[:, 3] << 2) | (fifth_byte_data >> 6 & 0x0003)
    
    return unpacked_img.reshape(height, width)


def read_pd_raw(file_path: str, width: int, height: int, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read PD RAW10 file and split into left/right PD images.

    Args:
        file_path: Path to PD RAW10 file
        width: Image width
        height: Image height  
        stride: Row stride in bytes

    Returns:
        Tuple of (left_pd, right_pd) as uint16 numpy arrays

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if file size doesn't match expected size
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PD raw file not found: {file_path}")
    
    # Read raw data
    with open(file_path, 'rb') as f:
        data = f.read()
    
    expected_size = stride * height
    if len(data) != expected_size:
        raise ValueError(f"File size mismatch. Expected {expected_size}, got {len(data)}")
    
    print(f"Reading PD raw: {file_path.name}")
    print(f"  Dimensions: {width}x{height}, stride: {stride}")
    print(f"  File size: {len(data)} bytes")
    
    # Unpack RAW10 data
    pd_image = unpack_raw10(data, width, height, stride)
    print(f"  Unpacked range: [{pd_image.min()}, {pd_image.max()}], mean: {pd_image.mean():.2f}")
    
    # Split into left/right using 2x1 column blocks pattern
    left_pd, right_pd = split_pd_2x1_blocks(pd_image)
    
    print(f"  Left PD:  range [{left_pd.min()}, {left_pd.max()}], mean: {left_pd.mean():.2f}")
    print(f"  Right PD: range [{right_pd.min()}, {right_pd.max()}], mean: {right_pd.mean():.2f}")
    
    return left_pd, right_pd


def split_pd_2x1_blocks(pd_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split PD image into left/right using 2x1 column blocks pattern.
    
    Pattern: [L L R R L L R R ...]
    - Left:  columns 0-1, 4-5, 8-9, ...
    - Right: columns 2-3, 6-7, 10-11, ...

    Args:
        pd_image: Input PD image (height, width)

    Returns:
        Tuple of (left_pd, right_pd) with same height, width/2
    """
    height, width = pd_image.shape
    
    # Create masks for left and right columns
    left_mask = np.zeros(width, dtype=bool)
    right_mask = np.zeros(width, dtype=bool)
    
    # 2x1 column blocks pattern: L L R R L L R R ...
    for i in range(0, width, 4):
        if i + 1 < width:
            left_mask[i:i+2] = True  # First 2 columns
        if i + 3 < width:
            right_mask[i+2:i+4] = True  # Next 2 columns
    
    # Extract left and right images
    left_pd = pd_image[:, left_mask]
    right_pd = pd_image[:, right_mask]
    
    return left_pd, right_pd


def read_bayer_raw(file_path: str, width: int, height: int, stride: int) -> np.ndarray:
    """
    Read Bayer RAW10 file.

    Args:
        file_path: Path to Bayer RAW10 file
        width: Image width
        height: Image height
        stride: Row stride in bytes

    Returns:
        Bayer image as uint16 numpy array

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if file size doesn't match expected size
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Bayer raw file not found: {file_path}")
    
    # Read raw data
    with open(file_path, 'rb') as f:
        data = f.read()
    
    expected_size = stride * height
    if len(data) != expected_size:
        raise ValueError(f"File size mismatch. Expected {expected_size}, got {len(data)}")
    
    print(f"Reading Bayer raw: {file_path.name}")
    print(f"  Dimensions: {width}x{height}, stride: {stride}")
    print(f"  File size: {len(data)} bytes")
    
    # Unpack RAW10 data
    bayer_image = unpack_raw10(data, width, height, stride)
    print(f"  Unpacked range: [{bayer_image.min()}, {bayer_image.max()}], mean: {bayer_image.mean():.2f}")
    
    return bayer_image


def demosaic_bayer(bayer_image: np.ndarray, pattern: str = 'BGGR') -> np.ndarray:
    """
    Demosaic Bayer image to RGB.

    Args:
        bayer_image: Input Bayer image (height, width)
        pattern: Bayer pattern ('BGGR', 'RGGB', 'GBRG', 'GRBG')

    Returns:
        RGB image (height, width, 3) as uint16
    """
    # OpenCV expects BGR order, so we need to map the pattern
    pattern_map = {
        'BGGR': cv2.COLOR_BayerBG2BGR,
        'RGGB': cv2.COLOR_BayerRG2BGR, 
        'GBRG': cv2.COLOR_BayerGB2BGR,
        'GRBG': cv2.COLOR_BayerGR2BGR
    }
    
    if pattern not in pattern_map:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    
    # Convert to 8-bit for demosaicing (OpenCV limitation)
    bayer_8bit = ((bayer_image.astype(np.float32) / 65535.0) * 255.0).astype(np.uint8)
    
    # Demosaic
    rgb_8bit = cv2.cvtColor(bayer_8bit, pattern_map[pattern])
    
    # Convert back to 16-bit
    rgb_16bit = (rgb_8bit.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    
    print(f"  Demosaiced to RGB: {rgb_16bit.shape}")
    
    return rgb_16bit


def save_image_8bit(image: np.ndarray, output_path: str, use_actual_range: bool = True) -> None:
    """
    Save uint16 image as 8-bit JPG with proper scaling.

    Args:
        image: Input image (uint16)
        output_path: Output file path
        use_actual_range: If True, use actual data range; if False, use full 16-bit range
    """
    if image.dtype != np.uint16:
        image = image.astype(np.uint16)
    
    if use_actual_range:
        # Use actual data range for better contrast
        img_min = image.min()
        img_max = image.max()
        if img_max > img_min:
            scaled = ((image.astype(np.float32) - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            scaled = np.zeros_like(image, dtype=np.uint8)
    else:
        # Use full 16-bit range
        scaled = (image.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
    
    cv2.imwrite(output_path, scaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Saved: {output_path}")


# Example usage function
def process_raw_data_example():
    """Example of how to use the raw data readers"""
    
    # PD data parameters
    pd_file = "data/20250929_223233_713_RAW10_4032x756_RS_5056_PS_1_cam_3_sensor_raw_pd_frame__default_pd_type_622.raw"
    pd_width, pd_height, pd_stride = 4032, 756, 5056
    
    # Bayer data parameters  
    bayer_file = "data/20250929_223233_713_RAW10_4032x3024_RS_5120_PS_1_cam_3_sensor_raw_output_frame_622.raw"
    bayer_width, bayer_height, bayer_stride = 4032, 3024, 5120
    
    try:
        # Read PD data
        print("=== Reading PD Data ===")
        left_pd, right_pd = read_pd_raw(pd_file, pd_width, pd_height, pd_stride)
        
        # Read Bayer data
        print("\n=== Reading Bayer Data ===")
        bayer_image = read_bayer_raw(bayer_file, bayer_width, bayer_height, bayer_stride)
        
        # Demosaic Bayer to RGB
        print("\n=== Demosaicing Bayer ===")
        rgb_image = demosaic_bayer(bayer_image, pattern='BGGR')
        
        # Save images for verification
        print("\n=== Saving Images ===")
        save_image_8bit(left_pd, "data/pd_left.jpg")
        save_image_8bit(right_pd, "data/pd_right.jpg") 
        save_image_8bit(rgb_image[:,:,0], "data/bayer_red.jpg")  # Red channel
        save_image_8bit(rgb_image[:,:,1], "data/bayer_green.jpg")  # Green channel
        save_image_8bit(rgb_image[:,:,2], "data/bayer_blue.jpg")  # Blue channel
        
        print("\n✅ Raw data processing complete!")
        print("Ready for autofocus calculations...")
        
        return left_pd, right_pd, bayer_image, rgb_image
        
    except Exception as e:
        print(f"❌ Error processing raw data: {e}")
        return None, None, None, None


if __name__ == "__main__":
    process_raw_data_example()
