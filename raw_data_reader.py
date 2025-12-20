"""
Raw Data Reader for Autofocus System
====================================

This module provides functions to read and process raw sensor data in RAW10 format:
1. PD (Phase Detection) RAW10 images - for PDAF calculations
2. Bayer RAW10 images - for CDAF calculations

Key Functions:
- unpack_raw10(): Efficiently unpacks 10-bit packed data to 16-bit arrays
- read_pd_raw(): Reads PD data and splits into left/right views
- read_bayer_raw(): Reads Bayer data for contrast-based focusing
- split_pd_2x1_blocks(): Correctly separates PD pixels using 2x1 column blocks pattern
- demosaic_bayer(): Converts Bayer pattern to RGB (optional, for visualization)
- save_image_8bit(): Saves uint16 images as 8-bit JPEG with proper scaling

RAW10 Format:
RAW10 is a packed 10-bit image format used in Android cameras:
- Each pixel is 10 bits (0-1023 range)
- Data is packed: 4 pixels per 5 bytes
- Format: [P0_high P1_high P2_high P3_high P0_low|P1_low|P2_low|P3_low]
- The 5th byte contains the 2 LSBs of each of the 4 pixels

PD Pattern:
PD sensors use a 2x1 column blocks pattern:
- Pattern: [L L R R L L R R ...]
- Left PD:  columns 0-1, 4-5, 8-9, ... (captures left view)
- Right PD: columns 2-3, 6-7, 10-11, ... (captures right view)
- This creates two views for phase difference calculation

Usage:
    from raw_data_reader import read_pd_raw, read_bayer_raw
    
    left_pd, right_pd = read_pd_raw("data/pd.raw", 4032, 756, 5056)
    bayer_image = read_bayer_raw("data/bayer.raw", 4032, 3024, 5120)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import os


def unpack_raw10(data: bytes, width: int, height: int, stride: int) -> np.ndarray:
    """
    Unpacks RAW10 image data from packed byte format to 16-bit arrays.
    
    RAW10 Format Details:
    RAW10 is a packed 10-bit image format commonly used in Android camera systems.
    The format packs 4 pixels into 5 bytes for efficient storage:
    
    Byte Layout:
    - Bytes 0-3: High 8 bits of pixels 0-3 respectively
    - Byte 4: Contains 2 LSBs of all 4 pixels:
        Bit 0-1:   Pixel 0 LSBs
        Bit 2-3:   Pixel 1 LSBs
        Bit 4-5:   Pixel 2 LSBs
        Bit 6-7:   Pixel 3 LSBs
    
    Unpacking Process:
    1. Reshape data to (height, stride) to account for row padding
    2. Extract the packed width (width * 10 / 8 bytes)
    3. Reshape to groups of 5 bytes (4 pixels per group)
    4. Extract high 8 bits from bytes 0-3
    5. Extract low 2 bits from byte 4
    6. Combine: pixel_value = (high_8_bits << 2) | low_2_bits
    
    Reference: https://developer.android.com/reference/android/graphics/ImageFormat#RAW10

    Args:
        data: Packed byte image data from RAW10 file
        width: Image width in pixels (e.g., 4032)
        height: Image height in pixels (e.g., 756)
        stride: ISP byte stride (row width in bytes, including padding)
               This accounts for memory alignment requirements
               Example: For 4032 pixels at 10 bits = 5040 bytes, stride might be 5056

    Returns:
        Unpacked uint16 numpy array, shape (height, width)
        Pixel values range from 0-1023 (10-bit range), stored as uint16

    Raises:
        ValueError: If data is insufficient (less than 5 bytes or less than stride*height)
        
    Algorithm Complexity:
        - Time: O(height * width) - linear in number of pixels
        - Space: O(height * width) - creates output array
        
    Example:
        with open("image.raw", 'rb') as f:
            data = f.read()
        image = unpack_raw10(data, width=4032, height=3024, stride=5120)
        # image.shape = (3024, 4032), dtype = uint16
    """
    # Validate input data size
    if len(data) < 5:
        raise ValueError("Packed data must be at least 5 bytes.")
    if len(data) < stride * height:
        raise ValueError(
            f"len(data): {len(data)}, stride: {stride}, height: {height}. "
            "len(data) must be greater than or equal to stride * height"
        )
    
    # Calculate unpacked width: 4 pixels per 5 bytes, so width*10/8 bytes needed
    unpacked_width = width * 10 // 8
    
    # Step 1: Convert bytes to numpy array and reshape to (height, stride)
    # This accounts for row padding (stride may be larger than unpacked_width)
    # Step 2: Extract only the packed width (ignore padding)
    # Step 3: Reshape to groups of 5 bytes (each group = 4 pixels)
    packed_img = (
        np.frombuffer(data, dtype=np.uint8)      # Convert bytes to uint8 array
        .astype(np.uint16)                        # Convert to uint16 for bit operations
        .reshape(height, stride)[:, :unpacked_width]  # Extract packed width, ignore padding
        .reshape(unpacked_width * height // 5, 5)     # Group into 5-byte chunks
    )
    
    # Step 4: Create output array for 4 pixels per group
    unpacked_img = np.zeros((packed_img.shape[0], 4), dtype=np.uint16)
    
    # Step 5: Extract the 5th byte which contains LSBs of all 4 pixels
    fifth_byte_data = packed_img[:, 4]
    
    # Step 6: Unpack each pixel by combining high 8 bits with low 2 bits
    # Pixel 0: high_8_bits << 2 | (fifth_byte & 0b00000011)
    unpacked_img[:, 0] = (packed_img[:, 0] << 2) | (fifth_byte_data & 0x0003)
    # Pixel 1: high_8_bits << 2 | ((fifth_byte >> 2) & 0b00000011)
    unpacked_img[:, 1] = (packed_img[:, 1] << 2) | (fifth_byte_data >> 2 & 0x0003)
    # Pixel 2: high_8_bits << 2 | ((fifth_byte >> 4) & 0b00000011)
    unpacked_img[:, 2] = (packed_img[:, 2] << 2) | (fifth_byte_data >> 4 & 0x0003)
    # Pixel 3: high_8_bits << 2 | ((fifth_byte >> 6) & 0b00000011)
    unpacked_img[:, 3] = (packed_img[:, 3] << 2) | (fifth_byte_data >> 6 & 0x0003)
    
    # Step 7: Reshape to final image dimensions (height, width)
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
    Split PD image into left/right views using 2x1 column blocks pattern.
    
    PD Sensor Pattern:
    Phase Detection sensors use a specific pixel arrangement where alternating
    column pairs capture light from different angles (left and right views).
    This creates two views that can be used to calculate phase difference
    (disparity) for fast autofocus.
    
    Pattern Details:
    The sensor uses a repeating 4-column pattern: [L L R R L L R R ...]
    - Left PD pixels:  columns 0-1, 4-5, 8-9, 12-13, ... (capture left view)
    - Right PD pixels: columns 2-3, 6-7, 10-11, 14-15, ... (capture right view)
    
    Each pair of columns (2x1 block) captures the same view, providing:
    - Redundancy for noise reduction
    - Better signal-to-noise ratio
    - More reliable disparity calculation
    
    Physical Implementation:
    The PD pixels are typically implemented using:
    - Split photodiodes or
    - Microlens arrays that direct light from different angles
    This creates the left/right view separation needed for phase detection.

    Args:
        pd_image: Input PD image as uint16 numpy array, shape (height, width)
                  This is the unpacked RAW10 PD data

    Returns:
        Tuple of (left_pd, right_pd) as uint16 numpy arrays:
        - left_pd:  Left view PD image, shape (height, width/2)
        - right_pd: Right view PD image, shape (height, width/2)
        
        Both images have the same height but half the width of the original.

    Algorithm:
        1. Create boolean masks for left and right columns
        2. Apply pattern: every 4 columns, first 2 are left, next 2 are right
        3. Extract columns using masks
        4. Return separated left and right images
        
    Example:
        pd_image shape: (756, 4032)
        left_pd shape:  (756, 2016)
        right_pd shape: (756, 2016)
        
    Note:
        This pattern is sensor-specific. Some sensors may use different patterns
        (e.g., 1x1, 2x2). This implementation assumes 2x1 column blocks.
    """
    height, width = pd_image.shape
    
    # Create boolean masks to identify left and right columns
    # These masks will be used to extract the appropriate columns
    left_mask = np.zeros(width, dtype=bool)
    right_mask = np.zeros(width, dtype=bool)
    
    # Apply 2x1 column blocks pattern: [L L R R L L R R ...]
    # Process in groups of 4 columns
    for i in range(0, width, 4):
        # First 2 columns in each group are left PD pixels
        if i + 1 < width:
            left_mask[i:i+2] = True  # Mark columns i and i+1 as left
        
        # Next 2 columns in each group are right PD pixels
        if i + 3 < width:
            right_mask[i+2:i+4] = True  # Mark columns i+2 and i+3 as right
    
    # Extract left and right images using boolean indexing
    # This efficiently selects only the columns marked as left/right
    left_pd = pd_image[:, left_mask]   # Extract left columns
    right_pd = pd_image[:, right_mask] # Extract right columns
    
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
    Save uint16 image as 8-bit JPEG with proper contrast scaling.
    
    This function converts high-bit-depth images (uint16) to 8-bit (uint8) for
    visualization and storage. The key is proper normalization to preserve
    contrast and avoid dark/washed-out images.
    
    Scaling Methods:
    1. use_actual_range=True (recommended):
       - Uses the actual min/max values in the image
       - Formula: (pixel - min) / (max - min) * 255
       - Preserves full contrast range of the actual data
       - Best for visualization and debugging
       
    2. use_actual_range=False:
       - Uses full 16-bit range (0-65535)
       - Formula: pixel / 65535 * 255
       - May result in dark images if data doesn't use full range
       - Useful when you want consistent scaling across images

    Args:
        image: Input image as uint16 numpy array, shape (height, width)
               or (height, width, channels) for multi-channel images
        output_path: Output file path (e.g., "data/output.jpg")
        use_actual_range: Scaling method:
            - True: Normalize using actual data range (better contrast)
            - False: Normalize using full 16-bit range (consistent scaling)
            
    Returns:
        None (saves file to disk)
        
    Raises:
        ValueError: If image cannot be converted to uint16
        
    Example:
        # Save with actual range (recommended for better visibility)
        save_image_8bit(pd_image, "pd_image.jpg", use_actual_range=True)
        
        # Save with full range (for consistent scaling)
        save_image_8bit(pd_image, "pd_image.jpg", use_actual_range=False)
        
    Note:
        JPEG quality is set to 95 (high quality). For smaller files, you could
        modify the cv2.imwrite call to use lower quality (0-100).
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
