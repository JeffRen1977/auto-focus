"""
Autofocus Visualization Tools
=============================

This module provides visualization tools to help understand autofocus concepts,
including disparity maps, contrast curves, and AF convergence behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Dict, Optional
from autofocus_system import create_test_image, create_test_pd_data, PDAFProcessor, CDAFProcessor, AFConfig


def visualize_pdaf_disparity(left_pd: np.ndarray, right_pd: np.ndarray, 
                           save_path: Optional[str] = None):
    """
    Visualize PDAF disparity calculation process
    
    Args:
        left_pd: Left phase detection pixels
        right_pd: Right phase detection pixels  
        save_path: Optional path to save the visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PDAF Disparity Visualization', fontsize=16)
    
    # Show left and right PD images
    axes[0, 0].imshow(left_pd, cmap='gray')
    axes[0, 0].set_title('Left PD Pixels')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_pd, cmap='gray')
    axes[0, 1].set_title('Right PD Pixels')
    axes[0, 1].axis('off')
    
    # Show difference
    difference = np.abs(left_pd.astype(np.float32) - right_pd.astype(np.float32))
    axes[0, 2].imshow(difference, cmap='hot')
    axes[0, 2].set_title('Absolute Difference')
    axes[0, 2].axis('off')
    
    # Calculate disparity map using block matching
    config = AFConfig(pdaf_block_size=16, pdaf_search_range=20)
    pdaf_processor = PDAFProcessor(config)
    
    # Create a dummy ROI for the center region
    h, w = left_pd.shape
    roi_size = min(200, min(h, w) // 2)
    roi_x = (w - roi_size) // 2
    roi_y = (h - roi_size) // 2
    
    from autofocus_system import ROIData
    roi = ROIData(roi_x, roi_y, roi_size, roi_size, 1.0, 'center')
    
    # Process PDAF
    result = pdaf_processor.process_pdaf(left_pd, right_pd, roi)
    
    # Show disparity map (simplified visualization)
    disparity_map = np.zeros_like(left_pd)
    
    # Calculate block-wise disparities for visualization
    block_size = 16
    search_range = 20
    
    for y in range(0, roi_size - block_size, block_size):
        for x in range(0, roi_size - block_size, block_size):
            left_block = left_pd[roi_y + y:roi_y + y + block_size, 
                               roi_x + x:roi_x + x + block_size]
            
            best_cost = float('inf')
            best_disparity = 0
            
            for d in range(-search_range, search_range + 1):
                x_search = roi_x + x + d
                if x_search < 0 or x_search + block_size > w:
                    continue
                
                right_block = right_pd[roi_y + y:roi_y + y + block_size,
                                     x_search:x_search + block_size]
                
                cost = np.sum(np.abs(left_block.astype(np.float32) - 
                                   right_block.astype(np.float32)))
                
                if cost < best_cost:
                    best_cost = cost
                    best_disparity = d
            
            # Fill disparity map
            disparity_map[roi_y + y:roi_y + y + block_size,
                         roi_x + x:roi_x + x + block_size] = best_disparity
    
    # Show disparity map
    im1 = axes[1, 0].imshow(disparity_map, cmap='RdBu_r', vmin=-20, vmax=20)
    axes[1, 0].set_title(f'Disparity Map\n(Mean: {result["disparity"]:.2f} px)')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Show cost curve for center block
    center_y, center_x = roi_size // 2, roi_size // 2
    left_block = left_pd[roi_y + center_y:roi_y + center_y + block_size,
                        roi_x + center_x:roi_x + center_x + block_size]
    
    costs = []
    disparities = list(range(-search_range, search_range + 1))
    
    for d in disparities:
        x_search = roi_x + center_x + d
        if x_search < 0 or x_search + block_size > w:
            costs.append(np.inf)
            continue
        
        right_block = right_pd[roi_y + center_y:roi_y + center_y + block_size,
                             x_search:x_search + block_size]
        
        cost = np.sum(np.abs(left_block.astype(np.float32) - 
                           right_block.astype(np.float32)))
        costs.append(cost)
    
    costs = np.array(costs)
    valid_mask = ~np.isinf(costs)
    
    axes[1, 1].plot(np.array(disparities)[valid_mask], costs[valid_mask], 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Disparity (pixels)')
    axes[1, 1].set_ylabel('SAD Cost')
    axes[1, 1].set_title('Cost Curve (Center Block)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Mark minimum
    min_idx = np.argmin(costs[valid_mask])
    min_disp = np.array(disparities)[valid_mask][min_idx]
    min_cost = costs[valid_mask][min_idx]
    axes[1, 1].plot(min_disp, min_cost, 'ro', markersize=8, label=f'Min at {min_disp}')
    axes[1, 1].legend()
    
    # Show AF result
    axes[1, 2].text(0.1, 0.8, f'PDAF Result:', fontsize=14, weight='bold')
    axes[1, 2].text(0.1, 0.7, f'Disparity: {result["disparity"]:.3f} px', fontsize=12)
    axes[1, 2].text(0.1, 0.6, f'Confidence: {result["confidence"]:.3f}', fontsize=12)
    axes[1, 2].text(0.1, 0.5, f'Defocus: {result["defocus"]:.1f} DAC', fontsize=12)
    axes[1, 2].text(0.1, 0.4, f'Valid Blocks: {result["valid_blocks"]}', fontsize=12)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_cdaf_contrast_curve(image: np.ndarray, 
                                 lens_positions: np.ndarray,
                                 save_path: Optional[str] = None):
    """
    Visualize CDAF contrast curve as lens moves
    
    Args:
        image: Base image
        lens_positions: Array of lens positions to test
        save_path: Optional path to save the visualization
    """
    config = AFConfig()
    cdaf_processor = CDAFProcessor(config)
    
    # Create ROI for center region
    h, w = image.shape[:2]
    roi_size = 200
    roi_x = (w - roi_size) // 2
    roi_y = (h - roi_size) // 2
    
    from autofocus_system import ROIData
    roi = ROIData(roi_x, roi_y, roi_size, roi_size, 1.0, 'center')
    
    # Simulate different blur levels (representing different lens positions)
    blur_levels = np.linspace(0.5, 6.0, len(lens_positions))
    contrasts = []
    
    for blur in blur_levels:
        # Create blurred version of image
        if blur > 0:
            blurred = cv2.GaussianBlur(image, (0, 0), blur)
        else:
            blurred = image
        
        # Calculate contrast
        contrast = cdaf_processor._calculate_contrast(blurred)
        contrasts.append(contrast)
    
    # Find peak contrast position
    peak_idx = np.argmax(contrasts)
    peak_position = lens_positions[peak_idx]
    peak_contrast = contrasts[peak_idx]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot contrast curve
    ax1.plot(lens_positions, contrasts, 'b-', linewidth=2, label='Contrast')
    ax1.plot(peak_position, peak_contrast, 'ro', markersize=10, 
             label=f'Peak at {peak_position:.0f}')
    ax1.set_xlabel('Lens Position (DAC)')
    ax1.set_ylabel('Contrast (Laplacian Variance)')
    ax1.set_title('CDAF Contrast Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Show images at different positions
    positions_to_show = [0, len(lens_positions)//4, len(lens_positions)//2, 
                        3*len(lens_positions)//4, len(lens_positions)-1]
    
    for i, pos_idx in enumerate(positions_to_show):
        if pos_idx < len(lens_positions):
            blur = blur_levels[pos_idx]
            position = lens_positions[pos_idx]
            
            if blur > 0:
                img_show = cv2.GaussianBlur(image, (0, 0), blur)
            else:
                img_show = image
            
            # Extract ROI
            roi_img = img_show[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
            
            # Show in subplot
            row = i // 3
            col = i % 3
            if row == 0:
                ax2_sub = plt.subplot2grid((2, 3), (row, col))
            else:
                ax2_sub = plt.subplot2grid((2, 3), (row, col))
            
            ax2_sub.imshow(roi_img, cmap='gray')
            ax2_sub.set_title(f'Pos: {position:.0f}\nBlur: {blur:.1f}\nContrast: {contrasts[pos_idx]:.1f}')
            ax2_sub.axis('off')
    
    # Remove the unused subplot
    if len(positions_to_show) < 6:
        fig.delaxes(fig.axes[-1])
    
    plt.suptitle('CDAF: Contrast vs Lens Position', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_af_convergence(convergence_data: List[Dict],
                           save_path: Optional[str] = None):
    """
    Visualize AF convergence behavior over iterations
    
    Args:
        convergence_data: List of iteration data
        save_path: Optional path to save the visualization
    """
    iterations = [d['iteration'] for d in convergence_data]
    positions = [d['position'] for d in convergence_data]
    confidences = [d.get('confidence', 0) for d in convergence_data]
    disparities = [d.get('disparity', 0) for d in convergence_data]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Position over iterations
    ax1.plot(iterations, positions, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Lens Position (DAC)')
    ax1.set_title('AF Convergence: Lens Position')
    ax1.grid(True, alpha=0.3)
    
    # Add convergence threshold
    if positions:
        final_pos = positions[-1]
        threshold = 1.0  # DAC threshold
        ax1.axhspan(final_pos - threshold, final_pos + threshold, 
                   alpha=0.3, color='green', label='Convergence Zone')
        ax1.legend()
    
    # Confidence over iterations
    if any(confidences):
        ax2.plot(iterations, confidences, 'go-', linewidth=2, markersize=6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Confidence')
        ax2.set_title('AF Convergence: Confidence')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
    
    # Disparity over iterations
    if any(disparities):
        ax3.plot(iterations, disparities, 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Disparity (pixels)')
        ax3.set_title('AF Convergence: Disparity')
        ax3.grid(True, alpha=0.3)
        
        # Add convergence threshold
        ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Convergence Threshold')
        ax3.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)
        ax3.legend()
    
    plt.suptitle('Autofocus Convergence Analysis', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_af_system_diagram(save_path: Optional[str] = None):
    """
    Create a system architecture diagram for the autofocus system
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Autofocus System Architecture', 
           fontsize=20, weight='bold', ha='center')
    
    # Input layer
    ax.text(5, 10.5, 'Input Data Sources', fontsize=16, weight='bold', ha='center')
    
    inputs = [
        (1, 9.5, 'Image Sensor\n(Bayer + PD)'),
        (3, 9.5, 'ISP Statistics\n(PDAF, CDAF)'),
        (5, 9.5, 'Scene Analysis\n(Face, Saliency)'),
        (7, 9.5, 'Depth Sensors\n(LDAF, ToF)'),
        (9, 9.5, 'User Input\n(Touch)')
    ]
    
    for x, y, text in inputs:
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                           facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=10, ha='center', va='center')
    
    # Processing layer
    ax.text(5, 8.5, 'Processing Pipeline', fontsize=16, weight='bold', ha='center')
    
    processes = [
        (2, 7.5, 'ROI\nSelection'),
        (4, 7.5, 'PDAF\nProcessing'),
        (6, 7.5, 'CDAF\nProcessing'),
        (8, 7.5, 'LDAF\nProcessing')
    ]
    
    for x, y, text in processes:
        rect = plt.Rectangle((x-0.4, y-0.3), 0.8, 0.6, 
                           facecolor='lightgreen', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x, y, text, fontsize=10, ha='center', va='center')
    
    # Decision layer
    ax.text(5, 6.5, 'Hybrid Decision Making', fontsize=16, weight='bold', ha='center')
    
    decision_rect = plt.Rectangle((3, 5.8), 4, 0.8, 
                                facecolor='orange', edgecolor='black')
    ax.add_patch(decision_rect)
    ax.text(5, 6.2, 'Weighted Fusion\nPDAF + CDAF + LDAF', 
           fontsize=12, ha='center', va='center')
    
    # Temporal filtering
    ax.text(5, 5, 'Temporal Filtering', fontsize=16, weight='bold', ha='center')
    
    filter_rect = plt.Rectangle((3, 4.3), 4, 0.8, 
                               facecolor='yellow', edgecolor='black')
    ax.add_patch(filter_rect)
    ax.text(5, 4.7, 'Kalman-like Filtering\nSmooth Convergence', 
           fontsize=12, ha='center', va='center')
    
    # Output layer
    ax.text(5, 3.5, 'Output', fontsize=16, weight='bold', ha='center')
    
    output_rect = plt.Rectangle((3, 2.8), 4, 0.8, 
                               facecolor='lightcoral', edgecolor='black')
    ax.add_patch(output_rect)
    ax.text(5, 3.2, 'Lens Position\nCommands (VCM)', 
           fontsize=12, ha='center', va='center')
    
    # Arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Input to processing
    for x in [1, 3, 5, 7, 9]:
        ax.annotate('', xy=(x, 8.2), xytext=(x, 9.2), arrowprops=arrow_props)
    
    # Processing to decision
    for x in [2, 4, 6, 8]:
        ax.annotate('', xy=(x, 6.6), xytext=(x, 7.2), arrowprops=arrow_props)
    
    # Decision to filtering
    ax.annotate('', xy=(5, 5.1), xytext=(5, 5.7), arrowprops=arrow_props)
    
    # Filtering to output
    ax.annotate('', xy=(5, 3.6), xytext=(5, 4.2), arrowprops=arrow_props)
    
    # Feedback arrow
    ax.annotate('', xy=(1, 9), xytext=(1, 3.6), 
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(0.5, 6.3, 'Feedback\nLoop', fontsize=10, ha='center', va='center', 
           rotation=90, color='red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def demo_visualizations():
    """
    Run all visualization demos
    """
    print("Creating Autofocus Visualizations...")
    
    # 1. PDAF Disparity Visualization
    print("\n1. Creating PDAF Disparity Visualization...")
    test_image = create_test_image(size=(400, 600), blur_radius=2.0)
    left_pd, right_pd = create_test_pd_data(test_image, disparity=3.0)
    visualize_pdaf_disparity(left_pd, right_pd)
    
    # 2. CDAF Contrast Curve
    print("\n2. Creating CDAF Contrast Curve...")
    test_image = create_test_image(size=(400, 600), blur_radius=0.5)
    lens_positions = np.linspace(300, 700, 20)
    visualize_cdaf_contrast_curve(test_image, lens_positions)
    
    # 3. AF Convergence
    print("\n3. Creating AF Convergence Visualization...")
    # Simulate convergence data
    convergence_data = [
        {'iteration': 0, 'position': 512, 'confidence': 0.3, 'disparity': 5.2},
        {'iteration': 1, 'position': 580, 'confidence': 0.6, 'disparity': 2.8},
        {'iteration': 2, 'position': 610, 'confidence': 0.8, 'disparity': 1.1},
        {'iteration': 3, 'position': 625, 'confidence': 0.9, 'disparity': 0.3},
        {'iteration': 4, 'position': 628, 'confidence': 0.95, 'disparity': 0.1},
    ]
    visualize_af_convergence(convergence_data)
    
    # 4. System Architecture
    print("\n4. Creating System Architecture Diagram...")
    create_af_system_diagram()
    
    print("\nVisualization demo complete!")


if __name__ == "__main__":
    demo_visualizations()
