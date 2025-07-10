import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import os
from pathlib import Path
import warnings
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import glob
from datetime import datetime

# Handle missing dependencies gracefully
try:
    from skimage import morphology, measure, segmentation, filters
    from skimage.feature import peak_local_max
    from skimage import io as skio
    from PIL import Image, ImageTk
except ImportError as e:
    print(f"Error: Missing required packages!")
    print("Please install them with:")
    print("pip install scikit-image pillow matplotlib")
    print(f"Missing: {e}")
    input("Press Enter to exit...")
    sys.exit(1)

warnings.filterwarnings('ignore')

class InteractiveCropper:
    def __init__(self, image):
        self.image = image
        self.crop_coords = None
        self.fig = None
        self.ax = None
        
    def crop_image_interactive(self):
        """Interactive image cropping using matplotlib"""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.ax.imshow(self.image, cmap='gray')
        self.ax.set_title('Select area to crop (drag to select, close window when done)')
        
        # Rectangle selector
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
        plt.show()
        
        if self.crop_coords is not None:
            x1, y1, x2, y2 = self.crop_coords
            cropped = self.image[int(y1):int(y2), int(x1):int(x2)]
            return cropped, self.crop_coords
        else:
            return self.image, None
    
    def on_select(self, eclick, erelease):
        """Handle rectangle selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.crop_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"Selected crop area: {self.crop_coords}")

class EnhancedBubbleDetector:
    def __init__(self, min_bubble_area=20, max_bubble_area=50000, 
                 min_radius=3, max_radius=50, block_size=81, c_value=7):
        """
        Enhanced bubble detector with cropping and background subtraction
        """
        self.min_bubble_area = min_bubble_area
        self.max_bubble_area = max_bubble_area
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.block_size = block_size
        self.c_value = c_value
        self.background_image = None
        self.px_per_mm = 1.0  # Default pixel to mm ratio
        
    def load_image_robust(self, image_path):
        """Robust image loading that handles various TIFF formats"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = None
        
        # Method 1: Try OpenCV first
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"Loaded with OpenCV: {img.shape}, dtype: {img.dtype}")
                
                # Convert to grayscale if needed
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Enhance contrast for TIFF images
                if img.dtype == np.uint16:
                    # Normalize 16-bit to 8-bit with contrast enhancement
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                elif img.dtype != np.uint8:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                
                return img
        except Exception as e:
            print(f"OpenCV loading failed: {e}")
        
        # Fallback methods...
        try:
            img = skio.imread(image_path, as_gray=True)
            if img is not None:
                if img.dtype != np.uint8:
                    p2, p98 = np.percentile(img, (2, 98))
                    img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                return img
        except Exception as e:
            print(f"Scikit-image loading failed: {e}")
        
        raise ValueError(f"Could not load image with any method: {image_path}")
    
    def load_background_image(self, background_path):
        """Load background image for subtraction"""
        try:
            self.background_image = self.load_image_robust(background_path)
            print(f"Background image loaded: {self.background_image.shape}")
            return True
        except Exception as e:
            print(f"Error loading background image: {e}")
            return False
    
    def crop_image_interactive(self, image):
        """Interactive cropping interface"""
        print("Opening interactive cropping window...")
        cropper = InteractiveCropper(image)
        cropped_image, crop_coords = cropper.crop_image_interactive()
        
        if crop_coords is not None:
            print(f"Image cropped to: {cropped_image.shape}")
            return cropped_image, crop_coords
        else:
            print("No cropping applied")
            return image, None
    
    def apply_background_subtraction(self, image, background=None):
        """Apply background subtraction"""
        if background is None:
            background = self.background_image
            
        if background is None:
            print("No background image available")
            return image
        
        # Resize background to match image if needed
        if background.shape != image.shape:
            background = cv2.resize(background, (image.shape[1], image.shape[0]))
        
        # Perform background subtraction
        subtracted = cv2.absdiff(image, background)
        
        # Enhance the result
        subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        
        print("Background subtraction applied")
        return subtracted
    
    def enhance_contrast_adaptive(self, image):
        """Apply adaptive contrast enhancement"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        
        # Additional gamma correction for TIFF images
        gamma = 1.2
        gamma_corrected = np.power(enhanced / 255.0, 1.0 / gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        
        return gamma_corrected
    
    def preprocess_image_enhanced(self, image, apply_bg_subtraction=False, background=None):
        """Enhanced preprocessing pipeline"""
        preprocessing_steps = {}
        
        # Step 1: Original
        preprocessing_steps['original'] = image.copy()
        
        # Step 2: Background subtraction (if requested)
        if apply_bg_subtraction and (background is not None or self.background_image is not None):
            image = self.apply_background_subtraction(image, background)
            preprocessing_steps['background_subtracted'] = image.copy()
        
        # Step 3: Contrast enhancement
        enhanced = self.enhance_contrast_adaptive(image)
        preprocessing_steps['contrast_enhanced'] = enhanced
        
        # Step 4: Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        preprocessing_steps['blurred'] = blurred
        
        # Step 5: Median filtering for noise reduction
        denoised = cv2.medianBlur(blurred, 5)
        preprocessing_steps['denoised'] = denoised
        
        # Step 6: Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c_value
        )
        preprocessing_steps['binary'] = binary
        
        # Step 7: Morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        preprocessing_steps['morphology_cleaned'] = cleaned
        
        return cleaned, preprocessing_steps
    
    def detect_bubbles_enhanced(self, image_path, crop_image=True, apply_bg_subtraction=False, 
                              background_path=None, px_per_mm=1.0, save_results=True, output_dir='results',
                              crop_coords=None):
        """Enhanced bubble detection with all features"""
        
        print(f"Processing: {image_path}")
        self.px_per_mm = px_per_mm
        
        try:
            # Load main image
            image = self.load_image_robust(image_path)
            original_image = image.copy()
            
            # Load background if needed
            background = None
            if apply_bg_subtraction and background_path:
                if self.load_background_image(background_path):
                    background = self.background_image
            
            # Apply cropping
            if crop_coords is not None:
                # Use provided crop coordinates
                x1, y1, x2, y2 = crop_coords
                image = image[int(y1):int(y2), int(x1):int(x2)]
                if background is not None:
                    background = background[int(y1):int(y2), int(x1):int(x2)]
                print(f"Applied crop coordinates: {crop_coords}")
            elif crop_image:
                # Interactive cropping
                image, crop_coords = self.crop_image_interactive(image)
                if crop_coords and background is not None:
                    # Crop background too
                    x1, y1, x2, y2 = crop_coords
                    background = background[int(y1):int(y2), int(x1):int(x2)]
            
            # Preprocessing
            binary_mask, preprocessing_steps = self.preprocess_image_enhanced(
                image, apply_bg_subtraction, background
            )
            
            # Find contours
            contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours)} raw contours.")
            
            # Analyze bubbles
            bubble_data = []
            valid_contours = []
            
            for i, contour in enumerate(contours):
                area_px = cv2.contourArea(contour)
                if self.min_bubble_area < area_px < self.max_bubble_area:
                    # Calculate properties
                    bubble_props = self.calculate_bubble_properties(contour, area_px, i+1)
                    bubble_data.append(bubble_props)
                    valid_contours.append(contour)
            
            print(f"Detected {len(bubble_data)} valid bubbles")
            
            if save_results:
                self.save_enhanced_results(
                    image_path, preprocessing_steps, binary_mask, 
                    image, bubble_data, valid_contours, output_dir, crop_coords
                )
            
            return bubble_data, valid_contours, preprocessing_steps
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return [], [], {}
    
    def calculate_bubble_properties(self, contour, area_px, bubble_id):
        """Calculate comprehensive bubble properties"""
        # Basic measurements
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0
        
        perimeter = cv2.arcLength(contour, True)
        equiv_diameter_px = np.sqrt(4 * area_px / np.pi)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        # Ellipse fitting
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center_x, center_y), (minor_axis, major_axis), orientation = ellipse
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        else:
            major_axis = minor_axis = equiv_diameter_px
            aspect_ratio = 1.0
            orientation = 0
        
        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Convert to real-world units
        area_mm2 = area_px / (self.px_per_mm ** 2)
        diameter_mm = equiv_diameter_px / self.px_per_mm
        
        return {
            'bubble_id': bubble_id,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'area_px': area_px,
            'area_mm2': area_mm2,
            'perimeter_px': perimeter,
            'diameter_px': equiv_diameter_px,
            'diameter_mm': diameter_mm,
            'major_axis_px': major_axis,
            'minor_axis_px': minor_axis,
            'orientation': orientation,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
        }
    
    def save_enhanced_results(self, image_path, preprocessing_steps, binary_mask, 
                            processed_image, bubble_data, contours, output_dir, crop_coords):
        """Save comprehensive results with multiple visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        filename = Path(image_path).stem
        
        # 1. Stage-wise preprocessing progression
        self.save_preprocessing_stages(preprocessing_steps, filename, output_dir)
        
        # 2. Bubble detection results
        self.save_bubble_detection_results(processed_image, bubble_data, contours, filename, output_dir)
        
        # 3. Individual bubble analysis
        self.save_individual_bubble_analysis(processed_image, bubble_data, contours, filename, output_dir)
        
        # 4. CSV results
        self.save_csv_results(bubble_data, filename, output_dir)
        
        print(f"Results for {filename} saved to: {output_dir}")
    
    def save_preprocessing_stages(self, preprocessing_steps, filename, output_dir):
        """Save stage-wise preprocessing visualization"""
        n_steps = len(preprocessing_steps)
        cols = 3
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        step_names = list(preprocessing_steps.keys())
        for i, (step_name, step_image) in enumerate(preprocessing_steps.items()):
            row, col = i // cols, i % cols
            axes[row, col].imshow(step_image, cmap='gray')
            axes[row, col].set_title(f'Step {i+1}: {step_name.replace("_", " ").title()}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_steps, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}_preprocessing_stages.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_bubble_detection_results(self, image, bubble_data, contours, filename, output_dir):
        """Save bubble detection visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original with detected bubbles
        result_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i, contour in enumerate(contours):
            cv2.drawContours(result_img, [contour], -1, (0, 255, 0), 2)
            # Add bubble ID
            if i < len(bubble_data):
                centroid = (bubble_data[i]['centroid_x'], bubble_data[i]['centroid_y'])
                cv2.putText(result_img, str(bubble_data[i]['bubble_id']), 
                           centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        axes[0].imshow(result_img)
        axes[0].set_title(f'Detected Bubbles (n={len(bubble_data)})')
        axes[0].axis('off')
        
        # Size distribution
        if bubble_data:
            diameters = [b['diameter_mm'] for b in bubble_data]
            axes[1].hist(diameters, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1].set_title('Bubble Diameter Distribution')
            axes[1].set_xlabel('Diameter (mm)')
            axes[1].set_ylabel('Count')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}_bubble_detection.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_individual_bubble_analysis(self, image, bubble_data, contours, filename, output_dir):
        """Save individual bubble analysis with IDs and measurements"""
        if not bubble_data:
            return
        
        n_bubbles = len(bubble_data)
        cols = 5
        rows = (n_bubbles + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (bubble, contour) in enumerate(zip(bubble_data, contours)):
            row, col = i // cols, i % cols
            
            # Extract bubble region
            x, y, w, h = cv2.boundingRect(contour)
            padding = 10
            x1, y1 = max(0, x-padding), max(0, y-padding)
            x2, y2 = min(image.shape[1], x+w+padding), min(image.shape[0], y+h+padding)
            
            bubble_region = image[y1:y2, x1:x2]
            
            axes[row, col].imshow(bubble_region, cmap='gray')
            axes[row, col].set_title(
                f'ID: {bubble["bubble_id"]}\n'
                f'D: {bubble["diameter_mm"]:.2f}mm\n'
                f'A: {bubble["area_mm2"]:.3f}mm²',
                fontsize=8
            )
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_bubbles, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / f'{filename}_individual_bubbles.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_csv_results(self, bubble_data, filename, output_dir):
        """Save bubble measurements to CSV"""
        if bubble_data:
            df = pd.DataFrame(bubble_data)
            csv_path = Path(output_dir) / f'{filename}_bubble_measurements.csv'
            df.to_csv(csv_path, index=False)
            
            # Print summary
            print(f"\n=== BUBBLE ANALYSIS SUMMARY FOR {filename} ===")
            print(f"Total bubbles detected: {len(df)}")
            print(f"Average diameter: {df['diameter_mm'].mean():.3f} ± {df['diameter_mm'].std():.3f} mm")
            print(f"Diameter range: {df['diameter_mm'].min():.3f} - {df['diameter_mm'].max():.3f} mm")
            print(f"Average area: {df['area_mm2'].mean():.4f} ± {df['area_mm2'].std():.4f} mm²")
            print(f"Average circularity: {df['circularity'].mean():.3f} ± {df['circularity'].std():.3f}")
    
    def batch_process_images(self, folder_path, file_pattern="*.tif", crop_coords=None,
                           apply_bg_subtraction=False, background_path=None,
                           px_per_mm=1.0, output_dir='batch_results'):
        """Process multiple images in batch mode"""
        
        # Find all matching files
        search_pattern = os.path.join(folder_path, file_pattern)
        image_files = glob.glob(search_pattern)
        
        # Also check for .tiff extension
        if file_pattern == "*.tif":
            tiff_pattern = os.path.join(folder_path, "*.tiff")
            image_files.extend(glob.glob(tiff_pattern))
        
        if not image_files:
            print(f"No images found matching pattern: {search_pattern}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_output_dir = Path(output_dir) / f"batch_{timestamp}"
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each image
        all_results = []
        batch_summary = []
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n=== PROCESSING IMAGE {i}/{len(image_files)}: {Path(image_path).name} ===")
            
            try:
                # Process image
                bubble_data, contours, preprocessing_steps = self.detect_bubbles_enhanced(
                    image_path=image_path,
                    crop_image=False,  # Don't do interactive cropping in batch mode
                    apply_bg_subtraction=apply_bg_subtraction,
                    background_path=background_path,
                    px_per_mm=px_per_mm,
                    save_results=True,
                    output_dir=str(batch_output_dir),
                    crop_coords=crop_coords
                )
                
                # Collect results
                all_results.extend(bubble_data)
                
                # Summary for this image
                filename = Path(image_path).stem
                if bubble_data:
                    diameters = [b['diameter_mm'] for b in bubble_data]
                    areas = [b['area_mm2'] for b in bubble_data]
                    circularities = [b['circularity'] for b in bubble_data]
                    
                    summary = {
                        'filename': filename,
                        'total_bubbles': len(bubble_data),
                        'mean_diameter_mm': np.mean(diameters),
                        'std_diameter_mm': np.std(diameters),
                        'min_diameter_mm': np.min(diameters),
                        'max_diameter_mm': np.max(diameters),
                        'mean_area_mm2': np.mean(areas),
                        'std_area_mm2': np.std(areas),
                        'mean_circularity': np.mean(circularities),
                        'std_circularity': np.std(circularities),
                    }
                else:
                    summary = {
                        'filename': filename,
                        'total_bubbles': 0,
                        'mean_diameter_mm': 0,
                        'std_diameter_mm': 0,
                        'min_diameter_mm': 0,
                        'max_diameter_mm': 0,
                        'mean_area_mm2': 0,
                        'std_area_mm2': 0,
                        'mean_circularity': 0,
                        'std_circularity': 0,
                    }
                
                batch_summary.append(summary)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Save batch summary
        if batch_summary:
            summary_df = pd.DataFrame(batch_summary)
            summary_path = batch_output_dir / "batch_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nBatch summary saved to: {summary_path}")
            
            # Print overall statistics
            total_bubbles = summary_df['total_bubbles'].sum()
            print(f"\n=== BATCH PROCESSING COMPLETED ===")
            print(f"Images processed: {len(image_files)}")
            print(f"Total bubbles detected: {total_bubbles}")
            print(f"Average bubbles per image: {total_bubbles/len(image_files):.1f}")
            print(f"Results saved to: {batch_output_dir}")
        
        return all_results

def get_user_inputs():
    """Get user inputs with GUI dialogs"""
    root = tk.Tk()
    root.withdraw()  # Hide main window
    
    # Ask for processing mode
    mode = messagebox.askyesnocancel("Processing Mode", 
                                   "Choose processing mode:\n"
                                   "Yes = Single Image\n"
                                   "No = Batch Processing\n"
                                   "Cancel = Exit")
    
    if mode is None:  # User clicked Cancel
        return None
    
    if mode:  # Single image mode
        return get_single_image_inputs(root)
    else:  # Batch processing mode
        return get_batch_processing_inputs(root)

def get_single_image_inputs(root):
    """Get inputs for single image processing"""
    # Get image file
    image_path = filedialog.askopenfilename(
        title="Select image file",
        filetypes=[
            ("TIFF files", "*.tif *.tiff"),
            ("All image files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )
    
    if not image_path:
        return None
    
    # Ask about cropping
    crop_image = messagebox.askyesno("Cropping", "Do you want to crop the image interactively?")
    
    # Get remaining parameters
    params = get_common_parameters(root)
    if params is None:
        return None
    
    # Output directory
    output_dir = filedialog.askdirectory(title="Select output directory")
    if not output_dir:
        output_dir = "results"
    
    return {
        'mode': 'single',
        'image_path': image_path,
        'crop_image': crop_image,
        'output_dir': output_dir,
        **params
    }

def get_batch_processing_inputs(root):
    """Get inputs for batch processing"""
    # Get folder containing images
    folder_path = filedialog.askdirectory(title="Select folder containing TIFF images")
    
    if not folder_path:
        return None
    
    # Get file pattern
    file_pattern = simpledialog.askstring("File Pattern", 
                                        "Enter file pattern (e.g., *.tif, *.tiff, sample_*.tif):",
                                        initialvalue="*.tif")
    if not file_pattern:
        file_pattern = "*.tif"
    
    # Ask about uniform cropping
    use_crop = messagebox.askyesno("Batch Cropping", 
                                  "Do you want to apply the same crop to all images?\n"
                                  "(You'll define the crop area using the first image)")
    
    crop_coords = None
    if use_crop:
        # Load first image to define crop area
        search_pattern = os.path.join(folder_path, file_pattern)
        image_files = glob.glob(search_pattern)
        if file_pattern == "*.tif":
            image_files.extend(glob.glob(os.path.join(folder_path, "*.tiff")))
        
        if image_files:
            try:
                detector = EnhancedBubbleDetector()
                first_image = detector.load_image_robust(image_files[0])
                cropped_image, crop_coords = detector.crop_image_interactive(first_image)
                print(f"Crop coordinates for batch: {crop_coords}")
            except Exception as e:
                print(f"Error loading first image for cropping: {e}")
                crop_coords = None
    
    # Get remaining parameters
    params = get_common_parameters(root)
    if params is None:
        return None
    
    # Output directory
    output_dir = filedialog.askdirectory(title="Select output directory for batch results")
    if not output_dir:
        output_dir = "batch_results"
    
    return {
        'mode': 'batch',
        'folder_path': folder_path,
        'file_pattern': file_pattern,
        'crop_coords': crop_coords,
        'output_dir': output_dir,
        **params
    }

def get_common_parameters(root):
    """Get parameters common to both single and batch processing"""
    # Ask about background subtraction
    apply_bg_subtraction = messagebox.askyesno("Background Subtraction", 
                                              "Do you want to apply background subtraction?")
    
    background_path = None
    if apply_bg_subtraction:
        background_path = filedialog.askopenfilename(
            title="Select background image",
            filetypes=[
                ("TIFF files", "*.tif *.tiff"),
                ("All image files", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
    
    # Get pixel to mm conversion
    px_per_mm = simpledialog.askfloat("Calibration", 
                                     "Enter pixels per mm (e.g., 10.5):", 
                                     initialvalue=1.0, minvalue=0.001)
    if px_per_mm is None:
        px_per_mm = 1.0
    
    # Get detection parameters
    min_area = simpledialog.askinteger("Parameters", "Minimum bubble area (pixels):", 
                                      initialvalue=20, minvalue=1)
    if min_area is None:
        min_area = 20
        
    max_area = simpledialog.askinteger("Parameters", "Maximum bubble area (pixels):", 
                                      initialvalue=50000, minvalue=min_area)
    if max_area is None:
        max_area = 50000
    
    # Get thresholding parameters
    block_size = simpledialog.askinteger("Thresholding", 
                                        "Adaptive threshold block size (odd number, e.g., 81):", 
                                        initialvalue=81, minvalue=3)
    if block_size is None:
        block_size = 81
    elif block_size % 2 == 0:  # Ensure odd number
        block_size += 1
        print(f"Block size adjusted to {block_size} (must be odd)")
    
    c_value = simpledialog.askinteger("Thresholding", 
                                     "Adaptive threshold C value (e.g., 7):", 
                                     initialvalue=7, minvalue=0)
    if c_value is None:
        c_value = 7
    
    return {
        'apply_bg_subtraction': apply_bg_subtraction,
        'background_path': background_path,
        'px_per_mm': px_per_mm,
        'min_area': min_area,
        'max_area': max_area,
        'block_size': block_size,
        'c_value': c_value
    }

def main():
    """Main function"""
    print("=== ENHANCED BUBBLE DETECTION SYSTEM ===")
    print("Features:")
    print("- Single image or batch processing")
    print("- Interactive image cropping")
    print("- Configurable thresholding parameters")
    print("- Background subtraction")
    print("- Contrast enhancement for TIFF images")
    print("- Stage-wise processing visualization")
    print("- Individual bubble analysis")
    print("- Real-world measurements (mm)")
    print("- Batch processing with summary statistics")
    print()
    
    inputs = get_user_inputs()
    if inputs is None:
        print("Operation cancelled.")
        return
    
    try:
        # Initialize detector with user parameters
        detector = EnhancedBubbleDetector(
            min_bubble_area=inputs['min_area'],
            max_bubble_area=inputs['max_area'],
            block_size=inputs['block_size'],
            c_value=inputs['c_value']
        )
        
        print(f"\nDetector Parameters:")
        print(f"- Min bubble area: {inputs['min_area']} pixels")
        print(f"- Max bubble area: {inputs['max_area']} pixels")
        print(f"- Threshold block size: {inputs['block_size']}")
        print(f"- Threshold C value: {inputs['c_value']}")
        print(f"- Calibration: {inputs['px_per_mm']} pixels per mm")
        
        if inputs['mode'] == 'single':
            # Single image processing
            print(f"\n=== SINGLE IMAGE PROCESSING ===")
            print(f"Processing: {inputs['image_path']}")
            
            bubble_data, contours, preprocessing_steps = detector.detect_bubbles_enhanced(
                image_path=inputs['image_path'],
                crop_image=inputs['crop_image'],
                apply_bg_subtraction=inputs['apply_bg_subtraction'],
                background_path=inputs['background_path'],
                px_per_mm=inputs['px_per_mm'],
                save_results=True,
                output_dir=inputs['output_dir']
            )
            
            print(f"\n=== PROCESSING COMPLETED ===")
            print(f"Results saved to: {inputs['output_dir']}")
            print("Generated files:")
            print("- *_preprocessing_stages.png: Stage-wise processing")
            print("- *_bubble_detection.png: Detection results")
            print("- *_individual_bubbles.png: Individual bubble analysis")
            print("- *_bubble_measurements.csv: Detailed measurements")
            
        else:
            # Batch processing
            print(f"\n=== BATCH PROCESSING ===")
            print(f"Processing folder: {inputs['folder_path']}")
            print(f"File pattern: {inputs['file_pattern']}")
            
            all_results = detector.batch_process_images(
                folder_path=inputs['folder_path'],
                file_pattern=inputs['file_pattern'],
                crop_coords=inputs['crop_coords'],
                apply_bg_subtraction=inputs['apply_bg_subtraction'],
                background_path=inputs['background_path'],
                px_per_mm=inputs['px_per_mm'],
                output_dir=inputs['output_dir']
            )
            
            print(f"\n=== BATCH PROCESSING COMPLETED ===")
            print("Generated files for each image:")
            print("- *_preprocessing_stages.png: Stage-wise processing")
            print("- *_bubble_detection.png: Detection results")
            print("- *_individual_bubbles.png: Individual bubble analysis")
            print("- *_bubble_measurements.csv: Individual measurements")
            print("- batch_summary.csv: Summary statistics for all images")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()