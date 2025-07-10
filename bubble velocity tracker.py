import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

# Handle missing dependencies
try:
    from skimage import morphology, measure, segmentation, filters
    from PIL import Image, ImageTk
    import openpyxl
except ImportError as e:
    print(f"Error: Missing required packages!")
    print("Please install them with:")
    print("pip install scikit-image pillow matplotlib openpyxl")
    print(f"Missing: {e}")
    exit(1)

class BubbleTracker:
    """Tracks bubbles across frames and calculates velocities"""
    
    def __init__(self, px_per_mm=1.0, fps=6400, max_displacement_px=50):
        self.px_per_mm = px_per_mm
        self.fps = fps
        self.max_displacement_px = max_displacement_px
        self.previous_bubbles = []
        self.frame_count = 0
        self.all_tracked_data = []
        
    def track_bubbles(self, current_bubbles, frame_name):
        """Track bubbles between frames and calculate velocities"""
        tracked_bubbles = []
        
        if self.frame_count == 0:
            # First frame - no velocity calculation
            for bubble in current_bubbles:
                bubble['velocity_mm_per_s'] = 0
                bubble['frame'] = frame_name
                tracked_bubbles.append(bubble)
        else:
            # Calculate velocities based on previous frame
            if len(self.previous_bubbles) > 0 and len(current_bubbles) > 0:
                prev_pos = np.array([(b['centroid_x'], b['centroid_y']) for b in self.previous_bubbles])
                curr_pos = np.array([(b['centroid_x'], b['centroid_y']) for b in current_bubbles])
                
                # Find closest matches
                distances = cdist(curr_pos, prev_pos)
                assignments = np.argmin(distances, axis=1)
                
                for i, bubble in enumerate(current_bubbles):
                    min_dist = distances[i, assignments[i]]
                    
                    if min_dist <= self.max_displacement_px:
                        # Calculate velocity
                        dx = curr_pos[i][0] - prev_pos[assignments[i]][0]
                        dy = curr_pos[i][1] - prev_pos[assignments[i]][1]
                        dist_px = np.sqrt(dx**2 + dy**2)
                        velocity_mm_per_s = (dist_px / self.px_per_mm) * self.fps
                        bubble['velocity_mm_per_s'] = velocity_mm_per_s
                    else:
                        # New bubble or too far displacement
                        bubble['velocity_mm_per_s'] = 0
                    
                    bubble['frame'] = frame_name
                    tracked_bubbles.append(bubble)
        
        self.previous_bubbles = current_bubbles.copy()
        self.frame_count += 1
        self.all_tracked_data.extend(tracked_bubbles)
        
        return tracked_bubbles
    
    def export_to_excel(self, output_path):
        """Export all tracked bubble data to Excel"""
        if not self.all_tracked_data:
            print("No tracking data to export")
            return
            
        df = pd.DataFrame(self.all_tracked_data)
        
        # Create Excel file with multiple sheets
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # All data
            df.to_excel(writer, sheet_name='All_Bubbles', index=False)
            
            # Summary statistics per frame
            summary = df.groupby('frame').agg({
                'bubble_id': 'count',
                'diameter_mm': ['mean', 'std', 'min', 'max'],
                'velocity_mm_per_s': ['mean', 'std', 'min', 'max'],
                'area_mm2': ['mean', 'std']
            }).round(4)
            
            summary.columns = ['_'.join(col).strip() for col in summary.columns]
            summary.to_excel(writer, sheet_name='Frame_Summary')
            
            # High velocity bubbles (top 10%)
            velocity_threshold = df['velocity_mm_per_s'].quantile(0.9)
            high_velocity = df[df['velocity_mm_per_s'] >= velocity_threshold]
            high_velocity.to_excel(writer, sheet_name='High_Velocity_Bubbles', index=False)
        
        print(f"Velocity tracking data exported to: {output_path}")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total bubble detections: {len(self.all_tracked_data)}")
        if len(self.all_tracked_data) > 0:
            print(f"Average velocity: {df['velocity_mm_per_s'].mean():.2f} mm/s")
            print(f"Max velocity: {df['velocity_mm_per_s'].max():.2f} mm/s")

class EnhancedBubbleDetector:
    """Enhanced bubble detector with velocity tracking capabilities"""
    
    def __init__(self, min_bubble_area=20, max_bubble_area=50000, 
                 block_size=51, c_value=2):
        self.min_bubble_area = min_bubble_area
        self.max_bubble_area = max_bubble_area
        self.block_size = block_size
        self.c_value = c_value
        self.px_per_mm = 1.0
        
    def load_image_robust(self, image_path):
        """Load and preprocess TIFF images"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert to grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Normalize to 8-bit
        if img.dtype == np.uint16:
            p2, p98 = np.percentile(img, (2, 98))
            img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        
        return img
    
    def detect_bubbles_simple(self, image_path, crop_coords=None):
        """Simplified bubble detection for velocity tracking"""
        image = self.load_image_robust(image_path)
        
        # Apply crop if provided
        if crop_coords:
            x1, y1, x2, y2 = crop_coords
            image = image[int(y1):int(y2), int(x1):int(x2)]
        
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(image)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        denoised = cv2.medianBlur(blurred, 5)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.block_size, self.c_value
        )
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze bubbles
        bubble_data = []
        for i, contour in enumerate(contours):
            area_px = cv2.contourArea(contour)
            if self.min_bubble_area < area_px < self.max_bubble_area:
                bubble_props = self.calculate_bubble_properties(contour, area_px, i+1)
                bubble_data.append(bubble_props)
        
        return bubble_data
    
    def calculate_bubble_properties(self, contour, area_px, bubble_id):
        """Calculate bubble properties"""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
        else:
            centroid_x, centroid_y = 0, 0
        
        perimeter = cv2.arcLength(contour, True)
        equiv_diameter_px = np.sqrt(4 * area_px / np.pi)
        circularity = 4 * np.pi * area_px / (perimeter ** 2) if perimeter > 0 else 0
        
        # Convert to mm
        area_mm2 = area_px / (self.px_per_mm ** 2)
        diameter_mm = equiv_diameter_px / self.px_per_mm
        
        return {
            'bubble_id': bubble_id,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'area_px': area_px,
            'area_mm2': area_mm2,
            'diameter_mm': diameter_mm,
            'perimeter_px': perimeter,
            'circularity': circularity,
        }
    
    def process_video_frames_with_velocity(self, folder_path, file_pattern="*.tif", 
                                         crop_coords=None, px_per_mm=1.0, fps=6400,
                                         max_displacement_px=50, output_dir='velocity_results'):
        """Process video frames and track bubble velocities"""
        
        # Find all frames
        search_pattern = os.path.join(folder_path, file_pattern)
        image_files = sorted(glob.glob(search_pattern))
        
        if file_pattern == "*.tif":
            image_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.tiff"))))
            image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        
        if not image_files:
            print(f"No images found matching pattern: {search_pattern}")
            return
        
        print(f"Found {len(image_files)} frames to process")
        
        # Initialize tracker and detector
        self.px_per_mm = px_per_mm
        tracker = BubbleTracker(px_per_mm=px_per_mm, fps=fps, 
                              max_displacement_px=max_displacement_px)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Process each frame
        print("Processing frames for velocity tracking...")
        for i, image_path in enumerate(image_files):
            frame_name = Path(image_path).stem
            print(f"Processing frame {i+1}/{len(image_files)}: {frame_name}")
            
            try:
                # Detect bubbles
                bubble_data = self.detect_bubbles_simple(image_path, crop_coords)
                
                # Track bubbles and calculate velocities
                tracked_bubbles = tracker.track_bubbles(bubble_data, frame_name)
                
                if i % 50 == 0:  # Progress update every 50 frames
                    print(f"  Detected {len(tracked_bubbles)} bubbles")
                    
            except Exception as e:
                print(f"Error processing frame {frame_name}: {e}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = output_path / f"bubble_velocity_tracking_{timestamp}.xlsx"
        tracker.export_to_excel(excel_path)
        
        # Create summary visualization
        self.create_velocity_summary_plot(tracker.all_tracked_data, output_path / f"velocity_summary_{timestamp}.png")
        
        return tracker.all_tracked_data
    
    def create_velocity_summary_plot(self, tracked_data, output_path):
        """Create velocity summary visualization"""
        if not tracked_data:
            return
            
        df = pd.DataFrame(tracked_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Velocity distribution
        axes[0,0].hist(df['velocity_mm_per_s'], bins=50, alpha=0.7, color='blue')
        axes[0,0].set_title('Velocity Distribution')
        axes[0,0].set_xlabel('Velocity (mm/s)')
        axes[0,0].set_ylabel('Count')
        
        # Velocity vs bubble size
        axes[0,1].scatter(df['diameter_mm'], df['velocity_mm_per_s'], alpha=0.6, s=20)
        axes[0,1].set_title('Velocity vs Bubble Size')
        axes[0,1].set_xlabel('Diameter (mm)')
        axes[0,1].set_ylabel('Velocity (mm/s)')
        
        # Velocity over time (frames)
        frame_avg = df.groupby('frame')['velocity_mm_per_s'].mean()
        axes[1,0].plot(range(len(frame_avg)), frame_avg.values)
        axes[1,0].set_title('Average Velocity Over Time')
        axes[1,0].set_xlabel('Frame Number')
        axes[1,0].set_ylabel('Average Velocity (mm/s)')
        
        # Bubble count per frame
        frame_counts = df.groupby('frame').size()
        axes[1,1].plot(range(len(frame_counts)), frame_counts.values)
        axes[1,1].set_title('Bubble Count Over Time')
        axes[1,1].set_xlabel('Frame Number')
        axes[1,1].set_ylabel('Number of Bubbles')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Velocity summary plot saved to: {output_path}")

def get_velocity_tracking_inputs():
    """Get user inputs for velocity tracking"""
    root = tk.Tk()
    root.withdraw()
    
    # Get folder with video frames
    folder_path = filedialog.askdirectory(title="Select folder containing video frames (TIFF files)")
    if not folder_path:
        return None
    
    # Get parameters
    px_per_mm = simpledialog.askfloat("Calibration", "Enter pixels per mm:", initialvalue=1.0)
    fps = simpledialog.askfloat("Video FPS", "Enter video frame rate (fps):", initialvalue=6400)
    max_displacement = simpledialog.askfloat("Tracking", "Maximum displacement between frames (pixels):", initialvalue=50)
    
    # Optional cropping
    use_crop = messagebox.askyesno("Cropping", "Apply uniform crop to all frames?")
    crop_coords = None
    
    if use_crop:
        # Load first image for cropping
        first_image = None
        for pattern in ["*.tif", "*.tiff"]:
            files = glob.glob(os.path.join(folder_path, pattern))
            if files:
                try:
                    detector = EnhancedBubbleDetector()
                    first_image = detector.load_image_robust(sorted(files)[0])
                    break
                except:
                    continue
        
        if first_image is not None:
            from matplotlib.widgets import RectangleSelector
            
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(first_image, cmap='gray')
            ax.set_title('Select crop area (drag to select, close window when done)')
            
            crop_coords = [None]
            
            def on_select(eclick, erelease):
                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata
                crop_coords[0] = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                print(f"Crop area selected: {crop_coords[0]}")
            
            selector = RectangleSelector(ax, on_select, useblit=True, button=[1])
            plt.show()
            
            crop_coords = crop_coords[0]
    
    return {
        'folder_path': folder_path,
        'px_per_mm': px_per_mm if px_per_mm else 1.0,
        'fps': fps if fps else 6400,
        'max_displacement': max_displacement if max_displacement else 50,
        'crop_coords': crop_coords
    }

def main():
    """Main function for velocity tracking"""
    print("=== BUBBLE VELOCITY TRACKING SYSTEM ===")
    print("Features:")
    print("- Track bubbles across video frames")
    print("- Calculate velocities at 6400 fps")
    print("- Export to Excel with multiple sheets")
    print("- Velocity analysis and visualization")
    print()
    
    inputs = get_velocity_tracking_inputs()
    if inputs is None:
        print("Operation cancelled.")
        return
    
    try:
        # Initialize detector
        detector = EnhancedBubbleDetector()
        
        print(f"Processing parameters:")
        print(f"- Folder: {inputs['folder_path']}")
        print(f"- Calibration: {inputs['px_per_mm']} pixels/mm")
        print(f"- Frame rate: {inputs['fps']} fps")
        print(f"- Max displacement: {inputs['max_displacement']} pixels")
        
        # Process video frames
        results = detector.process_video_frames_with_velocity(
            folder_path=inputs['folder_path'],
            crop_coords=inputs['crop_coords'],
            px_per_mm=inputs['px_per_mm'],
            fps=inputs['fps'],
            max_displacement_px=inputs['max_displacement']
        )
        
        print(f"\n=== VELOCITY TRACKING COMPLETED ===")
        print("Generated files:")
        print("- bubble_velocity_tracking_*.xlsx: Excel file with velocity data")
        print("- velocity_summary_*.png: Velocity analysis plots")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()