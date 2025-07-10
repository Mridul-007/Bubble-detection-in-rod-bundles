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

class CollisionDetector:
    """Detects and analyzes bubble collisions"""
    
    def __init__(self, collision_threshold_factor=1.5, min_frames_for_collision=2):
        self.collision_threshold_factor = collision_threshold_factor
        self.min_frames_for_collision = min_frames_for_collision
        self.collision_events = []
        self.potential_collisions = []
        
    def detect_collisions(self, bubbles, frame_name):
        """Detect potential collisions between bubbles"""
        collisions = []
        
        if len(bubbles) < 2:
            return collisions
            
        # Get bubble positions and sizes
        positions = np.array([(b['centroid_x'], b['centroid_y']) for b in bubbles])
        sizes = np.array([b['diameter_mm'] for b in bubbles])
        
        # Calculate distances between all bubble pairs
        distances = cdist(positions, positions)
        
        # Check each pair for potential collision
        for i in range(len(bubbles)):
            for j in range(i + 1, len(bubbles)):
                bubble1 = bubbles[i]
                bubble2 = bubbles[j]
                
                # Calculate collision threshold (sum of radii * factor)
                radius1 = bubble1['diameter_mm'] / 2
                radius2 = bubble2['diameter_mm'] / 2
                collision_threshold = (radius1 + radius2) * self.collision_threshold_factor
                
                # Convert threshold to pixels for distance comparison
                collision_threshold_px = collision_threshold * bubble1.get('px_per_mm', 1.0)
                distance_px = distances[i, j]
                
                if distance_px <= collision_threshold_px and distance_px > 0:
                    # Potential collision detected
                    collision_data = {
                        'frame': frame_name,
                        'bubble1_id': bubble1['bubble_id'],
                        'bubble2_id': bubble2['bubble_id'],
                        'distance_px': distance_px,
                        'distance_mm': distance_px / bubble1.get('px_per_mm', 1.0),
                        'collision_threshold_mm': collision_threshold,
                        'bubble1_diameter_mm': bubble1['diameter_mm'],
                        'bubble2_diameter_mm': bubble2['diameter_mm'],
                        'bubble1_velocity': bubble1.get('velocity_mm_per_s', 0),
                        'bubble2_velocity': bubble2.get('velocity_mm_per_s', 0),
                        'bubble1_x': bubble1['centroid_x'],
                        'bubble1_y': bubble1['centroid_y'],
                        'bubble2_x': bubble2['centroid_x'],
                        'bubble2_y': bubble2['centroid_y'],
                        'relative_velocity': abs(bubble1.get('velocity_mm_per_s', 0) - bubble2.get('velocity_mm_per_s', 0)),
                        'collision_probability': self.calculate_collision_probability(distance_px, collision_threshold_px, bubble1, bubble2)
                    }
                    collisions.append(collision_data)
        
        self.collision_events.extend(collisions)
        return collisions
    
    def calculate_collision_probability(self, distance_px, threshold_px, bubble1, bubble2):
        """Calculate collision probability based on distance and velocities"""
        # Basic probability based on distance (closer = higher probability)
        distance_prob = max(0, 1 - (distance_px / threshold_px))
        
        # Velocity factor (higher relative velocity = higher collision chance)
        v1 = bubble1.get('velocity_mm_per_s', 0)
        v2 = bubble2.get('velocity_mm_per_s', 0)
        velocity_factor = min(1, (abs(v1 - v2) / 100))  # Normalize to 0-1
        
        # Combined probability
        probability = (distance_prob * 0.7) + (velocity_factor * 0.3)
        return min(1.0, probability)
    
    def export_collision_data(self, output_path):
        """Export collision data to Excel"""
        if not self.collision_events:
            print("No collision events to export")
            return
            
        df = pd.DataFrame(self.collision_events)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # All collision events
            df.to_excel(writer, sheet_name='All_Collisions', index=False)
            
            # High probability collisions (>0.7)
            high_prob = df[df['collision_probability'] > 0.7]
            if not high_prob.empty:
                high_prob.to_excel(writer, sheet_name='High_Probability_Collisions', index=False)
            
            # Collision statistics per frame
            frame_stats = df.groupby('frame').agg({
                'collision_probability': ['count', 'mean', 'max'],
                'distance_mm': ['mean', 'min'],
                'relative_velocity': ['mean', 'max']
            }).round(4)
            
            frame_stats.columns = ['_'.join(col).strip() for col in frame_stats.columns]
            frame_stats.to_excel(writer, sheet_name='Frame_Statistics')
            
            # Bubble pair analysis
            df['bubble_pair'] = df['bubble1_id'].astype(str) + '_' + df['bubble2_id'].astype(str)
            pair_stats = df.groupby('bubble_pair').agg({
                'collision_probability': ['count', 'mean', 'max'],
                'distance_mm': ['mean', 'min'],
                'frame': ['first', 'last']
            }).round(4)
            
            pair_stats.columns = ['_'.join(col).strip() for col in pair_stats.columns]
            pair_stats.to_excel(writer, sheet_name='Bubble_Pair_Analysis')
        
        print(f"Collision data exported to: {output_path}")
        print(f"Total collision events: {len(self.collision_events)}")
        if len(self.collision_events) > 0:
            print(f"Average collision probability: {df['collision_probability'].mean():.3f}")
            print(f"High probability collisions (>0.7): {len(high_prob)}")

class BubbleTracker:
    """Tracks bubbles across frames and calculates velocities"""
    
    def __init__(self, px_per_mm=1.0, fps=6400, max_displacement_px=50):
        self.px_per_mm = px_per_mm
        self.fps = fps
        self.max_displacement_px = max_displacement_px
        self.previous_bubbles = []
        self.frame_count = 0
        self.all_tracked_data = []
        self.collision_detector = CollisionDetector()
        
    def track_bubbles(self, current_bubbles, frame_name):
        """Track bubbles between frames and calculate velocities"""
        tracked_bubbles = []
        
        if self.frame_count == 0:
            # First frame - no velocity calculation
            for bubble in current_bubbles:
                bubble['velocity_mm_per_s'] = 0
                bubble['frame'] = frame_name
                bubble['px_per_mm'] = self.px_per_mm
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
                    bubble['px_per_mm'] = self.px_per_mm
                    tracked_bubbles.append(bubble)
        
        # Detect collisions in current frame
        collisions = self.collision_detector.detect_collisions(tracked_bubbles, frame_name)
        
        self.previous_bubbles = current_bubbles.copy()
        self.frame_count += 1
        self.all_tracked_data.extend(tracked_bubbles)
        
        return tracked_bubbles, collisions
    
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
    """Enhanced bubble detector with velocity tracking and collision detection"""
    
    def __init__(self, min_bubble_area=20, max_bubble_area=50000, 
                 block_size=51  , c_value=2):  # Updated default values
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
    
    def process_video_frames_with_collision_detection(self, folder_path, file_pattern="*.tif", 
                                                    crop_coords=None, px_per_mm=1.0, fps=6400,
                                                    max_displacement_px=50, output_dir='collision_results'):
        """Process video frames and detect bubble collisions"""
        
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
        print("Processing frames for collision detection...")
        all_frame_collisions = []
        
        for i, image_path in enumerate(image_files):
            frame_name = Path(image_path).stem
            print(f"Processing frame {i+1}/{len(image_files)}: {frame_name}")
            
            try:
                # Detect bubbles
                bubble_data = self.detect_bubbles_simple(image_path, crop_coords)
                
                # Track bubbles and detect collisions
                tracked_bubbles, collisions = tracker.track_bubbles(bubble_data, frame_name)
                
                if collisions:
                    all_frame_collisions.extend(collisions)
                    print(f"  Found {len(collisions)} potential collisions")
                
                if i % 50 == 0:  # Progress update every 50 frames
                    print(f"  Detected {len(tracked_bubbles)} bubbles")
                    
            except Exception as e:
                print(f"Error processing frame {frame_name}: {e}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export bubble tracking data
        excel_path = output_path / f"bubble_velocity_tracking_{timestamp}.xlsx"
        tracker.export_to_excel(excel_path)
        
        # Export collision data
        collision_excel_path = output_path / f"bubble_collisions_{timestamp}.xlsx"
        tracker.collision_detector.export_collision_data(collision_excel_path)
        
        # Create visualizations
        self.create_velocity_summary_plot(tracker.all_tracked_data, 
                                        output_path / f"velocity_summary_{timestamp}.png")
        
        self.create_collision_analysis_plots(tracker.collision_detector.collision_events,
                                           output_path / f"collision_analysis_{timestamp}.png")
        
        self.create_collision_heatmap(tracker.collision_detector.collision_events,
                                    output_path / f"collision_heatmap_{timestamp}.png")
        
        return tracker.all_tracked_data, tracker.collision_detector.collision_events
    
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
    
    def create_collision_analysis_plots(self, collision_data, output_path):
        """Create collision analysis visualization"""
        if not collision_data:
            print("No collision data to visualize")
            return
            
        df = pd.DataFrame(collision_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Collision probability distribution
        axes[0,0].hist(df['collision_probability'], bins=30, alpha=0.7, color='red')
        axes[0,0].set_title('Collision Probability Distribution')
        axes[0,0].set_xlabel('Collision Probability')
        axes[0,0].set_ylabel('Count')
        
        # Distance vs collision probability
        axes[0,1].scatter(df['distance_mm'], df['collision_probability'], alpha=0.6, s=20)
        axes[0,1].set_title('Distance vs Collision Probability')
        axes[0,1].set_xlabel('Distance (mm)')
        axes[0,1].set_ylabel('Collision Probability')
        
        # Relative velocity vs collision probability
        axes[1,0].scatter(df['relative_velocity'], df['collision_probability'], alpha=0.6, s=20)
        axes[1,0].set_title('Relative Velocity vs Collision Probability')
        axes[1,0].set_xlabel('Relative Velocity (mm/s)')
        axes[1,0].set_ylabel('Collision Probability')
        
        # Collision events over time
        frame_collisions = df.groupby('frame').size()
        axes[1,1].plot(range(len(frame_collisions)), frame_collisions.values)
        axes[1,1].set_title('Collision Events Over Time')
        axes[1,1].set_xlabel('Frame Number')
        axes[1,1].set_ylabel('Number of Collision Events')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Collision analysis plot saved to: {output_path}")
    
    def create_collision_heatmap(self, collision_data, output_path):
        """Create collision position heatmap"""
        if not collision_data:
            return
            
        df = pd.DataFrame(collision_data)
        
        # Create heatmap of collision positions
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bubble 1 positions
        axes[0].hexbin(df['bubble1_x'], df['bubble1_y'], C=df['collision_probability'], 
                      gridsize=20, cmap='Reds', reduce_C_function=np.mean)
        axes[0].set_title('Collision Hotspots - Bubble 1 Positions')
        axes[0].set_xlabel('X Position (pixels)')
        axes[0].set_ylabel('Y Position (pixels)')
        
        # Bubble 2 positions
        im = axes[1].hexbin(df['bubble2_x'], df['bubble2_y'], C=df['collision_probability'], 
                           gridsize=20, cmap='Reds', reduce_C_function=np.mean)
        axes[1].set_title('Collision Hotspots - Bubble 2 Positions')
        axes[1].set_xlabel('X Position (pixels)')
        axes[1].set_ylabel('Y Position (pixels)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1], label='Average Collision Probability')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Collision heatmap saved to: {output_path}")

def get_collision_detection_inputs():
    """Get user inputs for collision detection"""
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
    """Main function for collision detection"""
    print("=== BUBBLE COLLISION DETECTION SYSTEM ===")
    print("Features:")
    print("- Track bubbles across video frames")
    print("- Detect potential bubble collisions")
    print("- Calculate collision probabilities")
    print("- Export collision data to Excel")
    print("- Generate collision analysis plots")
    print("- Block size: 53, C value: 2")
    print()
    
    inputs = get_collision_detection_inputs()
    if inputs is None:
        print("Operation cancelled.")
        return
    
    try:
        # Initialize detector with specified parameters
        detector = EnhancedBubbleDetector(block_size=53, c_value=2)
        
        print(f"Processing parameters:")
        print(f"- Folder: {inputs['folder_path']}")
        print(f"- Calibration: {inputs['px_per_mm']} pixels/mm")
        print(f"- Frame rate: {inputs['fps']} fps")
        print(f"- Max displacement: {inputs['max_displacement']} pixels")
        print(f"- Block size: 53, C value: 2")
        
        # Process video frames
        bubble_data, collision_data = detector.process_video_frames_with_collision_detection(
            folder_path=inputs['folder_path'],
            crop_coords=inputs['crop_coords'],
            px_per_mm=inputs['px_per_mm'],
            fps=inputs['fps'],
            max_displacement_px=inputs['max_displacement']
        )
        
        print(f"\n=== COLLISION DETECTION COMPLETED ===")
        print("Generated files:")
        print("- bubble_velocity_tracking_*.xlsx: Bubble tracking data")
        print("- bubble_collisions_*.xlsx: Collision detection data")
        print("- velocity_summary_*.png: Velocity analysis plots")
        print("- collision_analysis_*.png: Collision analysis plots")
        print("- collision_heatmap_*.png: Collision position heatmap")
        
        if collision_data:
            df_collisions = pd.DataFrame(collision_data)
            print(f"\nCollision Statistics:")
            print(f"- Total collision events: {len(collision_data)}")
            print(f"- High probability collisions (>0.7): {len(df_collisions[df_collisions['collision_probability'] > 0.7])}")
            print(f"- Average collision probability: {df_collisions['collision_probability'].mean():.3f}")
            print(f"- Average distance between colliding bubbles: {df_collisions['distance_mm'].mean():.3f} mm")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()