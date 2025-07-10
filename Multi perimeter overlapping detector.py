import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.measure import regionprops, label
from skimage.morphology import closing, disk
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import json

class AdvancedCircleDetector:
    def __init__(self):
        self.default_params = {
            'gaussian_blur': 3,
            'clahe_clip_limit': 2.0,
            'canny_low': 50,
            'canny_high': 150,
            'hough_dp': 1,
            'hough_min_dist': 30,
            'hough_param1': 50,
            'hough_param2': 30,
            'min_radius': 10,
            'max_radius': 200,
            'circularity_threshold': 0.5,  # Increased from 0.3
            'dbscan_eps': 20,
            'dbscan_min_samples': 2,
            'confidence_threshold': 0.6,   # Increased from 0.5
            'enable_multi_param': True,    # New toggle
            'enable_partial_recovery': True, # New toggle
            'partial_strictness': 0.7      # New parameter for partial circle validation
        }
        self.current_params = self.default_params.copy()
        self.image = None
        self.original_image = None
        self.detected_circles = []
        
    def select_image(self):
        """Let user select an image file"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load the image file")
                return False
            
            # Convert to RGB for display
            self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            print(f"Image loaded successfully: {file_path}")
            print(f"Image dimensions: {self.image.shape}")
            return True
        return False
    
    def configure_parameters(self):
        """Allow user to modify detection parameters"""
        root = tk.Tk()
        root.withdraw()
        
        response = messagebox.askyesno(
            "Parameter Configuration",
            "Would you like to customize detection parameters?\n\n"
            "Select 'Yes' to modify parameters or 'No' to use defaults."
        )
        
        if response:
            print("\nCurrent parameters:")
            for key, value in self.current_params.items():
                print(f"{key}: {value}")
            
            print("\nEnter new values (press Enter to keep current value):")
            
            for key, current_value in self.current_params.items():
                try:
                    user_input = input(f"{key} (current: {current_value}): ").strip()
                    if user_input:
                        if isinstance(current_value, int):
                            self.current_params[key] = int(user_input)
                        elif isinstance(current_value, float):
                            self.current_params[key] = float(user_input)
                except ValueError:
                    print(f"Invalid value for {key}, keeping current value")
            
            print("\nUpdated parameters:")
            for key, value in self.current_params.items():
                print(f"{key}: {value}")
    
    def preprocess_image(self, image):
        """Level 1: Advanced preprocessing for better circle detection"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=self.current_params['clahe_clip_limit'])
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, 
                                 (self.current_params['gaussian_blur'], 
                                  self.current_params['gaussian_blur']), 0)
        
        return blurred
    
    def validate_circle_quality(self, image, x, y, r):
        """Validate if a detected circle is actually a good circle"""
        # Check if circle is within image bounds
        h, w = image.shape[:2]
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return False
        
        # Create a mask for the circle
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, 2)
        
        # Check edge density along the circle perimeter
        edges = cv2.Canny(image, 50, 150)
        circle_edges = cv2.bitwise_and(edges, mask)
        
        # Calculate what percentage of the circle perimeter has edges
        perimeter_pixels = 2 * np.pi * r
        edge_pixels = np.sum(circle_edges > 0)
        edge_ratio = edge_pixels / perimeter_pixels if perimeter_pixels > 0 else 0
        
        # Good circles should have at least 15% of their perimeter as edges
        return edge_ratio > 0.15
    
    def calculate_circularity(self, contour):
        """Calculate circularity score for a contour"""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity
        """Calculate circularity score for a contour"""
        area = cv2.contourArea(contour)
        if area == 0:
            return 0
        
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity
    
    def basic_hough_detection(self, processed_image):
        """Level 2: Basic Hough Circle Transform"""
        circles = cv2.HoughCircles(
            processed_image,
            cv2.HOUGH_GRADIENT,
            dp=self.current_params['hough_dp'],
            minDist=self.current_params['hough_min_dist'],
            param1=self.current_params['hough_param1'],
            param2=self.current_params['hough_param2'],
            minRadius=self.current_params['min_radius'],
            maxRadius=self.current_params['max_radius']
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return np.array([])
    
    def multi_parameter_detection(self, processed_image):
        """Level 2: Multiple parameter sets for better detection"""
        all_circles = []
        
        # More conservative parameter variations to reduce false positives
        param_sets = [
            # Standard detection (higher thresholds)
            {'param1': 60, 'param2': 40, 'dp': 1},
            # Slightly more sensitive (but still conservative)
            {'param1': 50, 'param2': 35, 'dp': 1},
            # Different resolution (with higher thresholds)
            {'param1': 70, 'param2': 45, 'dp': 2},
        ]
        
        for params in param_sets:
            circles = cv2.HoughCircles(
                processed_image,
                cv2.HOUGH_GRADIENT,
                dp=params['dp'],
                minDist=int(self.current_params['hough_min_dist'] * 1.2),  # Increase min distance
                param1=params['param1'],
                param2=params['param2'],
                minRadius=self.current_params['min_radius'],
                maxRadius=self.current_params['max_radius']
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                # Add validation - check circle quality
                valid_circles = []
                for circle in circles:
                    x, y, r = circle
                    if self.validate_circle_quality(processed_image, x, y, r):
                        valid_circles.append(circle)
                all_circles.extend(valid_circles)
        
        return np.array(all_circles) if all_circles else np.array([])
    
    def contour_based_detection(self, processed_image):
        """Level 3: Contour analysis with circularity scoring"""
        # Edge detection
        edges = cv2.Canny(processed_image, 
                         self.current_params['canny_low'], 
                         self.current_params['canny_high'])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circle_candidates = []
        
        for contour in contours:
            # Filter by area
            area = cv2.contourArea(contour)
            if area < 100:  # Skip very small contours
                continue
            
            # Calculate circularity
            circularity = self.calculate_circularity(contour)
            
            if circularity >= self.current_params['circularity_threshold']:
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Validate radius
                if (self.current_params['min_radius'] <= radius <= 
                    self.current_params['max_radius']):
                    
                    circle_candidates.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'circularity': circularity,
                        'confidence': circularity,
                        'method': 'contour'
                    })
        
        return circle_candidates
    
    def cluster_circles(self, circles):
        """Level 4: Cluster nearby detections using DBSCAN"""
        if len(circles) == 0:
            return []
        
        # Prepare data for clustering (center coordinates)
        if isinstance(circles[0], dict):
            points = np.array([c['center'] for c in circles])
        else:
            points = circles[:, :2]  # x, y coordinates
        
        # DBSCAN clustering
        clustering = DBSCAN(
            eps=self.current_params['dbscan_eps'],
            min_samples=self.current_params['dbscan_min_samples']
        ).fit(points)
        
        clustered_circles = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get all circles in this cluster
            cluster_mask = clustering.labels_ == label
            if isinstance(circles[0], dict):
                cluster_circles = [circles[i] for i in range(len(circles)) if cluster_mask[i]]
                # Choose the one with highest confidence
                best_circle = max(cluster_circles, key=lambda x: x.get('confidence', 0))
            else:
                cluster_circles = circles[cluster_mask]
                # Choose the median circle
                median_idx = len(cluster_circles) // 2
                best_circle = {
                    'center': (int(cluster_circles[median_idx][0]), int(cluster_circles[median_idx][1])),
                    'radius': int(cluster_circles[median_idx][2]),
                    'confidence': 0.7,
                    'method': 'hough'
                }
            
            clustered_circles.append(best_circle)
        
        return clustered_circles
    
    def partial_circle_recovery(self, processed_image):
        """Level 5: Detect partial circles from curved segments - More Conservative"""
        # More conservative edge detection for partial circles
        edges = cv2.Canny(processed_image, 40, 120)  # Higher thresholds
        
        # Less aggressive dilation to avoid connecting unrelated segments
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        partial_circles = []
        
        for contour in contours:
            if len(contour) < 15:  # Need more points for reliable fitting
                continue
            
            # Filter by contour area first
            area = cv2.contourArea(contour)
            if area < 200:  # Minimum area threshold
                continue
            
            # Fit circle to contour points
            if len(contour) >= 5:
                try:
                    # Fit ellipse (which includes circles)
                    ellipse = cv2.fitEllipse(contour)
                    center, axes, angle = ellipse
                    
                    # Stricter circularity check
                    aspect_ratio = max(axes) / min(axes)
                    if aspect_ratio < 1.5:  # More strict circular requirement
                        radius = np.mean(axes) / 2
                        
                        if (self.current_params['min_radius'] * 0.8 <= radius <= 
                            self.current_params['max_radius'] * 1.2):
                            
                            # More conservative visibility estimation
                            perimeter = cv2.arcLength(contour, True)
                            expected_perimeter = 2 * np.pi * radius
                            visibility = min(perimeter / expected_perimeter, 1.0)
                            
                            # Higher visibility threshold and additional validation
                            if visibility > 0.4:  # At least 40% visible
                                # Additional validation: check if the fitted circle makes sense
                                if self.validate_partial_circle(processed_image, center, radius, contour):
                                    partial_circles.append({
                                        'center': (int(center[0]), int(center[1])),
                                        'radius': int(radius),
                                        'confidence': visibility * 0.6,  # Lower confidence for partial
                                        'visibility': visibility,
                                        'method': 'partial'
                                    })
                except:
                    continue
        
        return partial_circles
    
    def multi_scale_detection(self, image):
        """Detect circles at multiple scales"""
        all_detections = []
        scales = [0.5, 1.0, 1.5]
        
        for scale in scales:
            if scale != 1.0:
                height, width = image.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                scaled_image = cv2.resize(image, (new_width, new_height))
            else:
                scaled_image = image
            
            processed = self.preprocess_image(scaled_image)
            
            # Basic detection
            hough_circles = self.basic_hough_detection(processed)
            
            # Scale back coordinates
            if scale != 1.0:
                for circle in hough_circles:
                    circle[0] = int(circle[0] / scale)  # x
                    circle[1] = int(circle[1] / scale)  # y
                    circle[2] = int(circle[2] / scale)  # radius
            
            # Convert to dict format
            for circle in hough_circles:
                all_detections.append({
                    'center': (circle[0], circle[1]),
                    'radius': circle[2],
                    'confidence': 0.6,
                    'method': f'multi_scale_{scale}'
                })
        
        return all_detections
    
    def detect_circles(self):
        """Main detection pipeline combining all techniques"""
        if self.image is None:
            print("No image loaded!")
            return
        
        print("Starting circle detection...")
        all_detections = []
        
        # Preprocess image
        processed = self.preprocess_image(self.image)
        
        # Level 1: Basic Hough detection
        print("Level 1: Basic Hough detection...")
        basic_circles = self.basic_hough_detection(processed)
        for circle in basic_circles:
            all_detections.append({
                'center': (circle[0], circle[1]),
                'radius': circle[2],
                'confidence': 0.7,
                'method': 'basic_hough'
            })
        
        # Level 2: Multi-parameter detection (optional)
        if self.current_params.get('enable_multi_param', True):
            print("Level 2: Multi-parameter detection...")
            multi_circles = self.multi_parameter_detection(processed)
            for circle in multi_circles:
                all_detections.append({
                    'center': (circle[0], circle[1]),
                    'radius': circle[2],
                    'confidence': 0.6,
                    'method': 'multi_param'
                })
        
        # Level 3: Contour-based detection
        print("Level 3: Contour analysis...")
        contour_circles = self.contour_based_detection(processed)
        all_detections.extend(contour_circles)
        
        # Level 4: Multi-scale detection
        print("Level 4: Multi-scale detection...")
        scale_circles = self.multi_scale_detection(self.image)
        all_detections.extend(scale_circles)
        
        # Level 5: Partial circle recovery (optional)
        if self.current_params.get('enable_partial_recovery', True):
            print("Level 5: Partial circle recovery...")
            partial_circles = self.partial_circle_recovery(processed)
            all_detections.extend(partial_circles)
        
        # Clustering to remove duplicates
        print("Clustering and filtering...")
        self.detected_circles = self.cluster_circles(all_detections)
        
        # Filter by confidence threshold
        self.detected_circles = [
            circle for circle in self.detected_circles 
            if circle.get('confidence', 0) >= self.current_params['confidence_threshold']
        ]
        
        print(f"Detection complete! Found {len(self.detected_circles)} circles")
        
        # Sort by confidence
        self.detected_circles.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return self.detected_circles
    
    def visualize_results(self):
        """Display the results with detected circles"""
        if self.image is None or not self.detected_circles:
            print("No results to display")
            return
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(self.image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Results
        result_image = self.image.copy()
        
        # Draw circles with different colors based on detection method
        method_colors = {
            'basic_hough': (255, 0, 0),      # Red
            'multi_param': (0, 255, 0),      # Green
            'contour': (0, 0, 255),          # Blue
            'multi_scale_0.5': (255, 255, 0), # Yellow
            'multi_scale_1.0': (255, 0, 255), # Magenta
            'multi_scale_1.5': (0, 255, 255), # Cyan
            'partial': (128, 0, 128)         # Purple
        }
        
        for i, circle in enumerate(self.detected_circles):
            center = circle['center']
            radius = circle['radius']
            method = circle.get('method', 'unknown')
            confidence = circle.get('confidence', 0)
            
            # Get color for method
            color = method_colors.get(method, (255, 255, 255))
            
            # Draw circle
            cv2.circle(result_image, center, radius, color, 2)
            
            # Draw center
            cv2.circle(result_image, center, 3, color, -1)
            
            # Add text with confidence
            text = f"{i+1}: {confidence:.2f}"
            cv2.putText(result_image, text, 
                       (center[0]-20, center[1]-radius-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        axes[1].imshow(result_image)
        axes[1].set_title(f"Detected Circles: {len(self.detected_circles)}")
        axes[1].axis('off')
        
        # Add legend
        legend_text = []
        for method, color in method_colors.items():
            count = sum(1 for c in self.detected_circles if c.get('method', '').startswith(method))
            if count > 0:
                legend_text.append(f"{method}: {count}")
        
        plt.figtext(0.02, 0.02, "\n".join(legend_text), fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 50)
        for i, circle in enumerate(self.detected_circles):
            print(f"Circle {i+1}:")
            print(f"  Center: {circle['center']}")
            print(f"  Radius: {circle['radius']}")
            print(f"  Confidence: {circle.get('confidence', 0):.3f}")
            print(f"  Method: {circle.get('method', 'unknown')}")
            if 'visibility' in circle:
                print(f"  Visibility: {circle['visibility']:.3f}")
            print()
    
    def save_results(self, filename=None):
        """Save detection results to JSON file"""
        if not filename:
            filename = "circle_detection_results.json"
        
        results = {
            'parameters': self.current_params,
            'total_circles': len(self.detected_circles),
            'circles': self.detected_circles
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def run(self):
        """Main application loop"""
        print("Advanced Circle Detection System")
        print("=" * 40)
        
        # Select image
        if not self.select_image():
            print("No image selected. Exiting.")
            return
        
        # Configure parameters
        self.configure_parameters()
        
        # Detect circles
        circles = self.detect_circles()
        
        if circles:
            # Visualize results
            self.visualize_results()
            
            # Ask to save results
            save_results = input("\nSave results to JSON file? (y/n): ").lower().strip()
            if save_results == 'y':
                filename = input("Enter filename (or press Enter for default): ").strip()
                self.save_results(filename if filename else None)
        else:
            print("No circles detected. Try adjusting parameters.")

# Main execution
if __name__ == "__main__":
    detector = AdvancedCircleDetector()
    detector.run()