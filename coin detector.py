import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class CoinAnalyzer:
    def __init__(self):
        # Default calibration value (pixels per mm)
        self.px_per_mm = 10.0
        self.image_path = ""
        self.original_image = None
        self.processed_image = None
        self.coin_data = []
        
        # Detection parameters
        self.min_coin_diameter = 15.0    # mm
        self.max_coin_diameter = 30.0    # mm
        self.param1 = 100                # Hough parameter
        self.param2 = 30                 # Hough parameter

    def load_image(self, path):
        """Load and prepare the image for processing"""
        try:
            self.image_path = path
            self.original_image = cv2.imread(path)
            if self.original_image is None:
                raise ValueError(f"Could not read image from {path}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            return False

    def calibrate(self, px_per_mm):
        """Set the calibration value"""
        try:
            self.px_per_mm = float(px_per_mm)
            return True
        except ValueError:
            messagebox.showerror("Error", "Invalid calibration value. Please enter a number.")
            return False

    def detect_coins(self):
        """Detect coins using Hough Circle Transform"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return False
        
        return self._detect_with_hough()

    def _detect_with_hough(self):
        """Detect coins using Hough Circle Transform"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            
            # Calculate min/max radius based on calibration
            min_radius = int((self.min_coin_diameter * self.px_per_mm) / 2 * 0.8)
            max_radius = int((self.max_coin_diameter * self.px_per_mm) / 2 * 1.2)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # Detect circles
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=int(self.min_coin_diameter * self.px_per_mm * 1.5),
                param1=self.param1,
                param2=self.param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            
            # Process detected circles
            self.coin_data = []
            self.processed_image = self.original_image.copy()
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for i, (x, y, r) in enumerate(circles):
                    # Calculate areas in pixels
                    area_px = math.pi * (r ** 2)
                    diameter_px = 2 * r
                    
                    # Draw on image
                    cv2.circle(self.processed_image, (x, y), r, (0, 255, 0), 4)
                    cv2.circle(self.processed_image, (x, y), 2, (0, 0, 255), 3)
                    cv2.putText(self.processed_image, str(i+1), (x-10, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    # Store data
                    self.coin_data.append({
                        'id': i+1,
                        'x': x,
                        'y': y,
                        'radius_px': r,
                        'diameter_px': diameter_px,
                        'area_px': area_px,
                        'diameter_mm': None,
                        'area_mm': None,
                        'px_per_mm': None
                    })
            return True
        except Exception as e:
            messagebox.showerror("Detection Error", f"Hough Circle detection failed: {str(e)}")
            return False

    def get_user_input_for_calibration(self):
        """Get user input for actual coin diameters to calculate px/mm"""
        if not self.coin_data:
            return False
            
        for coin in self.coin_data:
            # Ask user for actual diameter of this coin
            actual_diameter = simpledialog.askfloat(
                "Coin Diameter for Calibration",
                f"Enter actual diameter (mm) for coin {coin['id']}:\n"
                f"Detected diameter: {coin['diameter_px']:.2f} px",
                parent=self.root,
                minvalue=1.0,
                maxvalue=100.0
            )
            
            if actual_diameter is None:  # User canceled
                return False
                
            # Store user input and calculate px/mm for this coin
            coin['diameter_mm'] = actual_diameter
            coin['area_mm'] = math.pi * (actual_diameter / 2) ** 2
            coin['px_per_mm'] = coin['diameter_px'] / actual_diameter
            
        return True

    def calculate_with_calibration(self, px_per_mm):
        """Calculate measurements using provided calibration"""
        if not self.coin_data:
            return False
            
        for coin in self.coin_data:
            # Calculate diameter and area in mm using calibration
            coin['diameter_mm_cal'] = coin['diameter_px'] / px_per_mm
            coin['area_mm_cal'] = coin['area_px'] / (px_per_mm ** 2)
            
        return True

    def get_pixel_results(self):
        """Return formatted results for pixel measurements"""
        if not self.coin_data:
            return "No coins detected"
        
        results = "Coin Pixel Measurements & Calibration:\n"
        results += "=" * 85 + "\n"
        results += f"{'ID':<5}{'Diameter (px)':<15}{'Area (px)':<15}{'Diameter (mm)':<15}{'Area (mm²)':<15}{'Px/mm':<15}\n"
        results += "-" * 85 + "\n"
        
        for coin in self.coin_data:
            diam_mm = coin.get('diameter_mm', 'N/A')
            area_mm = coin.get('area_mm', 'N/A')
            px_per_mm = coin.get('px_per_mm', 'N/A')
            
            results += (f"{coin['id']:<5}{coin['diameter_px']:<15.2f}{coin['area_px']:<15.2f}"
                        f"{diam_mm if diam_mm == 'N/A' else f'{diam_mm:.2f}':<15}"
                        f"{area_mm if area_mm == 'N/A' else f'{area_mm:.2f}':<15}"
                        f"{px_per_mm if px_per_mm == 'N/A' else f'{px_per_mm:.2f}':<15}\n")
        
        results += "=" * 85 + "\n"
        results += f"Total coins detected: {len(self.coin_data)}\n"
        
        # Calculate average px/mm if available
        px_per_mm_values = [coin['px_per_mm'] for coin in self.coin_data if coin.get('px_per_mm') is not None]
        if px_per_mm_values:
            avg_px_per_mm = np.mean(px_per_mm_values)
            std_px_per_mm = np.std(px_per_mm_values)
            results += f"Average px/mm: {avg_px_per_mm:.2f} ± {std_px_per_mm:.2f}"
        
        return results

    def get_calibration_results(self, px_per_mm):
        """Return formatted results for calibration-based measurements"""
        if not self.coin_data:
            return "No coins detected"
        
        results = f"Coin Measurements with Calibration ({px_per_mm} px/mm):\n"
        results += "=" * 70 + "\n"
        results += f"{'ID':<5}{'Diameter (px)':<15}{'Diameter (mm)':<15}{'Area (px)':<15}{'Area (mm²)':<15}\n"
        results += "-" * 70 + "\n"
        
        for coin in self.coin_data:
            diam_mm = coin.get('diameter_mm_cal', 'N/A')
            area_mm = coin.get('area_mm_cal', 'N/A')
            
            results += (f"{coin['id']:<5}{coin['diameter_px']:<15.2f}"
                        f"{diam_mm if diam_mm == 'N/A' else f'{diam_mm:.2f}':<15}"
                        f"{coin['area_px']:<15.2f}"
                        f"{area_mm if area_mm == 'N/A' else f'{area_mm:.2f}':<15}\n")
        
        results += "=" * 70 + "\n"
        results += f"Total coins detected: {len(self.coin_data)}"
        
        return results

class CoinAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dual Panel Coin Measurement Analyzer")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize analyzer
        self.analyzer = CoinAnalyzer()
        self.analyzer.root = root  # Reference for dialogs
        
        # Create GUI elements
        self.create_widgets()
        
    def create_widgets(self):
        # Configure style
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Panel.TLabel', font=('Arial', 11, 'bold'), foreground='#2c3e50')
        
        # Create main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create image frame (left side)
        image_frame = ttk.Frame(main_container)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        
        # Create controls frame (right side)
        controls_frame = ttk.Frame(main_container)
        controls_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === IMAGE SECTION ===
        ttk.Label(image_frame, text="Image Display", style='Header.TLabel').pack(pady=(0,10))
        
        # Image load button
        ttk.Button(image_frame, text="Load Image", command=self.load_image).pack(pady=5)
        
        # Detect coins button
        ttk.Button(image_frame, text="Detect Coins", command=self.detect_coins).pack(pady=5)
        
        # Image display
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # === CONTROLS SECTION ===
        # Panel 1: Pixel Measurements & Calibration Calculation
        panel1_frame = ttk.LabelFrame(controls_frame, text="Panel 1: Pixel Measurements & Calibration", padding=10)
        panel1_frame.pack(fill=tk.BOTH, expand=True, pady=(0,10))
        
        ttk.Label(panel1_frame, text="Get pixel measurements and calculate px/mm", style='Panel.TLabel').pack(pady=(0,10))
        
        ttk.Button(panel1_frame, text="Input Actual Diameters", command=self.get_calibration_input).pack(pady=5, fill=tk.X)
        
        self.results_text1 = tk.Text(panel1_frame, width=50, height=15, font=('Consolas', 9))
        self.results_text1.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Panel 2: User Calibration Measurements
        panel2_frame = ttk.LabelFrame(controls_frame, text="Panel 2: User Calibration Measurements", padding=10)
        panel2_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(panel2_frame, text="Use custom calibration for measurements", style='Panel.TLabel').pack(pady=(0,10))
        
        # Calibration input
        calib_frame = ttk.Frame(panel2_frame)
        calib_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(calib_frame, text="Calibration (px/mm):").pack(side=tk.LEFT)
        self.calibration_entry = ttk.Entry(calib_frame, width=10)
        self.calibration_entry.insert(0, "10.0")
        self.calibration_entry.pack(side=tk.LEFT, padx=(5,0))
        
        ttk.Button(panel2_frame, text="Calculate with Calibration", command=self.calculate_with_calibration).pack(pady=10, fill=tk.X)
        
        self.results_text2 = tk.Text(panel2_frame, width=50, height=15, font=('Consolas', 9))
        self.results_text2.pack(pady=10, fill=tk.BOTH, expand=True)
        
    def load_image(self):
        """Load an image from file"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            if self.analyzer.load_image(file_path):
                self.show_image(self.analyzer.original_image)
                # Clear previous results
                self.clear_results()
    
    def detect_coins(self):
        """Detect coins"""
        if self.analyzer.detect_coins():
            self.show_image(self.analyzer.processed_image)
            self.show_pixel_results()
    
    def get_calibration_input(self):
        """Get user input for calibration calculation"""
        if self.analyzer.coin_data:
            if self.analyzer.get_user_input_for_calibration():
                self.show_pixel_results()
        else:
            messagebox.showwarning("Warning", "Please detect coins first")
    
    def calculate_with_calibration(self):
        """Calculate measurements using user-set calibration"""
        if not self.analyzer.coin_data:
            messagebox.showwarning("Warning", "Please detect coins first")
            return
            
        try:
            px_per_mm = float(self.calibration_entry.get())
            if self.analyzer.calculate_with_calibration(px_per_mm):
                self.show_calibration_results(px_per_mm)
        except ValueError:
            messagebox.showerror("Error", "Invalid calibration value. Please enter a number.")
    
    def show_image(self, image):
        """Display image in the GUI"""
        # Convert to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Resize image to fit the window
        max_width = 600
        max_height = 400
        
        width_ratio = max_width / pil_image.width
        height_ratio = max_height / pil_image.height
        scale = min(width_ratio, height_ratio, 1.0)
        
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        resized_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Display the image
        tk_image = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image
    
    def show_pixel_results(self):
        """Display pixel measurement results in panel 1"""
        results = self.analyzer.get_pixel_results()
        self.results_text1.config(state=tk.NORMAL)
        self.results_text1.delete(1.0, tk.END)
        self.results_text1.insert(tk.END, results)
        self.results_text1.config(state=tk.DISABLED)
    
    def show_calibration_results(self, px_per_mm):
        """Display calibration-based results in panel 2"""
        results = self.analyzer.get_calibration_results(px_per_mm)
        self.results_text2.config(state=tk.NORMAL)
        self.results_text2.delete(1.0, tk.END)
        self.results_text2.insert(tk.END, results)
        self.results_text2.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear both result panels"""
        self.results_text1.config(state=tk.NORMAL)
        self.results_text1.delete(1.0, tk.END)
        self.results_text1.config(state=tk.DISABLED)
        
        self.results_text2.config(state=tk.NORMAL)
        self.results_text2.delete(1.0, tk.END)
        self.results_text2.config(state=tk.DISABLED)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CoinAnalyzerApp(root)
    root.mainloop()