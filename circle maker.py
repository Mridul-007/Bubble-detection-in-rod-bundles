from PIL import Image, ImageDraw
import random
import math

# Get user input
width = int(input("Enter image width (pixels): ") or 800)
height = int(input("Enter image height (pixels): ") or 600)
num_circles = int(input("Number of circles: ") or 20)
output_filename = input("Output filename (e.g., circles.png): ") or "circles.png"
color_choice = input("Color scheme (1: black bubbles/white bg, 2: white bubbles/black bg): ") or "1"

# Create image with background color
bg_color = (255, 255, 255) if color_choice == "1" else (0, 0, 0)
bubble_color = (0, 0, 0) if color_choice == "1" else (255, 255, 255)
image = Image.new('RGB', (width, height), bg_color)
draw = ImageDraw.Draw(image)

# Calculate uniform circle size
max_radius = min(width, height) // 15
min_radius = int(max_radius * 0.8)
radius = random.randint(min_radius, max_radius)

# Store placed circles to prevent overlap
placed_circles = []
max_attempts = 500  # Max attempts to place a circle without overlap

def circles_overlap(x1, y1, r1, x2, y2, r2):
    """Check if two circles overlap significantly"""
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance < (r1 + r2) * 0.9  # 10% gap between circles

for _ in range(num_circles):
    placed = False
    attempts = 0
    
    while not placed and attempts < max_attempts:
        attempts += 1
        # Random position with margin
        x = random.randint(radius, width - radius)
        y = random.randint(radius, height - radius)
        
        # Check overlap with existing circles
        overlap = False
        for (px, py, pr) in placed_circles:
            if circles_overlap(x, y, radius, px, py, pr):
                overlap = True
                break
                
        if not overlap:
            # Draw circle
            bbox = (x - radius, y - radius, x + radius, y + radius)
            draw.ellipse(bbox, fill=bubble_color)
            placed_circles.append((x, y, radius))
            placed = True

# Save and show
image.save(output_filename)
print(f"Created {len(placed_circles)} circles out of requested {num_circles}")
print(f"Circle radius: {radius}px")
image.show()