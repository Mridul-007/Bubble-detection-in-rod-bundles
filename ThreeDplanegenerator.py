import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import openpyxl

def create_output_folder():
    """Create output3d folder if it doesn't exist."""
    output_dir = "output3d"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_spheres(n=100, box_size=2000, radii=[20, 40, 60, 80, 100], seed=None):
    """Generate n spheres within the cuboid that don't extend outside boundaries."""
    if seed is not None:
        np.random.seed(seed)
    
    spheres = []
    attempts = 0
    max_attempts = n * 100  # Prevent infinite loops
    
    while len(spheres) < n and attempts < max_attempts:
        attempts += 1
        
        # Sample random center and radius
        x, y, z = np.random.uniform(0, box_size, size=3)
        r = np.random.choice(radii)
        
        # Check if sphere fits completely inside cuboid
        if (x - r >= 0) and (x + r <= box_size) and \
           (y - r >= 0) and (y + r <= box_size) and \
           (z - r >= 0) and (z + r <= box_size):
            
            diameter = 2 * r
            volume = (4/3) * np.pi * r**3
            surface_area = 4 * np.pi * r**2
            spheres.append([x, y, z, r, diameter, volume, surface_area])
    
    if len(spheres) < n:
        print(f"Warning: Only generated {len(spheres)} spheres out of {n} requested")
    
    return np.array(spheres)

def create_3d_visualization(spheres, output_dir, box_size=2000):
    """Create 3D visualization of the spheres in the cuboid."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define radius to color mapping
    radius_colors = {
        20: 'red',
        40: 'blue',
        60: 'green',
        80: 'orange',
        100: 'purple'
    }
    
    # Draw cuboid wireframe
    # Define the 8 vertices of the cube
    vertices = np.array([
        [0, 0, 0], [box_size, 0, 0], [box_size, box_size, 0], [0, box_size, 0],  # bottom face
        [0, 0, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [0, box_size, box_size]  # top face
    ])
    
    # Define the 12 edges of the cube
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Draw the edges
    for edge in edges:
        points = vertices[edge]
        ax.plot3D(*points.T, 'k-', alpha=0.6, linewidth=1)
    
    # Draw spheres
    for sphere in spheres:
        x, y, z, r = sphere[0], sphere[1], sphere[2], sphere[3]
        r_int = int(round(r))
        color = radius_colors.get(r_int, 'gray')  # Default to gray if radius not found
        
        # Create sphere surface
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        sphere_x = r * np.outer(np.cos(u), np.sin(v)) + x
        sphere_y = r * np.outer(np.sin(u), np.sin(v)) + y
        sphere_z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
        
        ax.plot_surface(sphere_x, sphere_y, sphere_z, alpha=0.6, color=color)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of Spheres in Cuboid')
    
    # Set equal aspect ratio
    ax.set_xlim([0, box_size])
    ax.set_ylim([0, box_size])
    ax.set_zlim([0, box_size])
    
    plt.savefig(os.path.join(output_dir, '3d_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D visualization saved to: {os.path.join(output_dir, '3d_visualization.png')}")

def project_spheres_to_face(spheres, face):
    """Project 3D spheres onto a 2D face plane."""
    projections = []
    
    for sphere in spheres:
        x, y, z, r = sphere[0], sphere[1], sphere[2], sphere[3]
        
        # For orthographic projection, we project ALL spheres
        # The visibility is determined by the projection coordinates
        if face == "+X":  # Looking along positive X axis (seeing Y-Z plane)
            proj_x, proj_y = y, z
        elif face == "-X":  # Looking along negative X axis (seeing Y-Z plane)
            proj_x, proj_y = y, z
        elif face == "+Y":  # Looking along positive Y axis (seeing X-Z plane)
            proj_x, proj_y = x, z
        elif face == "-Y":  # Looking along negative Y axis (seeing X-Z plane)
            proj_x, proj_y = x, z
        elif face == "+Z":  # Looking along positive Z axis (seeing X-Y plane)
            proj_x, proj_y = x, y
        elif face == "-Z":  # Looking along negative Z axis (seeing X-Y plane)
            proj_x, proj_y = x, y
        
        # All spheres are visible in orthographic projection
        projections.append([proj_x, proj_y, r])
    
    return projections

def render_face_view(projections, view_name, output_dir, filled=True, image_size=2000, box_size=2000):
    """Render orthographic view of projected spheres on a face with option for filled/unfilled."""
    # Create image with higher resolution for better quality
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    
    # Scale factor to map from box coordinates to image pixels
    scale = image_size / box_size
    
    # Draw border of the face
    border_color = "black"
    draw.rectangle([0, 0, image_size-1, image_size-1], outline=border_color, width=3)
    
    # Draw each projected sphere
    for proj_x, proj_y, r in projections:
        # Convert to image coordinates
        center_x = proj_x * scale
        center_y = proj_y * scale
        radius_px = r * scale
        
        # Draw circle
        bbox = [
            center_x - radius_px,
            center_y - radius_px,
            center_x + radius_px,
            center_y + radius_px
        ]
        
        if filled:
            # Color based on radius for better visualization
            colors = {
                20: "#FFE6E6", 
                40: "#E6F3FF", 
                60: "#E6FFE6", 
                80: "#FFFFB3", 
                100: "#FFE6CC"
            }
            outline_colors = {
                20: "#FF0000", 
                40: "#0000FF", 
                60: "#00FF00", 
                80: "#FFAA00", 
                100: "#FF00FF"
            }
            
            fill_color = colors.get(int(r), "#CCCCCC")
            outline_color = outline_colors.get(int(r), "#666666")
            
            draw.ellipse(bbox, fill=fill_color, outline=outline_color, width=2)
        else:
            # Unfilled version - only outline
            draw.ellipse(bbox, fill=None, outline="black", width=1)
    
    # Save image
    fill_type = "filled" if filled else "unfilled"
    filename = os.path.join(output_dir, f"{view_name}_{fill_type}.png")
    img.save(filename)
    print(f"Saved face view: {filename}")
    
    return filename

def create_2d_pixel_analysis(projections, view_name, output_dir, box_size=2000):
    """Create 2D pixel-level analysis and visualization with optimized method."""
    # Create a grid to track coverage
    grid = np.zeros((box_size, box_size), dtype=int)
    
    # Mark pixels covered by spheres using bounding boxes
    for proj_x, proj_y, r in projections:
        # Calculate bounding box
        x_min = max(0, int(np.floor(proj_x - r)))
        x_max = min(box_size, int(np.ceil(proj_x + r))) + 1
        y_min = max(0, int(np.floor(proj_y - r)))
        y_max = min(box_size, int(np.ceil(proj_y + r))) + 1
        
        # Iterate only over the bounding box
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                # Pixel center coordinates
                pixel_x = i + 0.5
                pixel_y = j + 0.5
                
                # Distance from pixel center to circle center
                distance = np.sqrt((pixel_x - proj_x)**2 + (pixel_y - proj_y)**2)
                
                if distance <= r:
                    grid[i, j] = 1
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap='RdYlBu_r', origin='lower', extent=[0, box_size, 0, box_size])
    plt.colorbar(label='Coverage (0=Empty, 1=Covered)')
    plt.title(f'2D Pixel Coverage Analysis - {view_name.replace("_", " ").title()}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # Add grid lines
    plt.grid(True, alpha=0.3)
    
    # Save the analysis
    analysis_filename = os.path.join(output_dir, f"2d_analysis_{view_name}.png")
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    covered_pixels = np.sum(grid)
    total_pixels = box_size * box_size
    remaining_pixels = total_pixels - covered_pixels
    coverage_percentage = (covered_pixels / total_pixels) * 100
    
    return covered_pixels, remaining_pixels, coverage_percentage, analysis_filename

def analyze_all_faces(spheres, output_dir, box_size=2000):
    """Analyze all 6 faces of the cuboid with descriptive view names."""
    face_map = {
        "+Z": "top_view",
        "-Z": "bottom_view",
        "+Y": "front_view",
        "-Y": "back_view",
        "+X": "right_view",
        "-X": "left_view"
    }
    
    face_data = []
    
    print("\nAnalyzing face views...")
    
    for face, view_name in face_map.items():
        print(f"Processing face: {view_name}")
        
        # Project spheres onto this face
        projections = project_spheres_to_face(spheres, face)
        
        # Create filled and unfilled views
        filled_img = render_face_view(projections, view_name, output_dir, filled=True, box_size=box_size)
        unfilled_img = render_face_view(projections, view_name, output_dir, filled=False, box_size=box_size)
        
        # Create 2D pixel analysis
        covered_px, remaining_px, coverage_percent, analysis_file = create_2d_pixel_analysis(
            projections, view_name, output_dir, box_size=box_size
        )
        
        # Collect sphere data for this face
        sphere_data = []
        for i, sphere in enumerate(spheres):
            x, y, z, r, diameter, volume, surface_area = sphere
            sphere_data.append({
                "Sphere_ID": i+1,
                "x": x,
                "y": y,
                "z": z,
                "radius": r,
                "diameter": diameter,
                "volume": volume,
                "surface_area": surface_area
            })
        
        face_data.append({
            "Face": view_name,
            "Visible_Spheres": len(projections),
            "Covered_Pixels": covered_px,
            "Remaining_Pixels": remaining_px,
            "Coverage_Percentage": coverage_percent,
            "Filled_Image": os.path.basename(filled_img),
            "Unfilled_Image": os.path.basename(unfilled_img),
            "Analysis_Image": os.path.basename(analysis_file),
            "Spheres": sphere_data
        })
        
        print(f"  - Visible spheres: {len(projections)}")
        print(f"  - Covered pixels: {covered_px}/{box_size*box_size} ({coverage_percent:.1f}%)")
    
    return pd.DataFrame(face_data)

def export_to_excel(spheres, face_analysis, output_dir, box_size):
    """Export all data to Excel with multiple sheets."""
    # Create main sphere dataframe
    sphere_df = pd.DataFrame(
        spheres, 
        columns=["x", "y", "z", "radius", "diameter", "volume", "surface_area"]
    )
    
    # Add sphere IDs
    sphere_df.insert(0, "Sphere_ID", range(1, len(sphere_df)+1))
    
    # Calculate summary statistics
    total_sphere_volume = sphere_df["volume"].sum()
    cuboid_volume = box_size**3
    remaining_volume = cuboid_volume - total_sphere_volume
    volume_utilization = (total_sphere_volume / cuboid_volume) * 100
    total_surface_area = sphere_df["surface_area"].sum()
    
    # Create summary dataframe
    summary_data = {
        "Metric": [
            "Total Spheres",
            "Total Sphere Volume",
            "Cuboid Volume", 
            "Remaining Volume",
            "Volume Utilization (%)",
            "Total Sphere Surface Area"
        ],
        "Value": [
            len(spheres),
            total_sphere_volume,
            cuboid_volume,
            remaining_volume,
            volume_utilization,
            total_surface_area
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Prepare face analysis data
    face_df = face_analysis[["Face", "Visible_Spheres", "Covered_Pixels", 
                             "Remaining_Pixels", "Coverage_Percentage",
                             "Filled_Image", "Unfilled_Image", "Analysis_Image"]].copy()
    
    # Write to Excel
    excel_path = os.path.join(output_dir, "spheres_analysis.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Main sheets
        sphere_df.to_excel(writer, sheet_name='Sphere_Data', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        face_df.to_excel(writer, sheet_name='Face_Analysis', index=False)
        
        # Add individual face sphere data
        for _, row in face_analysis.iterrows():
            face_name = row['Face']
            sphere_data = pd.DataFrame(row['Spheres'])
            sphere_data.to_excel(
                writer, 
                sheet_name=f"{face_name}_Spheres", 
                index=False
            )
    
    print(f"Excel file saved as: {excel_path}")
    return excel_path

def main():
    """Main function to run the complete analysis."""
    
    print("=== 3D Sphere Analysis Tool ===")
    print("Cuboid Size: 2000x2000x2000")
    
    # Create output directory
    output_dir = create_output_folder()
    print(f"Output directory: {output_dir}")
    
    print("Generating 100 random spheres in cuboid...")
    
    # Generate spheres with updated parameters
    box_size = 2000
    spheres = generate_spheres(
        n=100, 
        box_size=box_size, 
        radii=[20, 40, 60, 80, 100],  # Updated radii
        seed=42  # Fixed seed for reproducibility
    )
    print(f"Successfully generated {len(spheres)} spheres")
    
    # Create 3D visualization
    print("\nCreating 3D visualization...")
    create_3d_visualization(spheres, output_dir, box_size=box_size)
    
    # Analyze all faces
    print("\nAnalyzing all 6 faces...")
    face_analysis = analyze_all_faces(spheres, output_dir, box_size=box_size)
    
    # Export to Excel
    print("\nExporting data to Excel...")
    excel_path = export_to_excel(spheres, face_analysis, output_dir, box_size)
    
    # Print summary
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Total spheres generated: {len(spheres)}")
    print(f"Total volume occupied: {spheres[:,5].sum():.2f}")
    print(f"Total surface area: {spheres[:,6].sum():.2f}")
    cuboid_volume = box_size**3
    print(f"Volume utilization: {(spheres[:,5].sum() / cuboid_volume) * 100:.2f}%")
    print(f"All files saved in: {output_dir}/")
    
    print("\nGenerated files:")
    print("- 3d_visualization.png (3D view of all spheres)")
    print("- spheres_analysis.xlsx (Complete data analysis)")
    
    print("\nFace views generated (for each face):")
    print("  - [face]_filled.png (Filled circle view)")
    print("  - [face]_unfilled.png (Outline-only view)")
    print("  - 2d_analysis_[face].png (Pixel coverage analysis)")
    
    print("\nFace Coverage Summary:")
    for _, row in face_analysis.iterrows():
        print(f"  {row['Face']}: {row['Covered_Pixels']}/{box_size*box_size} pixels ({row['Coverage_Percentage']:.1f}%)")

if __name__ == "__main__":
    main()