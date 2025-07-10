import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import os

class SphereProjector:
    def __init__(self, cuboid_size=100):
        self.cuboid_size = cuboid_size
        self.spheres = []
        
    def generate_spheres(self, num_spheres=100, min_radius=1, max_radius=5):
        """Generate spheres with random positions and radii inside the cuboid"""
        self.spheres = []
        
        for i in range(num_spheres):
            # Generate random radius
            radius = random.uniform(min_radius, max_radius)
            
            # Generate random position ensuring sphere is fully inside cuboid
            # Position range: [radius, cuboid_size - radius] for each axis
            x = random.uniform(radius, self.cuboid_size - radius)
            y = random.uniform(radius, self.cuboid_size - radius)
            z = random.uniform(radius, self.cuboid_size - radius)
            
            # Calculate volume
            volume = (4/3) * np.pi * radius**3
            
            sphere_data = {
                'id': i + 1,
                'x': x,
                'y': y,
                'z': z,
                'radius': radius,
                'volume': volume
            }
            
            self.spheres.append(sphere_data)
    
    def project_to_face(self, face):
        """Project spheres onto a specific face and return visible circles"""
        projections = []
        
        for sphere in self.spheres:
            x, y, z = sphere['x'], sphere['y'], sphere['z']
            r = sphere['radius']
            
            if face == 'front':  # YZ plane at x=0 (looking along +X axis)
                proj_x, proj_y = y, z
            elif face == 'back':   # YZ plane at x=100 (looking along -X axis)
                proj_x, proj_y = y, z
            elif face == 'left':   # XZ plane at y=0 (looking along +Y axis)
                proj_x, proj_y = x, z
            elif face == 'right':  # XZ plane at y=100 (looking along -Y axis)
                proj_x, proj_y = x, z
            elif face == 'bottom': # XY plane at z=0 (looking along +Z axis)
                proj_x, proj_y = x, y
            elif face == 'top':    # XY plane at z=100 (looking along -Z axis)
                proj_x, proj_y = x, y
            
            projections.append({
                'id': sphere['id'],
                'x': proj_x,
                'y': proj_y,
                'radius': r,
                'original_sphere': sphere
            })
        
        return projections
    
    def create_face_visualization(self, face, projections, save_path):
        """Create visualization for a specific face"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Set face-specific labels and title
        face_labels = {
            'front': ('Y', 'Z', 'Front View (YZ plane, looking along +X)'),
            'back': ('Y', 'Z', 'Back View (YZ plane, looking along -X)'),
            'left': ('X', 'Z', 'Left View (XZ plane, looking along +Y)'),
            'right': ('X', 'Z', 'Right View (XZ plane, looking along -Y)'),
            'bottom': ('X', 'Y', 'Bottom View (XY plane, looking along +Z)'),
            'top': ('X', 'Y', 'Top View (XY plane, looking along -Z)')
        }
        
        xlabel, ylabel, title = face_labels[face]
        
        # Draw spheres as circles
        for proj in projections:
            circle = Circle((proj['x'], proj['y']), proj['radius'], 
                          fill=False, edgecolor='blue', alpha=0.6, linewidth=0.8)
            ax.add_patch(circle)
        
        # Set up the plot
        ax.set_xlim(0, self.cuboid_size)
        ax.set_ylim(0, self.cuboid_size)
        ax.set_xlabel(f'{xlabel} axis')
        ax.set_ylabel(f'{ylabel} axis')
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add cuboid boundary
        rect = plt.Rectangle((0, 0), self.cuboid_size, self.cuboid_size, 
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_3d_visualization(self):
        """Create 3D visualization of all spheres in the cuboid"""
        fig = plt.figure(figsize=(15, 12))
        
        # Create 4 subplots for different 3D views
        views = [
            (1, 2, 1, (30, 45)),   # View 1: elevation=30, azimuth=45
            (1, 2, 2, (60, 135)),  # View 2: elevation=60, azimuth=135
        ]
        
        for i, (rows, cols, pos, (elev, azim)) in enumerate(views):
            ax = fig.add_subplot(rows, cols, pos, projection='3d')
            
            # Plot spheres as hollow wireframe circles
            for sphere in self.spheres:
                x, y, z = sphere['x'], sphere['y'], sphere['z']
                r = sphere['radius']
                
                # Create hollow sphere wireframe
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                x_sphere = r * np.outer(np.cos(u), np.sin(v)) + x
                y_sphere = r * np.outer(np.sin(u), np.sin(v)) + y
                z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
                
                # Plot hollow wireframe sphere
                ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                                color=plt.cm.viridis(r/5.0), alpha=0.6, linewidth=0.8)
            
            # Draw cuboid wireframe
            # Define cuboid vertices
            vertices = np.array([
                [0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],  # bottom face
                [0, 0, 100], [100, 0, 100], [100, 100, 100], [0, 100, 100]  # top face
            ])
            
            # Define edges of the cuboid
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
                [4, 5], [5, 6], [6, 7], [7, 4],  # top face
                [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
            ]
            
            # Draw edges
            for edge in edges:
                points = vertices[edge]
                ax.plot3D(*points.T, 'r-', linewidth=2, alpha=0.8)
            
            # Set labels and title
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            ax.set_title(f'3D View {i+1} (elev={elev}°, azim={azim}°)')
            
            # Set equal aspect ratio and limits
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 100)
            
            # Set view angle
            ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        save_path = 'sphere_projections/3d_visualization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved 3D visualization to {save_path}")
    
    def create_detailed_3d_views(self):
        """Create additional detailed 3D views"""
        # Create a comprehensive 3D view with 6 different angles
        fig = plt.figure(figsize=(20, 12))
        
        views = [
            (2, 3, 1, (0, 0), "Front"),      # Front view
            (2, 3, 2, (0, 90), "Side"),     # Side view
            (2, 3, 3, (90, 0), "Top"),      # Top view
            (2, 3, 4, (30, 45), "Isometric 1"),   # Isometric view 1
            (2, 3, 5, (60, 135), "Isometric 2"),  # Isometric view 2
            (2, 3, 6, (45, 225), "Isometric 3"),  # Isometric view 3
        ]
        
        for i, (rows, cols, pos, (elev, azim), title) in enumerate(views):
            ax = fig.add_subplot(rows, cols, pos, projection='3d')
            
            # Plot spheres as hollow circles (wireframe or scatter)
            x_coords = [sphere['x'] for sphere in self.spheres]
            y_coords = [sphere['y'] for sphere in self.spheres]
            z_coords = [sphere['z'] for sphere in self.spheres]
            radii = [sphere['radius'] for sphere in self.spheres]
            
            # Create hollow circles using scatter plot with edge only
            scatter = ax.scatter(x_coords, y_coords, z_coords, 
                               s=[r*30 for r in radii],  # Size proportional to radius
                               c='none',  # No fill color (hollow)
                               edgecolors=plt.cm.viridis(np.array(radii)/5.0),  # Edge color based on radius
                               linewidth=1.5,
                               alpha=0.8)
            
            # Draw cuboid wireframe
            vertices = np.array([
                [0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],
                [0, 0, 100], [100, 0, 100], [100, 100, 100], [0, 100, 100]
            ])
            
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            for edge in edges:
                points = vertices[edge]
                ax.plot3D(*points.T, 'r-', linewidth=1.5, alpha=0.6)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'{title}\n(elev={elev}°, azim={azim}°)')
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 100)
            
            ax.view_init(elev=elev, azim=azim)
        
        # Add colorbar manually since scatter plot is hollow
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=1, vmax=5))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20, label='Radius')
        
        plt.tight_layout()
        save_path = 'sphere_projections/3d_detailed_views.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved detailed 3D views to {save_path}")
    
    def create_interactive_3d_view(self):
        """Create a single high-quality 3D view for display"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each sphere as hollow wireframe
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.spheres)))
        
        for i, sphere in enumerate(self.spheres):
            x, y, z = sphere['x'], sphere['y'], sphere['z']
            r = sphere['radius']
            
            # Create hollow sphere wireframe
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 12)
            x_sphere = r * np.outer(np.cos(u), np.sin(v)) + x
            y_sphere = r * np.outer(np.sin(u), np.sin(v)) + y
            z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + z
            
            # Plot hollow wireframe sphere
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, 
                            color=colors[i], alpha=0.7, linewidth=0.8)
        
        # Draw cuboid
        vertices = np.array([
            [0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],
            [0, 0, 100], [100, 0, 100], [100, 100, 100], [0, 100, 100]
        ])
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, 'r-', linewidth=2)
        
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D Sphere Distribution in 100×100×100 Cuboid\n(100 spheres with radii 1-5)')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_zlim(0, 100)
        
        # Set nice viewing angle
        ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        save_path = 'sphere_projections/3d_main_view.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved main 3D view to {save_path}")
    
    def create_all_visualizations(self):
        """Create visualizations for all 6 faces and 3D views"""
        faces = ['front', 'back', 'left', 'right', 'bottom', 'top']
        
        # Create output directory if it doesn't exist
        if not os.path.exists('sphere_projections'):
            os.makedirs('sphere_projections')
        
        # Create 2D projections for each face
        for face in faces:
            projections = self.project_to_face(face)
            save_path = f'sphere_projections/{face}_view.png'
            self.create_face_visualization(face, projections, save_path)
            print(f"Saved {face} view to {save_path}")
        
        # Create 3D visualizations
        print("Creating 3D visualizations...")
        self.create_interactive_3d_view()
        self.create_3d_visualization()
        self.create_detailed_3d_views()
    
    def save_to_excel(self, filename='sphere_data.xlsx'):
        """Save sphere data to Excel file"""
        # Create DataFrame
        df = pd.DataFrame(self.spheres)
        
        # Add summary statistics
        total_volume = df['volume'].sum()
        avg_radius = df['radius'].mean()
        
        # Create workbook and worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Sphere Data"
        
        # Add sphere data
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)
        
        # Add summary section
        ws.append([])  # Empty row
        ws.append(['Summary Statistics'])
        ws.append(['Total Spheres:', len(self.spheres)])
        ws.append(['Total Volume:', total_volume])
        ws.append(['Average Radius:', avg_radius])
        ws.append(['Cuboid Size:', f'{self.cuboid_size}x{self.cuboid_size}x{self.cuboid_size}'])
        
        # Add projection data for each face
        faces = ['front', 'back', 'left', 'right', 'bottom', 'top']
        
        for face in faces:
            projections = self.project_to_face(face)
            
            # Create new worksheet for each face
            ws_face = wb.create_sheet(title=f"{face.capitalize()} Projection")
            
            # Add headers
            headers = ['Sphere ID', 'Projected X', 'Projected Y', 'Radius', 'Original X', 'Original Y', 'Original Z']
            ws_face.append(headers)
            
            # Add projection data
            for proj in projections:
                row = [
                    proj['id'],
                    proj['x'],
                    proj['y'],
                    proj['radius'],
                    proj['original_sphere']['x'],
                    proj['original_sphere']['y'],
                    proj['original_sphere']['z']
                ]
                ws_face.append(row)
        
        # Save workbook
        wb.save(filename)
        print(f"Saved sphere data to {filename}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Generating 100 spheres with random radii (1-5) inside 100x100x100 cuboid...")
        self.generate_spheres()
        
        print(f"Creating visualizations for all 6 faces and 3D views...")
        self.create_all_visualizations()
        
        print("Saving data to Excel file...")
        self.save_to_excel()
        
        # Print summary
        total_volume = sum(sphere['volume'] for sphere in self.spheres)
        avg_radius = sum(sphere['radius'] for sphere in self.spheres) / len(self.spheres)
        
        print(f"\nAnalysis Complete!")
        print(f"Total spheres generated: {len(self.spheres)}")
        print(f"Total volume: {total_volume:.2f} cubic units")
        print(f"Average radius: {avg_radius:.2f} units")
        print(f"All visualizations saved to 'sphere_projections/' directory")
        print(f"- 6 face projection views (2D)")
        print(f"- 3 different 3D visualization files")
        print(f"All data saved to 'sphere_data.xlsx'")

# Run the analysis
if __name__ == "__main__":
    projector = SphereProjector(cuboid_size=100)
    projector.run_complete_analysis()