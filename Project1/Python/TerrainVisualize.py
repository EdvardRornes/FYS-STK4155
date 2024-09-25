import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from utils import *

latex_fonts()

# File paths
files = {
    'Grand Canyon': 'Data/GrandCanyon.tif',
    'Mount Everest': 'Data/Everest.tif'
}

# Load terrain data and conversion factors
terrain_data = {}
lat_conversion_factors = {}

for name, filepath in files.items():
    terrain_data[name] = imageio.imread(filepath)
    lat_conversion_factors[name] = get_latitude_and_conversion(filepath)[1]

# Get the dimensions of the terrain (assuming they are the same for all)
height, width = terrain_data['Grand Canyon'].shape

# Transformation parameters
pixel_size_x_degrees = 0.0002777777777777778  # degrees per pixel (longitude)
pixel_size_y_degrees = -0.0002777777777777778  # degrees per pixel (latitude)

# Convert pixel sizes to kilometers
pixel_sizes_km = {name: {
    'x': pixel_size_x_degrees * conv / 1000,
    'y': abs(pixel_size_y_degrees * conv / 1000)
} for name, conv in lat_conversion_factors.items()}

# Create coordinate arrays for each terrain file
coordinates_km = {name: {
    'x': np.arange(width) * pixel_sizes_km[name]['x'],
    'y': np.arange(height) * pixel_sizes_km[name]['y']
} for name in terrain_data}

# Function to create and store terrain and contour plots
def plot_terrain(name, terrain, coords, axes):
    # Convert terrain to a NumPy array explicitly to avoid deprecation warnings
    terrain_array = np.array(terrain, dtype=np.float64) / 1000
    
    # Define color mapping for each terrain
    color_map = {
        'Grand Canyon': 'Oranges_r',
        'Mount Everest': 'Blues_r'
    }
    
    # Plot terrain image
    ax1 = axes[0]  # Access the first subplot for terrain
    im = ax1.imshow(terrain_array, cmap=color_map[name], origin='lower',
                    extent=(0, coords['x'][-1], 0, coords['y'][-1]))
    ax1.set_title(f'Elevation of {name}')
    ax1.set_xlabel('km')
    ax1.set_ylabel('km')
    
    # Adjust color bar size
    cbar1 = plt.colorbar(im, ax=ax1, label=r'Altitude (km)', shrink=0.7)  # Adjust the height of the color bar
    cbar1.ax.tick_params(labelsize=15)  # Keep the tick label size as it is
    
    # Set x-axis ticks for terrain image
    ax1.set_xticks(np.arange(0, coords['x'][-1] + 1, 20))  # Set x ticks at intervals of 20

    # Plot contour
    ax2 = axes[1]  # Access the second subplot for contour
    contour = ax2.contourf(coords['x'], coords['y'], terrain_array, cmap=color_map[name], origin='lower')
    ax2.set_title(f'Contour Plot of {name} Elevation')
    ax2.set_xlabel('km')
    ax2.set_ylabel('km')
    
    # Adjust color bar size for contour plot
    cbar2 = plt.colorbar(contour, ax=ax2, label=r'Altitude (km)', shrink=0.7)  # Adjust the height of the color bar
    cbar2.ax.tick_params(labelsize=15)  # Keep the tick label size as it is

    # Set x-axis ticks for contour
    ax2.set_xticks(np.arange(0, coords['x'][-1] + 1, 20))  # Set x ticks at intervals of 20
    
    ax2.set_aspect('equal', adjustable='box')  # Maintain aspect ratio

# Create a figure with subplots for terrain and contour plots
fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns

# Generate plots for each terrain
for i, name in enumerate(terrain_data):
    plot_terrain(name, terrain_data[name], coordinates_km[name], axes[i])

# Adjust layout and show all plots at once
plt.tight_layout()
plt.savefig(f'Figures/TerrainContourAndImage.pdf')
plt.show()
