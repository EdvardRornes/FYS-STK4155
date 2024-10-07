import numpy as np
import matplotlib.pyplot as plt
from utils import *


if __name__ == "__main__":
    # latex_fonts()
    contour_plot = False; surface_3D_plot = True

    surface_3D_skip_every = 50
    # Filenames
    files = {
        'Grand Canyon': 'Data/GrandCanyon.tif',
        'Mount Everest': 'Data/Everest.tif'
    }

    # Color mapping for each terrain
    color_map = {
        'Grand Canyon': 'Oranges_r',
        'Mount Everest': 'Blues_r'
    }

    terrain_data = read_terrain(files)

    if contour_plot:
        # Create a figure with subplots for terrain and contour plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns
        for i, name in enumerate(terrain_data):
            plot_terrain(name, terrain_data[name]["x"], terrain_data[name]["y"], terrain_data[name]["z"], axes[i], color_map[name])

        # Adjust layout and show all plots at once
        plt.tight_layout()
        plt.savefig(f'Figures/Terrain/TerrainContourAndImage.pdf')
        plt.show()

    if surface_3D_plot:
        for i, name in enumerate(terrain_data):
            fig = plt.figure(figsize=(10, 10), constrained_layout=True)
            ax = fig.add_subplot(111, projection='3d')  # Use add_subplot instead of gca
            x, y, z = terrain_data[name]["x"], terrain_data[name]["y"], terrain_data[name]["z"]

            x, y= x[0:surface_3D_skip_every:], y[0:surface_3D_skip_every:]
            z = z[:surface_3D_skip_every:, :surface_3D_skip_every:]

            X, Y = np.meshgrid(x,y)
            surface_3D(X, Y, z, ax, fig)
            ax.set_title(name)
            plt.show()
