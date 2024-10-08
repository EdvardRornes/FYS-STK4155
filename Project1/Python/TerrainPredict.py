import numpy as np
from utils import * 

latex_fonts()
if __name__ == "__main__":
    save = True 

    ################ Franke ################
    visualize_franke = True; regr_method_franke = "Ridge"
    additional_description_franke = "Unscaled"
    # additional_description_franke = "MINMAX"
    # additional_description_franke = "StandardScaling"
    p_franke = 6 # Poly degree to train method on

    ################ Terrain ################
    visualize_terrain = True; regr_method_terrain = "Ridge"
    additional_description_terrain = "Unscaled"
    # additional_description_terrain = "MINMAX"
    additional_description_terrain = "StandardScaling"

    p_terrain = 6 # Poly degree to train method on
    lmbdas = [1e-8]

    files = {
    'Grand Canyon': 'Data/GrandCanyon.tif',
    'Mount Everest': 'Data/Everest.tif'
    }

    ##### Franke #####
    if visualize_franke:
        N = 100; eps = 0.1
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        franke = Franke(N, eps)
        x, y = np.meshgrid(x, y)
        z = franke.franke(x, y) 
        z += eps * np.random.normal(0, 1, z.shape)

        TMP = PolynomialRegression(regr_method_franke, p_franke, [x.flatten(),y.flatten(),z.flatten()], start_training=False, scaling=additional_description_franke)
        TMP.surface3D_visualize(x, y, z, p_franke,1e-8, title=f"Franke predicted by Ridge")
        if save:
            plt.savefig(f"Figures/Terrain/{regr_method_franke}_franke_{additional_description_franke}_3D.pdf")

        plt.show()
        TMP.contour_visualize(x, y, z, p_franke,1e-8, title=f"Franke predicted by Ridge")
        if save:
            plt.savefig(f"Figures/Terrain/{regr_method_franke}_franke_{additional_description_franke}.pdf")
        plt.show()
    
    if visualize_terrain:
        for name, file in files.items():
            terrain = imageio.imread(file)
            # Use a subset of the terrain data for analysis
            N = 1000
            z = terrain[::int(terrain.shape[0]/N), ::int(terrain.shape[1]/N)]
            

            z_shape = np.shape(z)
            x = np.linspace(0, 1, z_shape[0])
            y = np.linspace(0, 1, z_shape[1])
            x, y = np.meshgrid(x, y)

            # z = terrain_subset.flatten()

            TMP = PolynomialRegression(regr_method_terrain, p_terrain, [x.flatten(),y.flatten(),z.flatten()], start_training=False, scaling=additional_description_terrain)

            #### Actual:
            TMP.surface3D_visualize(x, y, z, p_terrain, 1e-9, title=f"Actual terrain ({name})", give_me_data=True)
            if save:
                plt.savefig(f"Figures/Terrain/{regr_method_terrain}_{name}_{additional_description_terrain}_actual_3D.pdf")
            plt.show()

            TMP.contour_visualize(x, y, z, p_terrain, 1e-9, title=f"Actual terrain ({name})", give_me_data=True)
            if save:
                plt.savefig(f"Figures/Terrain/{regr_method_terrain}_{name}_{additional_description_terrain}_actual.pdf")
            plt.show()

            ### Predicted:
            TMP.surface3D_visualize(x, y, z, p_terrain, 1e-9, title=f"Predicted terrain using {regr_method_terrain} ({name})")
            if save:
                plt.savefig(f"Figures/Terrain/{regr_method_terrain}_{name}_{additional_description_terrain}_3D.pdf")
            plt.show()

            TMP.contour_visualize(x, y, z, p_terrain, 1e-9, title=f"Predicted terrain using {regr_method_terrain} ({name})")
            if save:
                plt.savefig(f"Figures/Terrain/{regr_method_terrain}_{name}_{additional_description_terrain}.pdf")
            plt.show()