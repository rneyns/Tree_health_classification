import os
import geopandas as gpd
import rasterio
import pickle
from rasterio.windows import Window
from rasterio.transform import from_origin


def build_tile_extent_library(tiles_folder, save_path=None):
    """
    Builds a library of extents for each ortho tile in the folder.
    The extents are stored as (min_lon, min_lat, max_lon, max_lat) for each tile.
    Optionally saves the library to a file.
    """
    tile_extents = {}

    for tile in os.listdir(tiles_folder):
        if tile.endswith('tif'):
            tile_path_rgb = os.path.join(tiles_folder, tile)

            if os.path.exists(tile_path_rgb):
                with rasterio.open(tile_path_rgb) as src:
                    # Get the bounds of the tile
                    min_lon, min_lat, max_lon, max_lat = src.bounds
                    print(min_lon, min_lat, max_lon, max_lat)
                    tile_extents[tile] = (min_lon, min_lat, max_lon, max_lat)

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(tile_extents, f)

    return tile_extents


def load_tile_extent_library(save_path):
    """
    Loads the tile extents library from a file.
    """
    with open(save_path, 'rb') as f:
        tile_extents = pickle.load(f)
    return tile_extents


def get_tile_for_coordinates(tile_extents, lon, lat):
    """
    Given the coordinates (lon, lat), determine which tile contains the point
    based on the extents of the tiles.
    """
    for tile_id, (min_lon, min_lat, max_lon, max_lat) in tile_extents.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return tile_id
    return None


def cut_patch_from_tile(tile_path, central_lon, central_lat, patch_size_m=6):
    """
    Cut out a patch of size patch_size_m x patch_size_m meters around the central location
    from the ortho tile image.
    """
    with rasterio.open(tile_path) as src:
        # Get metadata and transform of the image
        transform = src.transform

        # Calculate the pixel coordinates for the center location
        x_pixel, y_pixel = ~transform * (central_lon, central_lat)

        # Calculate the number of pixels corresponding to the patch size in meters
        meters_per_pixel_x = abs(transform[0])  # Pixel size in x-direction (in meters)
        meters_per_pixel_y = abs(transform[4])  # Pixel size in y-direction (in meters)

        patch_size_pixels_x = int(patch_size_m / meters_per_pixel_x)
        patch_size_pixels_y = int(patch_size_m / meters_per_pixel_y)

        # Define the window for cropping (centered around the central location)
        window = Window(
            int(x_pixel - patch_size_pixels_x // 2),
            int(y_pixel - patch_size_pixels_y // 2),
            patch_size_pixels_x,
            patch_size_pixels_y
        )

        # Read the data within the window
        patch = src.read(window=window)

        # Update metadata for the new cropped file
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'count': patch.shape[0],  # number of bands
            'height': patch.shape[1],
            'width': patch.shape[2],
            'transform': src.window_transform(window)
        })

    return patch, meta


def process_coordinates(shapefile, tiles_folder, output_folder, tile_extents):
    """
    Processes the shapefile with coordinates and extracts the 6x6 meter patch
    from the corresponding ortho tiles.
    """
    # Read the shapefile using geopandas
    gdf = gpd.read_file(shapefile)

    # Loop through each row in the shapefile
    for _, row in gdf.iterrows():
        central_lon = row['x']  # Assumes column name for longitude is 'longitude'
        central_lat = row['y']  # Assumes column name for latitude is 'latitude'
        #id = row['field_1']
        id = row['crown_id']
        tile_id = get_tile_for_coordinates(tile_extents, central_lon, central_lat)

        if tile_id:
            # Locate the correct tile
            tile_path_rgb = os.path.join(tiles_folder, tile_id)

            # Cut out the patch
            patch, meta = cut_patch_from_tile(tile_path_rgb, central_lon, central_lat)

            # Save the patch to the output folder with the tile ID as filename
            output_path = os.path.join(output_folder, f'{id}.tif')
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(patch)


if __name__ == "__main__":
    shapefile = '/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/Tree mapping/Tree locations/Brussels Environment Layers/mobiliteit_shape_manual_adjustment_project/mobiliteit_shape_manual_adjustment_X_Y.shp'  # Path to the shapefile with coordinates
    shapefile = '/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/Tree mapping/Tree locations/flai layers/tree_centroids_x_y_lambert.shp'  # Path to the shapefile with coordinates
    tiles_folder = "/Users/robbe_neyns/Documents/research/UHI tree health/merged_tiles"  # Path to the folder containing the ortho tiles
    output_folder = "/Users/robbe_neyns/Documents/research/UHI tree health/output_patches_apply"  # Path where the patches will be saved
    extent_library_path = "tile_extents.pkl"  # Path to save/load the tile extents library

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if the tile extents library exists
    if os.path.exists(extent_library_path):
        tile_extents = load_tile_extent_library(extent_library_path)
    else:
        # Build the library and save it if it doesn't exist
        tile_extents = build_tile_extent_library(tiles_folder, extent_library_path)

    # Process the coordinates from the shapefile
    process_coordinates(shapefile, tiles_folder, output_folder, tile_extents)
