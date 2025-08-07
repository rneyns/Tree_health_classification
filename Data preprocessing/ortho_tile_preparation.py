"""
This script merges RGB and NIR orthophoto tiles into 4-band RGBNIR GeoTIFFs.

Each tile is stored in its own subfolder under two root directories:
- VIS (RGB) imagery: contains 3-band images with Red, Green, Blue bands.
- NIR (CIR) imagery: contains 3-band images with NIR, Green, Blue bands.
  (Only the first band, NIR, is used from these images.)

The script:
1. Iterates through tile folders in the VIS directory.
2. Matches each tile with its corresponding NIR folder (by tile ID).
3. Reads the R, G, B bands from the VIS image and the NIR band from the NIR image.
4. Stacks them in the order [Red, Green, Blue, NIR].
5. Saves the merged image as a 4-band GeoTIFF in a specified output directory.

Assumptions:
- File naming follows a consistent pattern including "RGB" in VIS and "CIR" in NIR filenames.
- All input images are co-registered and have matching dimensions and georeferencing.

Adjust the VIS_ROOT, NIR_ROOT, and OUTPUT_FOLDER paths to your local setup before running.
"""

import os
import rasterio
import numpy as np
import shutil

# Paths
VIS_ROOT = '/Users/robbe_neyns/Documents/research/UHI tree health/ortho_tiles/unzipped'
NIR_ROOT = '/Users/robbe_neyns/Documents/research/UHI tree health/ortho_tiles_NIR/unzipped'
OUTPUT_FOLDER = '/Users/robbe_neyns/Documents/research/UHI tree health/merged_tiles'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# List all tile folders from VIS
tile_folders = [f for f in os.listdir(VIS_ROOT) if f.startswith("tile_")]
print('hallo')
for tile in tile_folders:
    vis_tile_path = os.path.join(VIS_ROOT, tile)
    nir_tile_path = os.path.join(NIR_ROOT, tile)

    # Find RGB file
    try:
        vis_files = [f for f in os.listdir(vis_tile_path) if f.endswith('.tif') and 'RGB' in f]
        nir_files = [f for f in os.listdir(nir_tile_path) if f.endswith('.tif') and 'CIR' in f]

        vis_file = os.path.join(vis_tile_path, vis_files[0])
        nir_file = os.path.join(nir_tile_path, nir_files[0])

        with rasterio.open(vis_file) as vis_src, rasterio.open(nir_file) as nir_src:
            # Read VIS: R=1, G=2, B=3
            red = vis_src.read(1)
            green = vis_src.read(2)
            blue = vis_src.read(3)

            # Read NIR: NIR=1 (band 1), G=2, B=3 but we only need band 1
            nir = nir_src.read(1)

            # Stack: [R, G, B, NIR]
            merged = np.stack([red, green, blue, nir])

            profile = vis_src.profile
            profile.update(count=4)

            output_path = os.path.join(OUTPUT_FOLDER, f"{tile}_RGBNIR.tif")
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(merged)
                dst.colorinterp = (
                    rasterio.enums.ColorInterp.red,
                    rasterio.enums.ColorInterp.green,
                    rasterio.enums.ColorInterp.blue,
                    rasterio.enums.ColorInterp.gray  # or undefined for NIR
                )

            print(f"✅ Merged {tile} -> {output_path}")

            # Delete original folders after success
            shutil.rmtree(vis_tile_path)
            shutil.rmtree(nir_tile_path)
            print(f"🗑️ Deleted original folders: {vis_tile_path}, {nir_tile_path}")


    except Exception as e:
        print(f"Skipping {tile}: Missing RGB or NIR file")
        print(f"❌ Failed to merge {tile}: {e}")