"""
Script: PlanetScope Tree Reflectance Sampler
Author: Robbe Neyns (with help from genAI)
Date: [Today]
Description:
    This script samples reflectance values from PlanetScope images for a set of tree points.
    - One CSV is generated per band (band_1.csv, band_2.csv, etc.).
    - Rows represent individual trees (tree_id + optional species_code).
    - Columns represent image acquisition dates, labeled by day-of-year (DOY) only, sorted chronologically.
    - Supports automatic conversion of species names to numeric codes based on a user-defined mapping.

Requirements:
    - rasterio
    - geopandas
    - pandas
"""

import os
import rasterio
import geopandas as gpd
import pandas as pd
from datetime import datetime

# --- USER INPUTS ---
image_folder = '/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/PlanetScope preprocessing/Old PlanetScope data'
tree_points_file = '/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/Tree mapping/Tree locations/Brussels Environment Layers/mobiliteit_shape_manual_adjustment_project/mobiliteit_shape_manual_adjustment_X_Y.shp'  # or .geojson
output_folder = '/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/Tree mapping'

# Species string-to-code mapping
species_map = {
    "Platanus x acerifolia": 1,
    "Tilia x euchlora": 2,
    "Aesculus hippocastanum": 3,
    "Acer pseudoplatanus": 4,
    "Acer platanoides": 5,
}

# --- LOAD TREE POINTS ---
trees = gpd.read_file(tree_points_file)

# Add species code column if species name exists
if "essence" in trees.columns:
    def species_to_code(name):
        if pd.isna(name):
            return None
        for key, code in species_map.items():
            if key.lower() in name.lower():
                return code
        return None
    trees["species_code"] = trees["essence"].apply(species_to_code)
else:
    trees["species_code"] = 0

# Ensure tree ID exists
if "field_1" not in trees.columns:
    trees["field_1"] = range(len(trees))

# Prepare data structure for each band
band_data = {}  # {band_index: pd.DataFrame}

# --- PROCESS IMAGES ---
for filename in os.listdir(image_folder):
    if filename.endswith(".tif"):
        # Extract date from filename
        date_str = filename.split("_")[0]  # yyyymmdd
        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"Skipping file {filename}, cannot parse date.")
            continue
        doy = date.timetuple().tm_yday  # day of year

        image_path = os.path.join(image_folder, filename)
        with rasterio.open(image_path) as src:
            # Reproject tree points if needed
            if trees.crs != src.crs:
                trees_proj = trees.to_crs(src.crs)
            else:
                trees_proj = trees

            # Sample reflectance values
            for idx, row in trees_proj.iterrows():
                coords = [(row.geometry.x, row.geometry.y)]
                values = [val for val in src.sample(coords)][0]  # values per band
                for b, val in enumerate(values, start=1):
                    if b not in band_data:
                        # Initialize DataFrame
                        band_data[b] = pd.DataFrame({
                            "tree_id": trees["field_1"],
                            "species_code": trees["species_code"]
                        })
                    # Use DOY as column name
                    col_name = str(doy)
                    band_data[b].loc[idx, col_name] = val

# --- SORT COLUMNS BY DOY AND SAVE CSVs ---
for b, df in band_data.items():
    # Exclude 'tree_id' and 'species_code' from sorting
    fixed_cols = ["tree_id", "species_code"]
    date_cols = [c for c in df.columns if c not in fixed_cols]

    # Sort columns numerically by DOY
    date_cols_sorted = sorted(date_cols, key=lambda x: int(x))
    df = df[fixed_cols + date_cols_sorted]

    output_csv = os.path.join(output_folder, f"band_{b}.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
