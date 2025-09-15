"""
Script: PlanetScope Tree Reflectance Sampler (Polygon Means)
Author: Robbe Neyns (with help from genAI)
Date: 2025-10-01
Description:
    This script samples average (mean) reflectance per polygon from PlanetScope images.
    - One CSV is generated per band (band_1.csv, band_2.csv, etc.).
    - Rows represent individual polygons (polygon_id + optional species_code).
    - Columns represent image acquisition dates, labeled by day-of-year (DOY) only, sorted chronologically.
    - Supports automatic conversion of species names to numeric codes based on a user-defined mapping.

Notes:
    - This version uses zonal statistics via raster masking (rasterio.mask) and computes np.nanmean.
    - If a polygon has no valid pixels (all nodata), the cell will be left NaN.
    - Make sure your polygons reasonably overlap the rasters and CRSs match (the script reprojects polygons as needed).

Requirements:
    - rasterio
    - geopandas
    - pandas
    - numpy
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import mapping
from datetime import datetime

# --- USER INPUTS ---
image_folder = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope'
polygons_file = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/tree_crowns_with_species.shp"  # polygon layer
output_folder = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Planet data'

# Species string-to-code mapping (optional; applied if "essence" exists on polygons)
species_map = {
    "Platanus x acerifolia": 1,
    "Tilia x euchlora": 2,
    "Aesculus hippocastanum": 3,
    "Acer pseudoplatanus": 4,
    "Acer platanoides": 5,
}

# --- LOAD POLYGONS ---
polys = gpd.read_file(polygons_file)

# Add species code column if species name exists
if "essence" in polys.columns:
    def species_to_code(name):
        if pd.isna(name):
            return 0
        for key, code in species_map.items():
            if key.lower() in str(name).lower():
                return code
        return 0
    polys["species_code"] = polys["essence"].apply(species_to_code)
else:
    polys["species_code"] = 0

# Ensure polygon ID exists
# Will use "field_1" if present, else "crown_id" if present, else create sequential ID
if "field_1" in polys.columns:
    polys["polygon_id"] = polys["field_1"]
elif "crown_id" in polys.columns:
    polys["polygon_id"] = polys["crown_id"]
else:
    polys["polygon_id"] = range(len(polys))

# Prepare data structure for each band: {band_index: DataFrame}
# Initialize with ID + species_code so we don't recreate repeatedly
band_data = {}

# --- HELPER: ensure output folder exists ---
os.makedirs(output_folder, exist_ok=True)

# --- PROCESS IMAGES ---
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(".tif"):
        continue

    # Extract date from filename (expecting prefix yyyymmdd_*.tif)
    date_str = filename.split("_")[1]  # yyyymmdd
    try:
        date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"Skipping file {filename}, cannot parse date prefix as YYYYMMDD.")
        continue
    doy = date.timetuple().tm_yday  # day of year label
    col_name = str(doy)

    image_path = os.path.join(image_folder, filename)
    with rasterio.open(image_path) as src:
        # Reproject polygons if needed
        if polys.crs and src.crs and polys.crs != src.crs:
            polys_proj = polys.to_crs(src.crs)
        else:
            polys_proj = polys

        # Get nodata per band if available; fallback to None
        nodata = src.nodatavals  # tuple per band or (None,)
        count_bands = src.count

        # Initialize band DataFrames on first encounter with this raster (or first pass)
        for b in range(1, count_bands + 1):
            if b not in band_data:
                band_data[b] = pd.DataFrame({
                    "polygon_id": polys_proj["polygon_id"].values,
                    "species_code": polys_proj["species_code"].values
                })

        # For each polygon, mask raster and compute mean per band
        # (This loops polygons; for large datasets this can be slow. Consider rasterstats if performance is critical.)
        for idx, geom in enumerate(polys_proj.geometry):
            if geom is None or geom.is_empty:
                # leave NaN for this polygon/date
                continue

            try:
                # Crop & mask raster by polygon
                # returns data shape: (bands, rows, cols)
                data, _ = mask(src, [mapping(geom)], crop=True, nodata=nodata[0] if nodata else None)
            except ValueError:
                # Polygon outside raster extent; leave NaN
                continue

            # Compute mean per band ignoring nodata/NaNs
            # Convert nodata to NaN explicitly per band if needed
            for b in range(1, count_bands + 1):
                band_arr = data[b - 1].astype("float64")

                # Replace nodata with NaN if nodata is defined and present
                band_nodata = None
                if nodata and len(nodata) >= b:
                    band_nodata = nodata[b - 1]
                if band_nodata is not None:
                    band_arr[band_arr == band_nodata] = np.nan

                # If the masked area is entirely NaN or empty, result will be NaN
                mean_val = float(np.nanmean(band_arr)) if np.isfinite(band_arr).any() else np.nan

                # Ensure the DOY column exists, then set value for this polygon row
                df = band_data[b]
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df.at[idx, col_name] = mean_val

# --- SORT COLUMNS BY DOY AND SAVE CSVs ---
for b, df in band_data.items():
    # Exclude 'polygon_id' and 'species_code' from sorting
    fixed_cols = ["polygon_id", "species_code"]
    date_cols = [c for c in df.columns if c not in fixed_cols]
    # Keep only numeric DOY columns (defensive)
    date_cols_numeric = [c for c in date_cols if c.isdigit()]

    # Sort columns numerically by DOY
    date_cols_sorted = sorted(date_cols_numeric, key=lambda x: int(x))
    df_out = df[fixed_cols + date_cols_sorted]

    output_csv = os.path.join(output_folder, f"band_{b}.csv")
    df_out.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
