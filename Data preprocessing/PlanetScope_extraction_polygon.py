"""
Script: PlanetScope Tree Reflectance Sampler (Means by Polygon, Linked to Point ID)
Author: Robbe Neyns (with help from genAI)
Date: 2025-10-01
Description:
    This script computes average (mean) reflectance values from PlanetScope images
    over tree crown polygons, but outputs them keyed by the ID of a tree point
    that lies inside each polygon.

    - One CSV per band (band_1.csv, band_2.csv, ...).
    - Rows: unique tree points (point_id + species_code).
    - Columns: acquisition day-of-year (DOY), sorted.
    - If multiple points are inside a polygon, the one closest to the centroid is chosen.
    - If no point is inside, that polygon is dropped (not included in output).

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
image_folder   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope'
polygons_file  = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/tree_crowns_with_species.shp"
points_file    = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/Brussels Environment Layers/mobiliteit_shape_manual_adjustment_project/mobiliteit_shape_manual_adjustment_X_Y.shp'  # must exist, else script can't assign point_id
point_id_field = "field_1"  # column in points file with unique tree ID
output_folder  = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Planet data"
qa_points_out   = os.path.join(output_folder, "matched_points.gpkg")  # GeoPackage is robust


# Species string-to-code mapping (optional; applied if "essence" exists on polygons)
species_map = {
    "Platanus x acerifolia": 1,
    "Tilia x euchlora": 2,
    "Aesculus hippocastanum": 3,
    "Acer pseudoplatanus": 4,
    "Acer platanoides": 5,
}

# --- LOAD DATA ---
polys = gpd.read_file(polygons_file)
pts   = gpd.read_file(points_file)

# Optional polygon_id for QA only (NOT used in CSVs)
if "field_1" in polys.columns:
    polys["polygon_id"] = polys["field_1"]
elif "crown_id" in polys.columns:
    polys["polygon_id"] = polys["crown_id"]
else:
    polys["polygon_id"] = polys.index  # fallback for QA

# Species code on polygons (if 'essence' present)
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

# Reproject points to polygons CRS for spatial join + distance
if pts.crs and polys.crs and pts.crs != polys.crs:
    pts = pts.to_crs(polys.crs)

# Validate point_id field
if point_id_field not in pts.columns:
    raise ValueError(f"'{point_id_field}' not found in points. "
                     f"Available columns: {list(pts.columns)}")

# --- MATCH POINT -> POLYGON (point must be within) ---
# Compute polygon centroids as a GeoSeries (avoid the earlier TypeError)
centroids_gs = gpd.GeoSeries(polys.geometry.centroid, index=polys.index, crs=polys.crs)
centroids_gs.name = "centroid"

# Spatial join: keep only points that fall within polygons
joined = gpd.sjoin(
    pts[[point_id_field, "geometry"]],
    polys[["species_code", "polygon_id", "geometry"]],
    how="inner",
    predicate="within"
)
# joined has: [point_id_field, geometry(point), index_right (polygon index), species_code, polygon_id]

if joined.empty:
    raise RuntimeError("No points found inside any polygons!")

# Attach the polygon centroid to each joined row and compute distance
joined = joined.join(centroids_gs, on="index_right")
joined["_dist"] = joined.geometry.distance(joined["centroid"])

# Pick the closest point per polygon
sel = joined.sort_values("_dist").groupby("index_right", as_index=True).first()
# sel columns include: [point_id_field, geometry(point), species_code, polygon_id, centroid, _dist]

# Merge mapping back onto polygons: keep only polygons with a matched point
# IMPORTANT: only take the point id column to avoid 'species_code' overlap
polys_matched = polys.join(sel[[point_id_field]], how="inner")
polys_matched = polys_matched.rename(columns={point_id_field: "point_id"})

# --- BUILD QA POINT LAYER (unique matched points) ---
# 'sel' is indexed by the polygon index (index_right). Use that index directly.
qa_pts = sel[[point_id_field, "geometry"]].copy()
qa_pts = qa_pts.rename(columns={point_id_field: "point_id"})

# Pull species_code and polygon_id from the polygons using the same index
qa_pts["species_code"] = polys.loc[qa_pts.index, "species_code"].values
qa_pts["polygon_id"]  = polys.loc[qa_pts.index, "polygon_id"].values

# Drop duplicate point_ids if a single point got matched to multiple polygons
qa_pts = qa_pts[~qa_pts["point_id"].duplicated()].copy()

# Make it a GeoDataFrame and save
qa_pts_gdf = gpd.GeoDataFrame(qa_pts, geometry="geometry", crs=polys.crs)
os.makedirs(output_folder, exist_ok=True)
qa_pts_gdf.to_file(qa_points_out, layer="points_matched", driver="GPKG")

print(f"Saved QA points: {qa_points_out}")

# --- PREP CSV OUTPUT STRUCTURES ---
band_data = {}  # {band_index: DataFrame with rows aligned to polys_matched order}

# --- PROCESS IMAGES ---
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(".tif"):
        continue

    # Extract date (search for any 8-digit token)
    date_str = None
    for tok in filename.split("_"):
        if len(tok) == 8 and tok.isdigit():
            date_str = tok
            break
    if date_str is None:
        print(f"Skipping {filename}: no YYYYMMDD token found.")
        continue
    try:
        date = datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"Skipping {filename}: invalid date token {date_str}.")
        continue

    doy = date.timetuple().tm_yday
    col_name = str(doy)

    image_path = os.path.join(image_folder, filename)
    with rasterio.open(image_path) as src:
        # Reproject polygons if needed
        if polys_matched.crs and src.crs and polys_matched.crs != src.crs:
            polys_proj = polys_matched.to_crs(src.crs)
        else:
            polys_proj = polys_matched

        nodata_vals = src.nodatavals  # tuple per band
        n_bands = src.count

        # Initialize per-band DataFrames at first raster
        for b in range(1, n_bands + 1):
            if b not in band_data:
                band_data[b] = pd.DataFrame({
                    "point_id": polys_proj["point_id"].values,
                    "species_code": polys_proj["species_code"].values
                })

        # Loop polygons -> mask & mean per band
        for row_idx, geom in enumerate(polys_proj.geometry):
            if geom is None or geom.is_empty:
                continue
            try:
                data, _ = mask(src, [mapping(geom)], crop=True,
                               nodata=nodata_vals[0] if nodata_vals else None)
            except ValueError:
                # polygon outside raster
                continue

            for b in range(1, n_bands + 1):
                arr = data[b - 1].astype("float64")
                band_nodata = nodata_vals[b - 1] if (nodata_vals and len(nodata_vals) >= b) else None
                if band_nodata is not None:
                    arr[arr == band_nodata] = np.nan
                mean_val = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan

                df = band_data[b]
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df.at[row_idx, col_name] = mean_val

# --- SAVE CSVs (keyed by point_id + species_code) ---
for b, df in band_data.items():
    fixed = ["point_id", "species_code"]
    date_cols = [c for c in df.columns if c not in fixed and c.isdigit()]
    date_cols_sorted = sorted(date_cols, key=lambda x: int(x))
    out = df[fixed + date_cols_sorted]
    out_csv = os.path.join(output_folder, f"band_{b}.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")