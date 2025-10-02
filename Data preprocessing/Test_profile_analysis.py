#!/usr/bin/env python3
"""
Make 10 scatterplots (one per tree) of NDVI vs DOY using precomputed NDVI rasters.

- Tree layer: any point layer GeoPandas can read (SHP/GPKG/GeoJSON…)
- NDVI rasters: single-band GeoTIFFs (your previously created NDVI)
- Date: parsed from filename (YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD, or YYYYJJJ)

Outputs:
- PNGs in OUTPUT_PLOTS_DIR named: Tree_<idx>_NDVI_vs_DOY.png

Dependencies:
    pip install geopandas rasterio numpy matplotlib shapely
"""

import os
import re
import random
from glob import glob
from datetime import datetime

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.sample import sample_gen as rio_sample
import matplotlib.pyplot as plt

# ------------- EDIT THESE -------------
TREE_LAYER_PATH   = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/Brussels Environment Layers/mobiliteit_shape_manual_adjustment_project/mobiliteit_shape_manual_adjustment_X_Y.shp'  # your tree points (POINT geometries)
TREE_LAYER_NAME   = None                   # e.g., "trees" if GPKG has multiple layers; else None
NDVI_DIR          = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi' # folder with your NDVI GeoTIFFs
NDVI_GLOB         = "*_ndvi.tif"           # pattern for your NDVI files
OUTPUT_PLOTS_DIR  = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/plots'       # where to save PNGs

N_TREES           = 10                     # make 10 separate plots
RANDOM_SEED       = 42                     # reproducible selection
AUTO_SCALE_10000  = True                   # if values look like 0..10000, divide by 10000
# --------------------------------------


def parse_date_from_name(path):
    """Parse date from filename: YYYY-MM-DD / YYYY_MM_DD / YYYYMMDD / YYYYJJJ."""
    base = os.path.basename(path)

    m = re.search(r"(20\d{2})[-_](\d{2})[-_](\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).date()
        except ValueError: pass

    m = re.search(r"(20\d{2})(\d{2})(\d{2})", base)
    if m:
        y, mo, d = map(int, m.groups())
        try: return datetime(y, mo, d).date()
        except ValueError: pass

    m = re.search(r"(20\d{2})(\d{3})", base)  # YYYYJJJ (year+DOY)
    if m:
        y, jjj = int(m.group(1)), int(m.group(2))
        try: return datetime.strptime(f"{y}{jjj:03d}", "%Y%j").date()
        except ValueError: pass

    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).date()
    except Exception:
        return None


def to_doy(date_obj):
    return int(date_obj.strftime("%j")) if date_obj else None


def pick_points(gdf, n, seed=42):
    gdf = gdf[gdf.geometry.notnull() & gdf.geometry.geom_type.isin(["Point", "MultiPoint"])].copy()
    if gdf.empty:
        raise RuntimeError("No point geometries found in the tree layer.")
    if "MultiPoint" in gdf.geometry.geom_type.unique():
        gdf = gdf.explode(index_parts=False)
    if len(gdf) <= n:
        return gdf.reset_index(drop=True)
    random.seed(seed)
    return gdf.sample(n, random_state=seed).reset_index(drop=True)


def main():
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    # 1) Load trees and pick 10
    trees = gpd.read_file(TREE_LAYER_PATH, layer=TREE_LAYER_NAME) if TREE_LAYER_NAME else gpd.read_file(TREE_LAYER_PATH)
    trees_sel = pick_points(trees, N_TREES, RANDOM_SEED)

    # 2) Collect NDVI rasters and sort by acquisition date
    ndvi_files = sorted(glob(os.path.join(NDVI_DIR, NDVI_GLOB)))
    if not ndvi_files:
        raise RuntimeError(f"No NDVI rasters matched: {os.path.join(NDVI_DIR, NDVI_GLOB)}")

    dated = []
    for f in ndvi_files:
        d = parse_date_from_name(f)
        if d is not None:
            dated.append((f, d, to_doy(d)))
    if not dated:
        raise RuntimeError("Could not parse dates from NDVI filenames. Adjust parse_date_from_name().")

    dated.sort(key=lambda x: (x[1], x[0]))  # sort by date then name

    # 3) For each NDVI raster, sample NDVI at the 10 trees
    per_tree = {i: {"doy": [], "ndvi": []} for i in range(len(trees_sel))}

    for f, date_obj, doy in dated:
        if doy is None:
            continue

        with rasterio.open(f) as src:
            # Reproject points to the raster CRS if needed
            pts = trees_sel.to_crs(src.crs) if str(trees_sel.crs) != str(src.crs) else trees_sel

            # Build list of (x, y) coords
            coords = []
            for g in pts.geometry:
                if g.geom_type == "MultiPoint":
                    if len(g.geoms) == 0:
                        coords.append((np.nan, np.nan))
                    else:
                        p = g.geoms[0]
                        coords.append((p.x, p.y))
                else:
                    coords.append((g.x, g.y))

            # Sample single-band NDVI at coords
            # (rasterio returns an array per point; we take [0] since there is 1 band)
            samples = list(rio_sample(src, coords))
            vals = [float(s[0]) if (s is not None and len(s) > 0) else np.nan for s in samples]

            # Auto-scale if values look like 0..10000
            if AUTO_SCALE_10000:
                try:
                    mx = np.nanmax(vals)
                    if np.isfinite(mx) and mx > 2.0:
                        vals = [v / 10000.0 if np.isfinite(v) else v for v in vals]
                except Exception:
                    pass

            # Add to series
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    per_tree[i]["doy"].append(doy)
                    per_tree[i]["ndvi"].append(v)

    # 4) Make 10 separate scatterplots (one per tree)
    for i in range(min(N_TREES, len(trees_sel))):
        x = per_tree[i]["doy"]
        y = per_tree[i]["ndvi"]
        plt.figure(figsize=(7, 4.5))
        plt.scatter(x, y, s=20)
        plt.xlabel("Day of Year (DOY)")
        plt.ylabel("NDVI")
        plt.title(f"Tree {i+1}: NDVI vs DOY")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(OUTPUT_PLOTS_DIR, f"Tree_{i+1:02d}_NDVI_vs_DOY.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[Saved] {out_png}")

    print("Done.")

if __name__ == "__main__":
    main()
