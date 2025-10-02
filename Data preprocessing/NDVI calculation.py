#!/usr/bin/env python3
"""
Compute NDVI for all PlanetScope 8-band images in a folder, no argparse.

Assumptions (PlanetScope 8-band SR):
- 1-based bands: 1: Coastal, 2: Blue, 3: Green, 4: Yellow,
                 5: Red, 6: Red Edge, 7: NIR1, 8: NIR2
- NDVI = (NIR1 - Red) / (NIR1 + Red)
- Defaults: Red=5, NIR1=7
"""

import os
from glob import glob
import numpy as np
import rasterio

# --- EDIT THESE ---
INPUT_DIR = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Alex PlanetScope'
OUTPUT_DIR = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Data/PlanetScope/Planet_ndvi'
PATTERN = "*.tif"          # e.g., "*.tif" or "*_PSScene_8b.tif"
RED_BAND = 6               # 1-based
NIR_BAND = 8               # 1-based
# -------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_ndvi(in_path):
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_ndvi.tif")

    # If a previous broken file exists, remove it first to avoid metadata leftovers
    if os.path.exists(out_path):
        os.remove(out_path)

    with rasterio.open(in_path) as src:
        red = src.read(RED_BAND).astype(np.float32)
        nir = src.read(NIR_BAND).astype(np.float32)

        # Simple scale detection (PlanetScope SR often 0..10000)
        max_val = max(float(np.nanmax(red)), float(np.nanmax(nir)))
        if max_val > 2.0:
            red /= 10000.0
            nir /= 10000.0

        denom = nir + red
        ndvi = (nir - red) / np.where(denom == 0, np.nan, denom)
        ndvi = ndvi.astype(np.float32)
        ndvi[~np.isfinite(ndvi)] = -9999.0

        # Build a minimal, clean output profile (striped TIFF, no compression)
        out_profile = {
            "driver": "GTiff",
            "height": src.height,
            "width":  src.width,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": np.float32(-9999.0),
            "tiled": False,                 # STRIPED
            "rowsperstrip": min(512, src.height),
            "BIGTIFF": "IF_SAFER",
            # no 'compress', no 'predictor' -> avoids edge cases creating empty strips
        }

        with rasterio.open(out_path, "w", **out_profile) as dst:
            dst.write(ndvi, 1)
            dst.update_tags(1, long_name="NDVI", formula="(NIR-Red)/(NIR+Red)")
            dst.update_tags(SENSOR="PlanetScope", INDEX="NDVI")

    print(f"[OK] {os.path.basename(in_path)} -> {os.path.basename(out_path)}")

def main():
    files = sorted(glob(os.path.join(INPUT_DIR, PATTERN)))
    if not files:
        print(f"No files matched: {os.path.join(INPUT_DIR, PATTERN)}")
        return
    for f in files:
        try:
            make_ndvi(f)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    main()