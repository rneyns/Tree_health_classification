"""
Script: filter_missing_images_batch.py

Description:
    This script processes all CSV files in a specified folder, each containing
    an ID column. For each CSV:
      - Checks whether a corresponding .tif image exists in a given images folder.
      - Keeps only rows where the image exists.
      - Saves a new CSV with "_filtered" appended to the original filename.
      - Prints how many rows were deleted for each file.

    All CSVs are expected to have the same ID column name.
    Image filenames must match the ID exactly (e.g., ID: 12345 → file: 12345.tif).

Usage:
    1. Update the CONFIGURATION section with:
       - Path to folder containing your CSV files
       - Path to your images folder
       - Name of the ID column in your CSVs
    2. Run with:
       python filter_missing_images_batch.py
"""

import pandas as pd
import os

# === CONFIGURATION ===
csv_folder = "/Users/robbe_neyns/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/Documenten/Research/UHI_tree health/Data analysis/Tree mapping/PlanetScope_csvs/5 species and other"     # folder containing all CSV files
images_folder = "/Users/robbe_neyns/Documents/research/UHI tree health/output_patches"      # folder containing .tif images
id_column = "tree_id"                      # column name in all CSVs

# === LOAD EXISTING IMAGE IDS ===
existing_ids = set()
for fname in os.listdir(images_folder):
    if fname.lower().endswith(".tif"):
        existing_ids.add(os.path.splitext(fname)[0])  # filename without extension

print(f"Found {len(existing_ids)} image files in {images_folder}")

# === PROCESS EACH CSV ===
for file in os.listdir(csv_folder):
    if file.lower().endswith(".csv"):
        csv_path = os.path.join(csv_folder, file)
        df = pd.read_csv(csv_path)

        initial_count = len(df)
        df_filtered = df[df[id_column].astype(str).isin(existing_ids)]
        deleted_count = initial_count - len(df_filtered)

        output_path = os.path.join(csv_folder, f"{os.path.splitext(file)[0]}_filtered.csv")
        df_filtered.to_csv(output_path, index=False)

        print(f"✅ {file} → saved {output_path} ({deleted_count} rows deleted)")
