# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:26:01 2024

@author: Robbe Neyns
"""

import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import os
import pandas as pd

# Path to the shapefile containing polygons
shapefile_path = r'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/polygon training set/corrected_salix_polygons_one_to_one.shp'
shapefile_path = r'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/Lidar Braunsweig/braunschweig_tree_models_lidar_2019/crowns/merged.shp'
# Path to the raster file
#raster_path = r'C:\Users\Robbe_Neyns\OneDrive - Vrije Universiteit Brussel\Documenten\Research\Bee mapping\Images Freiburg\Merged images\01_10_23_ndvi.tif'


raster_folder = r'C:\Users\Robbe_Neyns\OneDrive - Vrije Universiteit Brussel\Documenten\Research\Bee mapping\Images Freiburg\Merged images'
raster_folder = r'E:\Research\Bee mapping\Images Brunswick\Brunswick_merged'

# Function to calculate the average value of a raster within a polygon
def zonal_stats(polygon, raster):
    with rasterio.open(raster) as src:
        try:
            # Read the raster data within the polygon
            geom = polygon['geometry']
            ID = polygon['treeID']
            out_image, out_transform = mask(src, geom, crop=True)
            out_image = np.array(out_image,dtype=float)
            # Set all 0 values to NaN
            out_image[out_image == 0] = np.nan
            
            avg_values_per_band = np.nanmean(out_image, axis=(1, 2))
        except:
            # Get the dimensions of the raster (number of bands, height, width)
            bands = src.count
            ID = polygon['treeID']
            # Create an array filled with zeros, matching the dimensions of the raster
            avg_values_per_band = np.zeros((bands), dtype=src.dtypes[0])
            
    return avg_values_per_band, ID



def sample_raster(shapefile_path, raster_path):
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    sampled = []
    IDs = []
    # Iterate through each polygon and calculate the average value
    for index, row in gdf.iterrows():
        polygon = gdf.iloc[[index]]
        avg_value, ID = zonal_stats(polygon, raster_path)
        sampled.append(avg_value)
        IDs.append(ID)
    return np.array(sampled), IDs
    
def sample_rasters(shapefile_path, raster_folder):
    sampled_all = {}
    for raster in os.listdir(raster_folder):
        print(raster)
        if not 'ndvi' in raster:
            try:
                full_path = raster_folder + '/' + raster
                sampled, IDs = sample_raster(shapefile_path, full_path)
                print(f"sampled shape = {sampled.shape}")
                for i in range(len(sampled[1,:])):
                    sampled_all[raster[:-8] +"2022." + str(i)] = sampled[:,i]
            except Exception as error:
                print(raster + " could not be opened")
                print(error)
    df = pd.DataFrame(sampled_all) 
    # Split column names into date and suffix, converting the date part to datetime and keeping the suffix as float for sorting
    columns_with_date_suffix = [(pd.to_datetime(col.split('.')[0], format='%d_%m_%Y'), int(col.split('.')[1])) for col in df.columns]
    
    print(columns_with_date_suffix)
    # Generate a sorted list of columns, considering both date and suffix
    sorted_columns = sorted(columns_with_date_suffix, key=lambda x: (x[0], x[1]))
    # Convert sorted tuples back to original column format
    sorted_column_names = [f"{dt.strftime('%d_%m_%Y')}.{suffix}" for dt, suffix in sorted_columns]

    # Reindex the DataFrame to order the columns
    df = df.reindex(columns=sorted_column_names)
    df['id'] = IDs
    df.to_csv('braunsweigh_sample_apply_polys.csv')

sample_rasters(shapefile_path, raster_folder)