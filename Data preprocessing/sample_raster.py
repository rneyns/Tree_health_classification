# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 12:38:28 2023

@author: Robbe Neyns
"""

import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import os 
import pandas as pd
import numpy as np
from osgeo import gdal


# Step 1: Load Shapefile and Raster
#shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data hannah/Baum_GruenInfo_JKI_240219/Baum_GruenInfo_id.shp'
shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data hannah/Baum_GruenInfo_JKI_240219/Baum_GruenInfo_id_corrected.shp'
#shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data_berlin/clipped_merged/berlin_clip_merge_id.shp'
#shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data Markus/municipal_tree_inventories/Freiburg_selected_id.shp'
#shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data Markus/municipal_tree_inventories/Freiburg_selected_id.shp'
#shapefile_path = "C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/data Markus/municipal_tree_inventories/baeume_mittlerer_ring_reproject.shp"
raster_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/Images Brunswick/Brunswick_merged/01_03_2023.tif'
raster_folder = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/Images Munich/merged_all/ndvi/2023'
raster_folder = 'E:\Research\Bee mapping\Images Brunswick\Brunswick_merged'

#Freiburg new
#raster_folder ="E:/Research/Bee mapping/Images Freiburg/merged auto"

#Braunsweigh appy
#raster_folder ="E:/Research/Bee mapping/Images Brunswick/Brunswick_merged"
#shapefile_path = 'C:/Users/Robbe_Neyns/OneDrive - Vrije Universiteit Brussel/Documenten/Research/Bee mapping/Lidar Braunsweig/braunschweig_tree_models_lidar_2019/crowns/Centroids_id_clipped.shp'



def search_highest_reflectance(ds_array, i, j, objn):
  cutout = ds_array[:,i-1:i+2,j-1:j+2]
  try:
      if len(cutout[:,1,1]>4):
          ndvi = (cutout[7]-cutout[5])/(cutout[7]+cutout[5])
      else:
          ndvi = (cutout[3]-cutout[2])/(cutout[3]+cutout[2])
    
      ndvi[ndvi < 0.1] = 0
      ndvi[ndvi >= 0.1] = 1
      cutout = cutout * ndvi
  except:
      print('ndvi calculation not possible')
  cutout = cutout.sum(axis=0)
  # multiply the cutout with the ndvi mask
  i_c,j_c,_ = cutout.argmax(axis=0)
  i = (i-1)+i_c
  j = (j-1)+j_c
  
  return i, j


def get_indices(x, y, ox, oy, pw, ph):
    """
    Gets the row (i) and column (j) indices in an array for a given set of coordinates.
    Based on https://gis.stackexchange.com/a/92015/86131

    :param x:   array of x coordinates (longitude)
    :param y:   array of y coordinates (latitude)
    :param ox:  raster x origin
    :param oy:  raster y origin
    :param pw:  raster pixel width
    :param ph:  raster pixel height
    :return:    row (i) and column (j) indices
    """

    i = np.floor((oy-y) / ph).astype('int')
    j = np.floor((x-ox) / pw).astype('int')

    return i, j


def sample_points(x, y, ds, objn,ndvi=False):
    xmin, xres, xskew, ymax, yskew, yres = ds.GetGeoTransform()
    
    i,j = get_indices(x, y, xmin, ymax, xres, -yres)
    arr = ds.ReadAsArray()
    #i_2 = i
    #j_2 = j
    
    #retrieve the index from the highest brightness pixel
    #for k in range(len(j)):
     #   i_2[k],j_2[k] = search_highest_reflectance(arr,i[k],j[k],objn[k])
    if ndvi:
        sampled = arr[i,j]
    else:
        sampled = arr[:,i,j]
    
    return sampled

def sample_raster(shapefile_path, raster_path, ndvi=False):
    gdf = gpd.read_file(shapefile_path)
    # Assuming 'geometry' column contains Point geometries
    x_coords = gdf['geometry'].x
    y_coords = gdf['geometry'].y
    anlag_objn = gdf['id']

    #with gdal.Open(raster_path,0) as src: 
    ds = gdal.Open(raster_path, 0)
    sampled = sample_points(x_coords, y_coords, ds, anlag_objn,ndvi)
    del ds
        
    return sampled

def sample_rasters(shapefile_path, raster_folder,ndvi=False):
    sampled_all = {}
    for raster in os.listdir(raster_folder):
        full_path = raster_folder + '/' + raster
        print(f"Working on raster {raster}")
        try:
            sampled = sample_raster(shapefile_path, full_path, ndvi)
            if ndvi:
                sampled_all[raster[:-13]+"2022"] = sampled
                print(raster[:-13]+"2022")
            else:
                for i in range(len(sampled[:,1])):
                    sampled_all[raster[:-8] +"2022." + str(i)] = sampled[i,:]
        except Exception as error:
            print(raster + " could not be opened")
            print(error)
    df = pd.DataFrame(sampled_all) 
    #Order the columns 
    if ndvi:
        #  converting the date part to datetime 
        columns_with_date_suffix = [pd.to_datetime(col, format='%d_%m_%Y') for col in df.columns]

        # Generate a sorted list of columns, considering both date and suffix
        sorted_columns = sorted(columns_with_date_suffix)
    
        # Convert sorted tuples back to original column format
        sorted_column_names = [f"{dt.strftime('%d_%m_%Y')}" for dt in sorted_columns]
    else:
        # Split column names into date and suffix, converting the date part to datetime and keeping the suffix as float for sorting
        columns_with_date_suffix = [(pd.to_datetime(col.split('.')[0], format='%d_%m_%Y'), int(col.split('.')[1])) for col in df.columns]
        
        print(columns_with_date_suffix)
        # Generate a sorted list of columns, considering both date and suffix
        sorted_columns = sorted(columns_with_date_suffix, key=lambda x: (x[0], x[1]))
        # Convert sorted tuples back to original column format
        sorted_column_names = [f"{dt.strftime('%d_%m_%Y')}.{suffix}" for dt, suffix in sorted_columns]
    
    # Reindex the DataFrame to order the columns
    df = df.reindex(columns=sorted_column_names)

    df.to_csv('Braunsweigh_cleaned.csv')

#print(sample_raster(shapefile_path,raster_path).shape)
#sample_raster(shapefile_path,raster_path)
sample_rasters(shapefile_path,raster_folder,ndvi=False)

  
# Step 6: Optional - Save Results

