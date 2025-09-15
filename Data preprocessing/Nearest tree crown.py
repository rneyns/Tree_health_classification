"""
Tree Crown Species Assignment (Point-in-Polygon Filter)
-------------------------------------------------------

Description:
    This script matches tree crown polygons with tree species information
    from a point layer of tree locations. Only polygons that contain at least
    one point are kept. Each polygon is assigned the species of the point
    inside it.

Workflow:
    1. Load tree crown polygons (crowns) and tree location points (trees).
    2. Ensure both layers use the same coordinate reference system (CRS).
    3. Perform a spatial join to find tree points that fall within each polygon.
    4. Keep only polygons that contain at least one point.
    5. Assign the species of the point(s) to the polygon.
       - If multiple points fall inside one polygon, the first match is used.
    6. Save the filtered polygons with species information.

Requirements:
    - Python 3.x
    - GeoPandas
    - Shapely

Output:
    A new shapefile (tree_crowns_with_species.shp) containing only polygons
    that had at least one tree point inside them, with an added "species" attribute.

Author: Robbe Neyns
Date: 15-09-2025
"""


import geopandas as gpd

# 1. specify the layer paths
crowns_path = "/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/crown_shapes_final_CRS.shp"
trees_path = '/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/Brussels Environment Layers/mobiliteit_shape_manual_adjustment_project/mobiliteit_shape_manual_adjustment_X_Y.shp'
# 2. Load polygons and points
crowns = gpd.read_file(crowns_path)
trees = gpd.read_file(trees_path)

# 3. Spatial join: find tree points that fall within each polygon
#    This will duplicate polygons if multiple points fall inside
crowns_with_points = gpd.sjoin(crowns, trees[['geometry', 'essence']], how="inner", predicate="contains")

# 4. If multiple points fall inside one polygon, keep only the first
#    (you can change this aggregation logic if needed)
crowns_unique = crowns_with_points.drop_duplicates(subset=crowns.columns.drop('geometry'))

# 5. Save the result
crowns_unique.to_file("/Users/robbe_neyns/Documents/Work_local/research/UHI tree health/Data analysis/Tree mapping/Tree locations/flai layers/tree_crowns_with_species.shp")

print("✅ Saved polygons that contain at least one point with species info!")
