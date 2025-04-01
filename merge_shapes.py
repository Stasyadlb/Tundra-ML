import os
import xarray as xr
import rioxarray
from glob import glob
import natsort
import pandas as pd
import geopandas as gpd
import numpy as np

all_points = pd.read_excel("G:/TundraVegetationProject/data/BB_model_points_shp/BB_model_10.xlsx")
points_2class = gpd.read_file('G:/TundraVegetationProject/data/BB_model_points_shp/BB_data_in_points_2C.shp')
ds = pd.DataFrame({"point_id": 9900 + points_2class.index, "LATITUDE": points_2class.geometry.y, "LONGITUDE": points_2class.geometry.x})
bb_model = pd.concat([all_points, ds])
bb_model.to_excel("G:/TundraVegetationProject/data/BB_model_points_shp/BB_model_11.xlsx")

file = pd.read_excel("G:/TundraVegetationProject/data/BB_model_points_shp/BB_model_11.xlsx")
xy = gpd.points_from_xy(file.LONGITUDE, file.LATITUDE)
gdf = gpd.GeoDataFrame(file, geometry=xy, crs='EPSG:4326')
gdf.to_file("G:/TundraVegetationProject/data/BB_model_points_shp/BB_model_11_points.shp")
