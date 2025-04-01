import os
import xarray as xr
import rioxarray
from glob import glob
import natsort
import pandas as pd
import geopandas as gpd
import numpy as np


def get_data_ManyTiles(folder_path, file_name_mask, points, col_name, id_col='id', bands_num=1):
    '''
    Get the values from a raster stored a set of files for not-merged tiles.

    folder_path: string
    file_name_mask: string, -glob- lib compatible
    points: geopandas dataframe
    col_name: string, name for the new column added in -points-
    id_col: string, name for -points- column that contains a unique identifier
    '''

    # duplicate field and set as index for further matching
    # points['ind'] = points[id_col]
    # points = points.set_index('ind')
    # create container
    col_name_list = []
    if bands_num == 1:
        points[col_name] = np.nan
        col_name_list.append(col_name)
    else:
        for i in range(bands_num):
            points[col_name + f'_{i + 1}'] = np.nan
            col_name_list.append(col_name + f'_{i + 1}')


    flist = natsort.natsorted(glob(folder_path + file_name_mask))
    print('Find files: ', len(flist))
    for f in flist:
        print(f)
        try:
            ds = xr.open_dataset(f, engine="rasterio")  # chunks='auto') ) # geo_py37 and old xarray (v.0.**)
        except:
            try:
                ds = rioxarray.open_rasterio(f) # geo_py312 and new xarray (v.202*)
            except:
                print(f'ERROR {f}')

        # reproject points and get points in raster extent // CRS = Coordinates Reference System
        crs_raster = ds.rio.crs
        df_loc = points.to_crs(crs_raster).cx[ds.x.values.min():ds.x.values.max(),
                                              ds.y.values.min():ds.y.values.max()]
        # get coordinates as DataArray
        x_loc = xr.DataArray(df_loc.geometry.x, dims=['location'])
        y_loc = xr.DataArray(df_loc.geometry.y, dims=['location'])
        # get data values in points
        data = ds.sel(x=x_loc, y=y_loc, method='nearest')
        #if data.shape != (1, 0):      
        # The number of repetitions of the cycle depends on the number of bands 
        for i, col_name_i in zip(data.band, col_name_list):
            try:
                data_i = data.sel(band=i)
                df_loc = data_i.to_dataframe().reset_index()
                df_loc = df_loc.drop(columns=['band', 'x','y','spatial_ref'])
                # append values to the main points list
                points = pd.merge(points, df_loc, how='left', left_on=points.index, right_on='location')
                points.loc[points[col_name_i].isna(), col_name_i] = points['band_data']
                points.drop(columns=['location', 'band_data'], axis=1, inplace=True)
            except:
                print('     check me: ', f)
    print('     load done: ', col_name)
    if points[col_name_list[0]].isna().any():
        print('     WARNING! Empty points: ', points[col_name_list[0]].isna().value_counts().values[1])
    return points

def get_data_Pack(folder_path, points, file_name_mask = '*.tif', id_col='id'):
    '''
    Get the values from many thematic rasters stored as full-cover files in the same folder.

    folder_path: string
    file_name_mask: string, -glob- lib compatible
    points: geopandas dataframe
    id_col: string, name for -points- column that contains a unique identifier
    '''

    # duplicate field and set as index for further matching
    # points['ind'] = points[id_col]
    # points = points.set_index('ind')

    flist = natsort.natsorted(glob(folder_path + file_name_mask))
    print('Find files: ', len(flist))
    for f in flist:
        print(f)
        try:
            ds = xr.open_dataset(f, engine="rasterio")  # chunks='auto') ) # geo_py37 and old xarray (v.0.**)
        except:
            ds = rioxarray.open_rasterio(f) # geo_py312 and new xarray (v.202*)
            print(f'    Warning{f}')
        

        # reproject points and get points in raster extent // CRS = Coordinates Reference System
        crs_raster = ds.rio.crs
        df_loc = points.to_crs(crs_raster).cx[ds.x.values.min():ds.x.values.max(),
                                              ds.y.values.min():ds.y.values.max()]
        # get coordinates as DataArray
        x_loc = xr.DataArray(df_loc.geometry.x, dims=['location'])
        y_loc = xr.DataArray(df_loc.geometry.y, dims=['location'])
        # get data values in points
        data = ds.sel(x=x_loc, y=y_loc, method='nearest')
        # The number of repetitions of the cycle depends on the number of bands 
        for i in data.band:
            try:
                data_i = data.sel(band=i)
                # create column container
                if len(data.band) == 1:
                    col_name = os.path.basename(f).split('.')[0]
                    points[col_name] = np.nan
                else:
                    col_name = os.path.basename(f).split('.')[0] + f'{int(i)}'
                    points[col_name] = np.nan
                df_loc = data_i.to_dataframe().reset_index()
                df_loc = df_loc.drop(columns=['band', 'x','y','spatial_ref'])
                # append values to the main points list
                points = pd.merge(points, df_loc, how='left', left_on=points.index, right_on='location')
                points.loc[points[col_name].isna(), col_name] = points['band_data']
                points.drop(columns=['location', 'band_data'], axis=1, inplace=True)
                if points[col_name].isna().any():
                    print('     WARNING! Empty points: ', col_name, points[col_name].isna().value_counts().values[1])
            except:
                print('     check me: ', f)

    print('     load done: ', folder_path)
    return points


### ======================================


# case of many not merged rasters for one dataset (e.g. ArcticDEM)
# set paths
# disk = 'G' # H F D
base_path = 'H:/TundraVegetationProject/data/'
points_shp_path = base_path + 'BB_model_points_shp/extract_2nd/BB_model_13_2ndCLASS.shp'

# open shp
points = gpd.read_file(points_shp_path)
print('open shp ok')
# get data from datasets
# ... add all datasets
folder_path = base_path + 'ArcticDEM/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'CALC_2020_RU/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'CHELSA/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'CHELSA_paleo/'
points = get_data_Pack(folder_path, points)

######### gald #########
dem_folder_path = base_path + "/GLAD/time1/10yrMed_splitLikeArcticDEM/"
dem_name_mask = '*time1*.tif'
col_name = 'GLAD_T1'
print(dem_folder_path)
points = get_data_ManyTiles(dem_folder_path, dem_name_mask, points, col_name, id_col='id', bands_num=7)

dem_folder_path = base_path + "/GLAD/time2/10yrMed_splitLikeArcticDEM/"
dem_name_mask = '*time2*.tif'
col_name = 'GLAD_T2'
print(dem_folder_path)
points = get_data_ManyTiles(dem_folder_path, dem_name_mask, points, col_name, id_col='id', bands_num=7)

dem_folder_path = base_path + "/GLAD/time3/10yrMed_splitLikeArcticDEM/"
dem_name_mask = '*time3*.tif'
col_name = 'GLAD_T3'
print(dem_folder_path)
points = get_data_ManyTiles(dem_folder_path, dem_name_mask, points, col_name, id_col='id', bands_num=7)
############################

folder_path = base_path + 'GlobalSurfaceWater/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'Hydrology/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'MODIS/'
points = get_data_Pack(folder_path, points)

######### sentinel #########
dem_folder_path = base_path + 'Sentinel_1/'
dem_name_mask = 'S1_VV*.tif'
col_name = 'S1_VV'
print(dem_folder_path)
points = get_data_ManyTiles(dem_folder_path, dem_name_mask, points, col_name, id_col='id')

dem_folder_path = base_path + 'Sentinel_1/'
dem_name_mask = 'S1_VH*.tif'
col_name = 'S1_VH'
print(dem_folder_path)
points = get_data_ManyTiles(dem_folder_path, dem_name_mask, points, col_name, id_col='id')
############################

folder_path = base_path + 'SOC/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'Soil_Sediment/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'UiO_PEX_MAGTM/'
points = get_data_Pack(folder_path, points)

folder_path = base_path + 'Vegetation height/'
points = get_data_Pack(folder_path, points)

# save the final dataframe to reuse it
saveto = base_path + 'BB_model_points_shp/extract_2nd/BB_model_data_13_2CLASS.shp'
points.to_file(saveto)


##########################

# disk = 'D' # H F D
# base_path = disk + ':\TundraVegetationProject\data\MODIS\Clip/'
# points_shp_path = base_path + 'BB_model_points.shp'

# # open shp
# points = gpd.read_file(points_shp_path)
# # get data from datasets
# # folder_path = 'H:\_HSE\TundraVegetationProject\Chelsa/'
# points = get_data_Pack(base_path, points)

# points.boxplot(column=points.columns[14], by='CLASS_NR')