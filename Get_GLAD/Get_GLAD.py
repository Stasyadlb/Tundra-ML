import os
import xarray as xr
import rioxarray
from glob import glob
import natsort
import pandas as pd
import geopandas as gpd
import numpy as np
import requests


disk = 'D' # 'H'
#base_path = disk + ':\_HSE\TundraVegetationProject\Landsat_Global_ARD_tiles/'
base_path = disk + ':/TundraVegetationProject/data/GLAD/'

tiles_list = pd.read_csv(base_path + 'GLAD_tiles_list.csv', sep=';')
time_list1 = pd.read_csv(base_path + 'GLAD_time_list_1.csv', header=None)[0].to_numpy()
time_list2 = pd.read_csv(base_path + 'GLAD_time_list_2.csv', header=None)[0].to_numpy()
time_list3 = pd.read_csv(base_path + 'GLAD_time_list_3.csv', header=None)[0].to_numpy()


def download_tif(base_path, tile_name, interval):
    proxies = { "http": None,
                "https": None, }

    # https://glad.umd.edu/dataset/glad_ard2/<lat>/<tile>/<interval>.tif
    base_url = 'https://glad.umd.edu/dataset/glad_ard2/'
    username = 'glad'
    password = 'ardpas'

    file_name = '{tile}_t{interval}.tif'.format(tile = tile_name, interval = interval)
    print(file_name)
    url = base_url + '{lat}/{tile}/{interval}.tif'.format(lat = tile_name.split('_')[1],
                                                       tile = tile_name,
                                                       interval = interval)

    if os.path.exists(base_path + '/SrcTiles/' + file_name):
        pass
    else:
        # Путь к сохраненному архиву
        save_path = os.path.join(base_path + '/SrcTiles/', file_name)
        # Скачивание архива
        response = requests.get(url, proxies=proxies, auth=(username, password))
        print('     resp:', response.status_code)
        with open(save_path, 'wb') as f:
            f.write(response.content)

def create_quality_mask(QA_band):
    QA_band = QA_band.astype(np.uint16)
    #cloud_mask = ( (QA_band & (1<<4))!=0 ) & ( (QA_band & (1<<5))!=0 )
    #cloud_shadow_mask = ( (QA_band & (1<<7))!=0 )
    #mask = cloud_mask | cloud_shadow_mask

    ##mask = ( QA_band != 1)

    # Create masks for all problematic flags
    cloud_mask = (QA_band == 3) # Cloud
    cloud_shadow_mask = (QA_band == 4)  # Cloud shadow
    cloud_shadow_mask2 = (QA_band == 10)  # Cloud shadow
    cloud_shadow_mask3 = (QA_band == 8)  # Cloud shadow
    cirrus_mask = (QA_band == 9) # Cirrus
    # Combine all problematic masks
    mask = cloud_mask | cloud_shadow_mask | cloud_shadow_mask2 | cloud_shadow_mask3 | cirrus_mask

    return mask

def apply_quality_mask(dataset):
    # Assuming the quality band is the 8th band (index 7)
    QA_band = dataset.isel(band=7)
    quality_mask = create_quality_mask(QA_band)
    # Mask the dataset where the quality mask is True
    dataset_masked = dataset.where(~quality_mask)
    return dataset_masked

def interannual_median(base_path_time,tile_name,time_name):
    # calculate interannual median per band
    file_name = '{tile}_{time_name}.tif'.format(tile=tile_name, time_name=time_name)
    saveto = base_path_time + '/' + file_name

    if not os.path.exists(base_path_time + '/' + file_name):

        flist = glob(base_path_time + '/SrcTiles/' + tile_name + '*_masked.tif')

        # load all tifs simultaneously
        ds = xr.open_mfdataset(flist, combine='nested', concat_dim='time')
        # check the band8 with quality flags
        ds_masked = apply_quality_mask(ds)

        med_img = ds_masked.median(dim='time')
        # force the xarray layzy mode to run computations for the next step
        med_img = med_img.compute()
        # delete QA band wich is uncorrest now
        med_img = med_img.drop_sel(band=8)

        # convert dataset to dataarray for managing rio.to_raster error
        med_img = med_img.to_array(dim='variable').squeeze()
        med_img.rio.to_raster(saveto, compress='zstd', tiled=True)
        print ('>>> save interannual image to: ', saveto)


###########################################

### RUN THE CODE ###
for time_name, time_list in zip(['time1', # week_ID=12 25-Jun 10-Jul
                                  'time2', # week_ID=13 11-Jul 26-Jul
                                  'time3'], # week_ID=14 27-Jul 11-Aug
                                   [time_list1, time_list2, time_list3]):
    base_path_time = base_path + '/' + time_name
    if not os.path.exists(base_path_time + '/SrcTiles/'):
        os.mkdir(base_path_time)
        os.mkdir(base_path_time + '/SrcTiles/')

    print('start loading ', base_path_time)
    # For each tile
    for index, row in tiles_list.iterrows():
        tile_name = row['TILE']
        # get tif for each time moment
        for interval in time_list:
            download_tif(base_path_time, tile_name, interval)

        # calculate the interannual median-per-band image
        interannual_median(base_path_time, tile_name, time_name)

