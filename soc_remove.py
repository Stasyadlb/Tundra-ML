import os
import xarray as xr
import rioxarray
from glob import glob
import natsort
import pandas as pd
import geopandas as gpd
import numpy as np

path = "G:/TundraVegetationProject/data/SOC/TUW_ASARGM_SOC.tif"
ds = xr.open_dataset(path, engine="rasterio")

x = xr.DataArray([1, 2], dims=['band'])
soc = ds.sel(band=x, method='nearest')

atribs = soc.band_data.attrs.items()
soc.band_data.attrs={}

soc.band_data.rio.to_raster("G:/TundraVegetationProject/data/SOC/TUW_ASARGM_SOC_12.tif")