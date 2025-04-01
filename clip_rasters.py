import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# Путь к папке с TIF-файлами и SHP-файлу
base_path = 'G:/TundraVegetationProject/'
folder_path = base_path + 'data_modeling/'
shp_file = base_path + 'data/Hydrology/coastline_shp/OSM_Coastline_divided_plg_prj.shp'

# Чтение shp-файла с помощью GeoPandas
shapefile = gpd.read_file(shp_file)
shapes = shapefile.geometry

# Функция для обрезки tif-файлов по shp-файлу
def crop_tif_by_shp(tif_file, shapes, out_folder):
    with rasterio.open(tif_file) as src:
        profile = src.profile
        compression = profile.get('compress', 'none')  # Параметр сжатия
        dtype = profile['dtype']  # Тип данных
        
        # Обрезаем изображение по границам shp
        out_image, out_transform = mask(src, shapes, crop=True, all_touched=True)
        
        # Обновляем метаданные для выхода
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": compression,  # Сохранение исходного параметра сжатия
            "dtype": dtype  # Сохранение исходного формата числа
        })
        
        # Путь для сохранения обрезанного изображения
        output_file = os.path.join(out_folder, os.path.basename(tif_file))
        
        # Сохранение результата
        with rasterio.open(output_file, 'w', **out_meta) as dest:
            dest.write(out_image)
            
        print(f"Файл {tif_file} успешно обрезан и сохранен как {output_file}")

# Проходим по всем файлам в папке
for tif_folder in os.listdir(folder_path):
    out_folder = folder_path + tif_folder + '_clip'
    os.makedirs(out_folder)
    for tif_file in os.listdir(f'{folder_path}/{tif_folder}'):
        if tif_file.endswith('.tif') and not os.path.exists(os.path.join(out_folder, os.path.basename(tif_file))):
            full_tif_path = os.path.join(f'{folder_path}/{tif_folder}', tif_file)
            crop_tif_by_shp(full_tif_path, shapes, out_folder)
