import os
from glob import glob

# путь с файлами 
path = 'D:/TundraVegetationProject/GLAD/time1/SrcTiles/'
# маска имени файлов, которые необходимо переместить
files_dir = glob(path + '*.tif')

for file_dir in files_dir:
    # новый путь 
    new_path = path + file_dir.split('_')[-1].split('.')[0] + '/'
    file_name = file_dir.split('\\')[-1]
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    os.rename(file_dir, new_path + file_name)