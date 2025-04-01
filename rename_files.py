import os
from glob import glob

# путь к файлам
path = 'G://TundraVegetationProject/zenodo/'
# маска имени файлов
files_dir = glob(path + 'tile_name*')
for file_dir in files_dir:
    file = file_dir.split('\\')[-1].split('.')
    
    # переименование файла в нужный вид
    file[0] = file[0].replace('tile_name', 'Tile_name')

    os.rename(file_dir, f'{path}{'.'.join(file)}')