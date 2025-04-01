import os
import xarray as xr
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import natsort
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns


# загрузка указанного в пути shape-файла
X = gpd.read_file("D:/TundraVegetationProject/data/BB_model_points_shp/BB_model_data_13.shp")
# Выбор столбцов с числовыми данными для анализа
df = X.loc[:, X.columns[19:-1]]
# Выбор столбца с номерами классов/порядков/союзов/альянсов/ассоциаций
names = X['CLASS_NR']

# рассчет 2D матрицы TSNE для каждого класса
tsne = TSNE(n_components=2)
s2 = tsne.fit_transform(df.values)
plt.figure()
scatter = plt.scatter(s2[:, 0], s2[:, 1], c=names, cmap='tab20')
plt.colorbar(scatter)
for land_class in X['CLASS_NR'].unique():
    X[f'color_{land_class}'] = X['CLASS_NR'].apply(lambda x: x if x == land_class else 0)
    names = X[f'color_{land_class}']
    tsne = TSNE(n_components=2)
    s2 = tsne.fit_transform(df.values)
    plt.figure()
    scatter = plt.scatter(s2[:, 0], s2[:, 1], c=names, cmap='Paired', alpha=0.5)
    plt.colorbar(scatter)
    plt.show()

# рассчет 3D матрицы TSNE 
tsne = TSNE(n_components=3)
s3 = tsne.fit_transform(df.values)
fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'3d'})
ax.scatter(s3[:, 0], s3[:, 1], s3[:, 2], c=names, cmap='tab20', alpha=0.8, s=15)
ax.view_init(azim=60, elev=9)
plt.show()

# рассчет PSA
pca = PCA(n_components=10)
principal_components = pca.fit_transform(df.values)
components = pca.components_
plt.figure(figsize=(19,4))
sns.heatmap(components, annot=False, cmap='jet', linewidths=0.5, xticklabels=df.columns, yticklabels=[f'Components {i + 1}' for i in range(10)])
# plt.title('')
plt.xlabel('Features')
plt.ylabel('Principal Components')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()