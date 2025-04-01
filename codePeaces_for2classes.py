import os
# import osgeo
import xarray as xr
from glob import glob
import pandas as pd
import geopandas as gpd # if "ImportError: DLL ..." > pip uninstall pyproj && pip install pyproj
                        # if "ERROR 1:" > import osgeo (gdal) BEFORE import lib
import numpy as np
import natsort
# for model creation and train
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt  # mamba update Pillow or mamba update -c conda-forge matplotlib
#plt.interactive(True)
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import shapely
import shap
from collections import Counter
# imbalanced learn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import cross_validate
# for model save
import joblib


def make_binary_dataset(df_train, df_test, i):
    # get only target class
    df_train_loc = df_train.copy(True)
    df_train_loc.loc[df_train_loc['NR'] != i, 'NR'] = 0
    df_test_loc = df_test.copy(True)
    df_test_loc.loc[df_test_loc['NR'] != i, 'NR'] = 0
    X, y = df_train_loc.drop(columns=['NR']).values, df_train_loc['NR'].values
    X_test, y_test = df_test_loc.drop(columns=['NR']).values, df_test_loc['NR'].values

    # for undersampling step
    k=0.5 # non-proved choice just to try. seems to be ok for many cases
    under_size = int(len(df_train_loc[df_train_loc['NR'] != i]) * k)

    return X, y, X_test, y_test, under_size

def make_new_brf():
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0,
                                         class_weight='balanced_subsample',
                                         max_features=0.95,  # None, #'sqrt', # None
                                         sampling_strategy="all", replacement=True,
                                         oob_score=True, bootstrap=True)
    return brf

def fix_ALLIANCE_N(assot):
    parts = assot.split('_')
    alliance = '_'.join(parts[:-1])
    return alliance

#########################
gdf = gpd.read_file('E:/TundraVegetationProject/data/BB_model_points_shp/extract_2nd/BB_model_data_13_2CLASS.shp')
gdf_old = gpd.read_file('E:/TundraVegetationProject/data/BB_model_points_shp/BB_model_data_13.shp')
gdf_old = gdf_old[gdf_old['CLASS_NR'] != 2]
gdf = pd.concat([gdf_old, gdf], ignore_index=True)
gdf = gdf[(gdf['point_id'] != 1868) & (gdf['point_id'] != 9925)]
gdf['ALLIANCE_N'] = gdf['ASSOCIATIO'].apply(fix_ALLIANCE_N)

# manage the working level
level = 'ORDER_NR'  # 'CLASS_NR' 'ORDER_NR' 'ALLIANCE_N' 'ASSOCIATIO'
drop_list = ['Unnamed__0', 'point_id', 'FIELD_NR', 'LATITUDE', 'LONGITUDE',
               'CLASS_NR', 'CLASS_NAME',
            'ORDER_NR', 'ORDER_NR_X', 'ORDER_NR_Y', 'ORDER_NAME',
          'ALLIANCE_N', 'ALLIANCE_1',
          'ASSOCIATIO', 'ASSOCIAT_1', 'ASSOCIAT_2', 'ASSOCIAT_3', 'SUBASSOCIA', 'SUBASSOC_1',
          'geometry']
drop_list.remove(level)
df = pd.DataFrame(gdf.drop(columns=drop_list))
# replace VEGETATION data : -300 = 16001, -200 = 0, -100 = 0
df['Vegetation'] = df['Vegetation'].replace([-300, -200, -100], [16001, 0, 0])

dff = df.copy(True)
# convert text *_*_* into numeric ****
dff['NR'] = dff[level].astype(str).str.replace('_','').astype(int)

# drop association with only 1 sample which riase an error of train-test split
if level == 'ALLIANCE_N':
    dff = dff.drop( dff[dff[level]=='5_1_1_12'].index )
################################

#####################
# when CLASS level, it will be used
sampling_strategy_over= {2:40, 3:50, 4:50, 7:40, 8:50, 10:50, 11:50}


df_train, df_test = train_test_split(dff, test_size=0.2, random_state=42, stratify=dff['NR'])


# to collect prediction PROBABILITY later
dfp = pd.DataFrame({'id': df_test.index, level:df_test[level], 'NR': df_test['NR']})
# make 1-vs-others dataset and train model with it
for i in dff[level].sort_values().unique():
    ii = int( str(i).replace('_',''))

    print(' ')
    print('### >>> {} #'.format(level), i)

    X, y, X_test, y_test, under_size = make_binary_dataset(df_train.drop(columns=[level]), df_test.drop(columns=[level]), ii)

    # print('Train')
    # print(sorted(Counter(y).items()))
    # print('Test')
    # print(sorted(Counter(y_test).items()))

    # resample
    # no MODEL approach for a moment as it requires more accurate management of under- and over- sampling size
    try: # workaround for any errors
        if level == 'CLASS_NR':
            X, y = SMOTE(sampling_strategy={ii:sampling_strategy_over[ii]}).fit_resample(X, y)
        else: # for other levels - just create twice more points
            X, y = SMOTE(sampling_strategy={ii: 2*len(df_train[df_train['NR']==ii])} ).fit_resample(X, y)
    except: pass
    try: # workaround for any errors when too few points
        X, y = NearMiss(sampling_strategy={0:under_size},version=1).fit_resample(X, y)
        print('     do undersampling others_class: ', under_size)
    except:
        pass
    brf_t = make_new_brf()
    brf_t = brf_t.fit(X,y)
    y_pred = brf_t.predict(X_test)
    y_prob = brf_t.predict_proba(X_test)

    # fill the dataset to compare max probabilities for the predicted classes
    dfp[ii]=y_prob[:,1]

    cr = classification_report_imbalanced(y_test, y_pred)
    # print(cr)
    print(cr.split('\n')[0])
    print(cr.split('\n')[-4])

############

# get the class name with the highest probability among all 1-vs-others predicted
dfp = dfp.reset_index().drop(columns='index')
for n in range(len(dfp)):
    x = dfp.iloc[dfp.index == n].drop(columns=['id', level, 'NR'])
    dfp.loc[dfp.index == n, 'predict'] = x.columns[x.iloc[0].argmax()]
dfp['predict'] = dfp['predict'].astype(int)
# set the "class of the highest probability among all 1-vs-others" as the "y_predict"
cr = classification_report_imbalanced(dfp['NR'], dfp['predict'], output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
print(cr_df)
