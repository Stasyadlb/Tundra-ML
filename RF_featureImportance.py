import os
# import osgeo
import xarray as xr
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import natsort
# for model creation and train
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import shap
from collections import Counter
# imbalanced learn
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced

def get_MeanDecreasInImpurity(RFmod, feature_names, class_name):
    # Which features was really used for Decision Trees construction
    importances_MDI = pd.DataFrame(data={'MDI_{0}'.format(class_name): RFmod.feature_importances_},
                                   index=feature_names)
    return importances_MDI

def get_MeanAccuracyDecrease(RFmod, x_test, y_test, feature_names, class_name):
    # Which features are the most usefull for the classification - approach of Mean Accuracy decrease
    # plot the first -k- features
    from sklearn.inspection import permutation_importance
    pm = permutation_importance(RFmod, x_test, y_test,
                                n_repeats=10, random_state=42, n_jobs=2)
    importances_PM = pd.DataFrame(data={'MAD_{0}'.format(class_name): pm.importances_mean},
                                  index=feature_names)
    return importances_PM

def get_SHAPEY(RFmod, x_test, feature_names,class_name):
    # Which features are the most useful for the classification - approach of SHapley Additive exPlanations
    explainer = shap.TreeExplainer(RFmod)
    shap_values = explainer.shap_values(x_test)

    rf_resultX = pd.DataFrame(shap_values[0], columns=feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                   columns=['feature_names','SHAP_{0}'.format(class_name)])
    shap_importance.set_index('feature_names', inplace=True)

    return shap_importance

def add_featureImpData(df_imp, brf, X_test, y_test, feature_names, ii):
    tech = get_MeanDecreasInImpurity(RFmod=brf, feature_names=feature_names, class_name='cl'+str(ii))
    df_imp = df_imp.merge(tech, left_on=df_imp.index, right_on=tech.index).set_index('key_0')
    tech = get_MeanAccuracyDecrease(RFmod=brf, x_test=X_test, y_test=y_test,
                             feature_names=feature_names, class_name='cl' + str(ii))
    df_imp = df_imp.merge(tech, left_on=df_imp.index, right_on=tech.index).set_index('key_0')
    tech = get_SHAPEY(RFmod=brf, x_test=X_test, feature_names=feature_names, class_name='cl' + str(ii))
    df_imp = df_imp.merge(tech, left_on=df_imp.index, right_on=tech.index).set_index('key_0')

    return df_imp

def make_binary_dataset(df_train, df_test, i):
    # get only target class
    df_train_loc = df_train.copy(True)
    df_train_loc.loc[df_train_loc['NR'] != i, 'NR'] = 0
    df_test_loc = df_test.copy(True)
    df_test_loc.loc[df_test_loc['NR'] != i, 'NR'] = 0
    X, y = df_train_loc.drop(columns=['NR']).values, df_train_loc['NR'].values
    X_test, y_test = df_test_loc.drop(columns=['NR']).values, df_test_loc['NR'].values

    under_size = int(len(df_train_loc[df_train_loc['NR'] != i]) * 0.5)

    return X, y, X_test, y_test, under_size

def run_samplingAndModel(X, y, X_test, sampling_strategy_over, under_size ):
    # resample
    try:
        if level == 'CLASS_NR':
            X, y = SMOTE(sampling_strategy={ii:sampling_strategy_over[ii]}).fit_resample(X, y)
            print('     do oversampling basic_class: ',sampling_strategy_over[ii])
        else:
            over_size = 2*len(df_train[df_train['NR']==ii])
            X, y = SMOTE(sampling_strategy={ii: over_size} ).fit_resample(X, y)
            print('     do oversampling: ', over_size)
    except: pass
    try:
        X, y = NearMiss(sampling_strategy={0:under_size},version=1).fit_resample(X, y)
        print('     do undersampling others_class: ', under_size)
    except:
        pass

    # model
    brf_t = make_brf()
    brf_t = brf_t.fit(X, y)
    y_pred = brf_t.predict(X_test)

    cr = classification_report_imbalanced(y_test, y_pred, output_dict=True)
    cr = pd.DataFrame(cr).transpose()

    return brf_t, cr, y_pred


def make_brf(random_state=0):
    if random_state is None:
        random_state = int(np.random.random(1) * 100)
    brf = BalancedRandomForestClassifier(n_estimators=100, random_state=random_state,
                                         class_weight='balanced_subsample',
                                         max_features=0.95,  # None, #'sqrt', # None
                                         sampling_strategy="all", replacement=True,
                                         oob_score=True, bootstrap=True)
    return brf



############### ============================== ###################
base = 'C:/Users/AnnD/Desktop/ВШЭ_ЛЛЭ/TundraVegetationProject/ВВ_model_13/'
gdf = gpd.read_file(base + 'BB_model_data_13.shp')

level = 'ASSOCIATIO'  # 'CLASS_NR' 'ORDER_NR' 'ALLIANCE_N' 'ASSOCIATIO'
if level == 'CLASS_NR':
    test_size = 0.2
else:
    test_size = 0.3
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
# some functions will not wirk with string data "*_*_*", so convert it to numeric "***"
dff['NR'] = dff[level].astype(str).str.replace('_','').astype(int)

# '5_1_1_12' has only 1 point, that will raise an error at -train_test_split- step
if level == 'ASSOCIATIO':
    # '5_1_1_12' has only 1 point, that will raise an error at -train_test_split- step
    dff = dff.drop(dff[dff[level] == '5_1_1_12'].index)
    # get only associations with many points
    a = dff['NR'].value_counts()
    a = a[a > 23].index.to_list()
    dff = dff[dff['NR'].isin(a)]

print('Train', sorted(Counter(y).items()))
print('Test', sorted(Counter(y_test).items()))

##########################################################################################
# for CLASS level, for 0.2 split
class_sampling_strategy_over= {2:40, 3:50, 4:50, 7:40, 8:50, 10:50, 11:50}
class_sampling_strategy_under={1:250, 5:180}

############################## FEATURE IMPORTANCE ##########################
# dff_fi - dataframe with the selected features. basic case - all data
dff_fi = dff.copy(True)

df_train_fi, df_test_fi = train_test_split(dff_fi,
                                    test_size=test_size, random_state=1, shuffle=True,
                                    stratify=dff[level].values)

# container for feature importance values
df_imp = pd.DataFrame(index=df_test_fi.drop(columns = [level, 'NR']).columns)

#######
# run 1:all cases
for i in dff['NR'].sort_values().unique():
    ii = i

    print(' ')
    print(' ')
    print('### >>> {} #'.format(level), i)

    # get only target class
    X, y, X_test, y_test, under_size = make_binary_dataset(df_train_fi.drop(columns=[level]),
                                                           df_test_fi.drop(columns=[level]), ii)

    # train model
    brf_t, cr, y_pred = run_samplingAndModel(X, y, X_test, class_sampling_strategy_over, under_size)

    df_imp = add_featureImpData(df_imp, brf_t, X_test, y_test,
                                feature_names = df_test_fi.drop(columns = [level, 'NR']).columns,
                                ii=ii )

df_imp.to_excel(base+'FeatureImportance_{}.xlsx'.format(level) )