import os
import xarray as xr
from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import natsort

# for -f_createPAmask- exclusevely
from scipy.cluster.vq import vq
# import dask.array as da
# for model creation and train
from sklearn.ensemble import RandomForestClassifier as RFC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import shapely
from sklearn.model_selection import train_test_split
# import shap
# for model usage
from joblib.parallel import cpu_count, Parallel, delayed
from scipy import stats as st
from imblearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import BalancedRandomForestClassifier
# from sklearn.model_selection import cross_validate
# resample
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import ClusterCentroids, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.under_sampling import AllKNN, NeighbourhoodCleaningRule


def f_createRF(df_train, df_test='', n_trees=250, save_model_to='',
               random_state_int=None, max_samples=0.9, verbose=1, occ_ID='CLASS_NR', occ_IDstr='CLASS_NAME'):
    # create the main body of the RandomForest model, train it, evaluate and save
    # Parameters:
    #   df_train : -dataset- : -occ- + -PA- dataset, coming from -f_modelTrainingData- function
    #   df_test : -dataset- or "" : dataset comming from -f_modelTestData- function. Keep it empty if you don't want to test
    #   n_trees : -int- : number of frees in forest
    #   save_model_to : -str- : full path -NAME.joblib- where to save the model to reuse it; to reload call joblib.load("NAME.joblib")
    #   random_state_int : -int- or None : random integer to fix the internal model randomizer and provide the results reproducibility
    #   max_samples : -float- or None : number of samples to draw from X to train each base estimator. If float, then draw max_samples * X.shape[0] samples
    #   verbose : -int- : how much stuff to print if test dataset is provided
    #   occ_ID : -int- : name of a -df_train- column with classes as integer
    #   occ_IDstr : -str- : name of a -df_train- column with classes as string

    from sklearn.metrics import classification_report, confusion_matrix
    if verbose==1:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.inspection import permutation_importance
        import shap
    if save_model_to!='':
        import joblib

    print('Start model generation ({0} trees)...'.format(n_trees))

    # create model
    if random_state_int is None: random_state_int=int(np.random.random(1)*100)
    RFmod = RFC(n_estimators=n_trees, # The number of trees in the forest.
                # class_weight='balanced_subsample'
                criterion='gini', # 'gini' 'entropy' #  function to measure the quality of a split
                max_features=0.9, # The number of features to consider when looking for the best split
                bootstrap=True, # Whether bootstrap samples are used when building trees
                max_samples=max_samples,  # the number of samples to draw from X to train each base estimator. If float, then draw max_samples * X.shape[0] samples
                oob_score=True, # Whether to use out-of-bag samples to estimate the generalization score
                n_jobs=-1, # The number of jobs to run in parallel
                random_state=random_state_int, # Control the randomness of the bootstrapping and the sampling of the features when split
                verbose=0, # Controls the verbosity when fitting and predicting.
                )
    RFmod.fit(df_train.drop(columns=[occ_ID, occ_IDstr]).values, df_train[occ_ID].values)
    oob_score=np.round(RFmod.oob_score_, 3)
    print('Ouf-of-bag quality score =', oob_score)

    # model quality
    cr = []
    ma = []
    if len(df_test)>0:
        ma = np.round(RFmod.score(df_test.drop(columns=[occ_ID, occ_IDstr]).values ,df_test[occ_ID].values), 3)
        print('Mean accuracy for test dataset =',ma )
        y_pred_test = RFmod.predict(df_test.drop(columns=[occ_ID, occ_IDstr]).values)
        cr = classification_report(df_test[occ_ID], y_pred_test)
        print('F1-score for test dataset = ', cr[-15:-11] )

        if verbose==1:
            print(cr[:53])
            print(cr[-54:])
            matrix = confusion_matrix(df_test[occ_ID], y_pred_test)
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            # Build the plot
            plt.figure(figsize=(16, 7))
            sns.set(font_scale=1.4)
            sns.heatmap(matrix, annot=True, annot_kws={'size': 8},
                        cmap=plt.cm.Greens, linewidths=0.2)
            class_names = df_train[occ_IDstr].unique()
            tick_marks = np.arange(len(class_names))
            tick_marks2 = tick_marks + 0.5
            plt.xticks(tick_marks, class_names, rotation=25)
            plt.yticks(tick_marks2, class_names, rotation=0)
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
    # features importance (keep only -k- most important)
    k=10
    if verbose==1:
        std_MDI = np.std([tree.feature_importances_ for tree in RFmod.estimators_], axis=0)
        importances_MDI = pd.DataFrame(data={'Imp':RFmod.feature_importances_,'std':std_MDI}, index=df_test.drop(columns=[occ_ID, occ_IDstr]).columns)
        importances_MDI = importances_MDI.sort_values(by='Imp',ascending=False)[:k]
        fig, ax = plt.subplots()
        importances_MDI['Imp'].plot.bar(yerr=importances_MDI['std'], ax=ax, fontsize=10)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        if len(df_test)>0:
            print('     estimate permutation features importance...')
            pm = permutation_importance(RFmod, df_test.drop(columns=[occ_ID, occ_IDstr]).values ,df_test[occ_ID].values,
                                                    n_repeats=10, random_state=42, n_jobs=2)
            importances_PM = pd.DataFrame(data={'Imp':pm.importances_mean, 'std':pm.importances_std}, index=df_test.drop(columns=[occ_ID, occ_IDstr]).columns)
            importances_PM = importances_PM.sort_values(by='Imp', ascending=False)[:k]
            fig, ax = plt.subplots()
            importances_PM['Imp'].plot.bar(yerr=importances_PM['std'], ax=ax, fontsize=10)
            ax.set_title("Feature importances using permutation on full model")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            plt.show()

            print('     estimate SHapley Additive exPlanations for features importance...' )
            explainer = shap.TreeExplainer(RFmod)
            shap_values = explainer.shap_values(df_test.drop(columns=[occ_ID, occ_IDstr]).to_numpy())
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, features=df_test.drop(columns=[occ_ID, occ_IDstr]).to_numpy(),
                              feature_names=df_test.drop(columns=[occ_ID, occ_IDstr]).columns,
                              class_names = df_test[occ_IDstr].unique(), # RFmod.classes_,
                              plot_size = "auto", # "auto"  , float, (float, float)
                              max_display=k)
            ax.tick_params(labelsize=10)
            ax.legend(fontsize=10, ncol=2, bbox_to_anchor=(1.1, 1.05))
            fig.tight_layout()

    if save_model_to!='':
        joblib.dump(RFmod, save_model_to.split('.')[0] + '_RFmod.joblib')
        names = occ.append({'ID': 'NoData', 'IDcode': 0}, ignore_index=True).set_index('IDcode').loc[RFmod.classes_, 'ID']
        names.to_csv(save_model_to+'_classesList.txt',header=False)

    return RFmod, oob_score, ma, cr


############### ============================== ###################
# data reading, separation of test and training samples
def date_to_ALLIANCE_N(date):
    parts = date.split('-')
    alliance = f'{int(parts[0][-2:])}_{int(parts[1])}_{int(parts[2])}'
    return alliance

def ASSOCIATIO_classufy(assot):
    if assot[0] == '2':
        return '2'
    elif assot[0] == '3':
        return '3'
    else:
        return assot
    

df = gpd.read_file("E:/TundraVegetationProject/data/BB_model_points_shp/BB_model_data_13.shp")
df['Vegetation'] = df['Vegetation'].replace([-300, -200, -100], [16001, 0, 0])
df['ALLIANCE_N'] = df['ALLIANCE_N'].apply(date_to_ALLIANCE_N)
df['ORDER_NR'] = df.loc[:, 'ORDER_NR'].apply(ASSOCIATIO_classufy)

X = df.loc[:, [*df.columns[19:-1]]]
y = df.loc[:, 'ORDER_NR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['ORDER_NR'])
# df['ASSOCIAT_NR'] = df.loc[:, 'ASSOCIATIO'].apply(ASSOCIATIO_classufy)
# df['ASSOCIAT_NR'] = df['ASSOCIAT_NR'].apply(lambda x: '5_1_1' if x.startswith('5_1_1') else x)
# X = df.loc[:, [*df.columns[19:-2]]]
# y = df.loc[:, 'ASSOCIAT_NR']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['CLASS_NR'])

# fix data imbalanse
over_ss = {4:60, 11:50, 10:50, 8:45, 3:40, 7:40, 2:40}
smote = SMOTE(sampling_strategy=over_ss, random_state=42)
under_ss_dict={1:220, 5:195, 9:120, 6:110}
nm1 = NearMiss(sampling_strategy=under_ss_dict, version=1)

# model option
RF_imbalanced = BalancedRandomForestClassifier(
    sampling_strategy="all", 
    replacement=True,
    n_estimators=120, # The number of trees in the forest.
    class_weight='balanced_subsample',
    criterion='gini', # 'gini' 'entropy' #  function to measure the quality of a split
    max_features=0.9, # The number of features to consider when looking for the best split
    bootstrap=True, # Whether bootstrap samples are used when building trees
    # max_samples=max_samples,  # the number of samples to draw from X to train each base estimator. If float, then draw max_samples * X.shape[0] samples
    oob_score=True, # Whether to use out-of-bag samples to estimate the generalization score
    n_jobs=-1, # The number of jobs to run in parallel
    random_state=1, # Control the randomness of the bootstrapping and the sampling of the features when split
    verbose=0, # Controls the verbosity when fitting and predicting.
    )
    
# creating and applying pipeline with model, over- and undersampling
model = make_pipeline(nm1, smote, RF_imbalanced)
# model = make_pipeline(RF_imbalanced)
result_model_forest = model.fit(X_train, y_train)
y_pred = result_model_forest.predict(X_test)
cr = classification_report_imbalanced(y_test, y_pred, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
print(cr_df)


# test various number of trees
# sc = []
# ma = []
# for n in [10, 300]:
#     RFmod, oob_score, mean_accuracy, classif_report = f_createRF(df_train, df_test=df_test, n_trees=n,
#                                                                 random_state_int=1, max_samples=0.9, 
#                                                                 verbose=0, occ_ID='CLASS_NR', occ_IDstr='CLASS_NAME')
#     sc.append([n,oob_score])
#     ma.append([n, mean_accuracy])
# sc = np.array(sc)
# ma = np.array(ma)
# fig, ax = plt.subplots()
# ax.plot(sc[:,0],sc[:,1])
# ax.plot(ma[:,0],ma[:,1])
# ax.set_ylim(0.5,1)
# ax.set_legend()   