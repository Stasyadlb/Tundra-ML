import os
import pandas as pd
import xarray as xr
import rioxarray
from glob import glob
import natsort
import geopandas as gpd
import numpy as np
import joblib
import dask
import datetime
#dask.config.set(**{'array.slicing.split_large_chunks': True})


def f_runFR(env, RFmod, results_path='', save_prefix='', save_classes=False, replace=False, max_batch_size=20000, fill_value=-9999):
    # Apply the model over the entire dataset
    # Parameters:
    #   env : -xarray Dataset- : environmental data load with -f_loadEnv- function
    #   RFmod : -RandomForestClassifier- : trained classifier from -f_createRF- function or loaded in memory
    #   results_path : -str- : folder path where to save prediction rastres
    #   save_prefix : -str- : the base of files names
    #   save_classes : -True- or -False- : save the classes raster as well or save only probability raster (by default)
    #   replace : -True- or -False- : overwrite the output tifs if existing or not
    #   max_batch_size : -int - : control the dataset split for small jobs to deal with its huge size
    #   fill_value : -int- or -float- : fill value for nan and inf that surely is not close to your true data

    from joblib.parallel import cpu_count, Parallel, delayed

    def _predict(estimator, X, method, start, stop):
        return getattr(estimator, method)(X[start:stop])

    def parallel_predict(estimator, X, method='predict_proba', max_batch_size=20000, batches_per_job=4):
        n_jobs = min(max(cpu_count() - 2, 1), 60 ) # 60 # manage lib issue of max n_jobs
        n_samples = len(X)
        # balance between n_batches sinchronized with n_jobs and max_batch_size respect
        n_batches_theor = batches_per_job * n_jobs
        n_batches_prac = int(n_samples/max_batch_size) + 1
        n_batches = max (n_batches_theor , n_batches_theor * int(n_batches_prac/n_batches_theor) )
        batch_size = int(np.ceil(n_samples / n_batches))

        parallel = Parallel(n_jobs=n_jobs) ### src : parallel = Parallel(n_jobs=n_jobs) #prefer="threads"
        print('     run prediction in parallel: ')
        print('          {0} CPU found, {1} jobs will be allocated'.format(cpu_count(), n_jobs))
        print('          {0} batches to split the data, {1} elements per batch'.format(n_batches, batch_size))
        results = parallel(delayed(_predict)(estimator, X, method, i, i + batch_size)
                           for i in range(0, n_samples, batch_size))
        # if sp.issparse(results[0]):
        #     return sp.vstack(results)

        return np.concatenate(results)

    print('here1')
    # create containers for future resulting rasters
    src_shape = env.band_data.values.shape  # (19, 5400, 20760)
    dss_prb = env.drop_isel(band=np.arange(1, src_shape[0]))
    dss_cls = dss_prb.copy(deep=True)
    valid_mask = env['band_data'][0].notnull()

    print('here2')
    # flatten the data
    flat_values = dask.array.from_array(env.band_data.values.astype('float16'))  # call the force of dask!
    del env # memory managing
    print('here3')
    flat_values = flat_values.transpose()  # (19, 5400, 20760) -> (20760, 5400, 19)
    flat_values = flat_values.reshape(flat_values.shape[0] * flat_values.shape[1], flat_values.shape[2]) # (20760, 5400, 19) -> (112104000, 19)
    print('here4')
    flat_values = flat_values.compute().astype('float16')
    print('here5')
    flat_values[np.isnan(flat_values)] = fill_value
    flat_values[np.isinf(flat_values)] = fill_value
    print('here6')

    # predict
    k = 1 # workaround for lack of memory
    while k<3:
        try:
            # prb = parallel_predict(RFmod, flat_values, method='predict_proba', batches_per_job=batches_per_job*k)
            #prb = parallel_predict(RFmod, flat_values, method='predict_proba', max_batch_size=max_batch_size / (k*2) )
            if k==1:
                prb = parallel_predict(RFmod, flat_values, method='predict_proba', max_batch_size=max_batch_size)
            if k==2:
                print('  *This is notification from f_runFR. Activate a workaround for the memory lack*')
                split=int(flat_values.shape[0] /2)
                prb_a = parallel_predict(RFmod, flat_values[ :split,:], method='predict_proba', max_batch_size=max_batch_size/2)
                prb_b = parallel_predict(RFmod, flat_values[split:,:], method='predict_proba', max_batch_size=max_batch_size/2)
                prb=np.concatenate( [prb_a, prb_b] )
                del prb_a, prb_b

            # restore data shape and geo space
            prb = prb.reshape(src_shape[2], src_shape[1], prb.shape[1]).T
            max_prb = np.amax(prb, axis=0)
            dss_prb = dss_prb.assign(band_data=(['band', 'y', 'x'], np.full([dss_prb.band.size, dss_prb.y.size, dss_prb.x.size], np.nan)))
            dss_prb.band_data[0, :, :] = np.round(max_prb,3)
            cls = np.argmax(prb, axis=0)
            dss_cls.band_data[0, :, :] = cls
            # mask areas with no-data in env
            dss_prb = dss_prb.where(valid_mask)
            dss_cls = dss_cls.where(valid_mask)

            if (results_path!='') and (save_prefix!=''):
                try:
                    # save to
                    loc_base = results_path + save_prefix
                    past=''
                    if os.path.exists(loc_base + '_prob.tif') and not replace:
                        past = '_1'
                    dss_prb.band_data[0].rio.set_nodata(-9999).rio.to_raster(loc_base + '_prob{0}.tif'.format(past) , compress='LZW') #,dtype='float32' )
                    #pd.Series(RFmod.classes_).to_csv(loc_base+'_classesList{0}.txt'.format(past),header=False)

                    try:print('     NO_CLASS index = ', RFmod.classes_.tolist().index('ZZZ'))
                    except: print('     NO_CLASS was not used this time')

                    if save_classes:
                        dss_cls.band_data[0].rio.set_nodata(-9999).rio.to_raster(loc_base + '_class{0}.tif'.format(past),dtype='int16', compress='LZW')

                    print('     save results to ', loc_base + '_***{0}.tif'.format(past))

                except:
                    print('     processing is finished BUT FAIL TO SAVE DATA. Do it manually')

                return dss_prb, dss_cls

            k = 3

        except:
            k=k+1
            if k<3:
                print('     prediction failed. Maybe this is due to the lack of memory. Retrying with max_batch_size/2...')
                print('     If that will not solve the memory problem, try to decrease significantly -max_batch_size- .')
            else:
                raise RuntimeError("Prediction step failed. Maybe this is due to the lack of computation resources. \n But still check the other warnings, any other reasons could crash it.")



def class_modeling(base_path, results_path, model_path, loaded_model, class_flist, adem_tile):
    flist_model = []
    for factor in class_flist.split(',\n'):
        flist_model.append(base_path + factor.format(adem_tile))
    #ds_tech = xr.open_dataset(flist_model[0])
    ds_tech = xr.open_dataset(base_path + 'data/ArcticDEM/DEM_x100_Splited/{}.tif'.format(adem_tile))
    arrays=[]
    names=[]
    for n,f in enumerate(flist_model):
        dss = xr.open_dataset(f)
        dss = dss.rio.reproject_match(ds_tech)
        arrays.append(dss)
        base_name = os.path.basename(f).split('.tif')[0]
        if dss.band.size !=1:
            for k in range(dss.band.size):
                names.append(base_name + '_l' + str(k) )
        else:
            names.append(base_name)

    ds = xr.concat(arrays, dim='band')
    del arrays
    # -band- values should be unique!
    if len(np.unique(names)) != len(names):
        names = [i for i in range(ds.band.size)]
    ds = ds.assign_coords(band=names)
    ds = ds.where(~np.isnan(ds) & ~np.isinf(ds), other=np.nan )
    
    dss, dss1 = f_runFR(env = ds,
            RFmod = loaded_model,
            results_path=results_path, save_prefix=adem_tile + '_' + os.path.basename(model_path).split('.')[0],
            max_batch_size=10000,
            save_classes=True)
    return dss, dss1

base_path = 'G:/TundraVegetationProject/'
# path to the DEM tiles
adem_tiles = gpd.read_file(base_path + 'data/ArcticDEM/ArcticDEM_tileGrid.shp')['tile']
# path to classes information (class name, short path to model, short path to data)
classes_info = pd.read_excel(base_path + 'Model/classes_info.xlsx')
# class modeling
for index, class_info in classes_info.iterrows():
    print(f'Modeling {class_info["Name"]} started')
    print(datetime.datetime.now().time())
    model_path = base_path + class_info['Model']
    loaded_model = joblib.load(model_path)
    class_flist = class_info['Flist']
    results_path = base_path + f'data_modeling/{class_info["Name"]}/'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    for adem_tile in adem_tiles:
        print(f'Tile {adem_tile} is started')
        class_modeling(base_path, results_path, model_path, loaded_model, class_flist, adem_tile)
        print(f'   Tile {adem_tile} is done')
        print(datetime.datetime.now().time())

        

##############################################################################
# all classes information could be insert into code. Example:
### CLASS 6
# #### PREDICT
# for adem_tile in adem_tiles:
#     flist_model = [base_path + '/ArcticDEM/Slope_ln_x100_Splited/{}.tif'.format(adem_tile), 
#             base_path + 'ArcticDEM/TI_x100_Splited/{}.tif'.format(adem_tile),
#             base_path + 'ArcticDEM/TRuI_x100_Splited/{}.tif'.format(adem_tile),
#             base_path + 'ArcticDEM/VRM_R3_Splited/{}.tif'.format(adem_tile),
#             base_path + 'ArcticDEM/32m_MRRTF_Splited/{}.tif'.format(adem_tile),
#             base_path2 + 'CH_pet_penman_range_cropADEM-{}.tif'.format(adem_tile),
#             base_path2 + 'CH_hurs_min_cropADEM-{}.tif'.format(adem_tile),
#             base_path2 + 'CH_hurs_range_cropADEM-{}.tif'.format(adem_tile),
#             base_path2 + 'CH_pet_penman_max_cropADEM-{}.tif'.format(adem_tile),
#             base_path2 + 'CHELSA_TraCE21k_gld_-73_V1_Clip_cropADEM-{}.tif'.format(adem_tile),
#             base_path + 'GLAD/time1/10yrMed_splitLikeArcticDEM/{}_time1.tif'.format(adem_tile),
#             base_path + 'GLAD/time2/10yrMed_splitLikeArcticDEM/{}_time2.tif'.format(adem_tile),
#             base_path + 'GLAD/time3/10yrMed_splitLikeArcticDEM/{}_time3.tif'.format(adem_tile),
#             base_path + 'Hydro/HydroYandex_all_EucDist_log2x100_int_Split/{}.tif'.format(adem_tile),
#             base_path2 + 'M16_ET_MONTH_m7_cropADEM-{}.tif'.format(adem_tile),
#             base_path2 + 'M17_GPP_MONTH_m7_cropADEM-{}.tif'.format(adem_tile),
#             ]
