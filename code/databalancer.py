import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours, TomekLinks, RepeatedEditedNearestNeighbours, NearMiss, ClusterCentroids
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
from dataloader import stack_sequences, prep_tensor
import cupy as cp
from numba import cuda
from cuml.cluster import KMeans as cuKMeans
from cuml.neighbors import NearestNeighbors as cuNN
import gc

import os
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'


def df_event_focus(df, event_col, event_focus=1):
    df2 = df.copy()
    df2[event_col] = np.where(df2[event_col] == event_focus, event_focus, 0)
    df2[event_col] = df2[event_col].replace({event_focus:1})
    return df2

def rebalance_data(df, event_col, event_focus, cat_features, params, random_state, method='ENN', n_jobs=10):
    sme = SMOTEENN(sampling_strategy=params['sampling_strategy'], random_state=random_state, n_jobs=n_jobs)
    smt = SMOTETomek(sampling_strategy=params['sampling_strategy'], random_state=random_state, n_jobs=n_jobs)
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    
    y = df[event_col].astype(int).values
    X = df.drop(columns=event_col)
    assert len(X) == len(y)
    
    if method == 'ENN':
        X_res, y_res = sme.fit_resample(X, y)
    elif method == 'Tomek':
        X_res, y_res = smt.fit_resample(X, y)
    
    assert len(X_res) == len(y_res)
    X_res.loc[:, event_col] = y_res
    
    # Check for NaN values and impute if necessary
    if X_res.isnull().values.any():
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=random_state,
                                   initial_strategy='mean', n_nearest_features=None, min_value=1e-6,
                                   imputation_order='ascending')
        X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
    
    assert not X_res.isnull().values.any(), "NaN values are still present after imputation"
    X_res[cat_features] = round(X_res[cat_features])
    print(f'Dataframe rebalanced with SMOTE and {method}.')
    del sme, smt, df, y, X, imputer
    gc.collect()
    return X_res

def define_medoid(df, feature_col, event_col, event_focus=1, n_neighbors=50):
    # Ensure event_col is a string
    if isinstance(event_col, list) and len(event_col) == 1:
        event_col = event_col[0]
    
    import numba.cuda
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
    numba.cuda.cudadrv.devices.get_context() 
    cp.get_default_memory_pool().free_all_blocks()
    
    # isolate the minority rows
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    df_minor = df[df[event_col] == event_focus].copy()
    df_major = df[df[event_col] != event_focus].copy()
    
    # use number of rws in minority group as the number of clusters
    n_clusters = min(len(df_major), len(df_minor))
    print(f"Setting n_neighbors to {n_clusters} (size of minority class)")
    # Initialize and fit NearestNeighbors
    df_major_fea = df_major[feature_col].copy()
    
    # nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)  # +1 because the first neighbor is the point itself
    nn = cuNN(n_neighbors=n_neighbors, algorithm='auto')  # +1 because the first neighbor is the point itself
    nn.fit(df_major_fea)
    # Compute the distances to the n_neighbors-th nearest neighbor for each point
    # Suppose medoid should be the row that have lowest distances with all other point
    distances, _ = nn.kneighbors(df_major_fea)
    total_distance = distances.sum(axis=1)
    cluster_center = np.argsort(total_distance)[:n_clusters]
    df_medoid = df_major.iloc[cluster_center].copy()
    
    assert sorted(df_medoid.columns) == sorted(df_minor.columns)
    
    # Combine the medoid of major group and the minority group
    df_final = pd.concat([df_medoid, df_minor])
    df_final = df_final.sample(frac=1.0)
    
    # Reamining dataframe (both major and minor group) after slicing out the medoid rows
    df_remain = df.drop(df_major.index[cluster_center])
    gc.collect()
    return df_final, df_remain

def medoid_cluster(df, feature_col, duration_col, event_col, event_focus, params, cluster_col, seq_length=1, n_neighbors=50):
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    remaining_data = df.copy()
    goal = round(1 / params['sampling_strategy'])
    repeat_count = 0
    all_clusters = []
    
    while len(remaining_data) > 0 and repeat_count < goal:
        print(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        X_cluster, remaining_data = define_medoid(df=remaining_data, feature_col=feature_col, event_col=event_col, event_focus=event_focus, n_neighbors=n_neighbors)
        # Append the current X_cluster to the list
        all_clusters.append(X_cluster)
        # Increment the repeat count
        repeat_count += 1
    # After the loop, concatenate all X_cluster DataFrames
    concatenated_clusters = pd.concat(all_clusters, ignore_index=True)
    # Remove duplicated rows, keeping only the first occurrence
    final_clusters = concatenated_clusters.drop_duplicates(keep='first')
    
    if seq_length >=2:
        X, y = stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, seq_length)
    else:
        X,y = prep_tensor(df=df, feature_col=feature_col, duration_col=duration_col, event_col=event_col)
    # grouped = final_clusters.groupby(cluster_col)
    # sequences = []
    # durations = []
    # events = []

    # for name, group in grouped:
    #     group = group.sort_values(by=duration_col).reset_index(drop=True)
    #     for i in range(len(group)):
    #         # Get the 'key' from the current group row
    #         key = group.iloc[i][cluster_col]
    #         # Filter the original dataframe to get all rows with the same 'key'
    #         key_group = df[df[cluster_col] == key].sort_values(by=duration_col).reset_index(drop=True)
    #         # Get the current row's duration
    #         current_duration = group.iloc[i][duration_col]
    #         # Find the position of the current row in the original data's sequence
    #         pos = key_group[key_group[duration_col] == current_duration].index[0]
    #         # Select the previous 5 rows (or pad if fewer rows are available)
    #         seq = key_group.iloc[max(0, pos-seq_length+1):pos+1][feature_col].values
    #         seq = np.pad(seq, ((seq_length - len(seq), 0), (0, 0)), 'constant')
             
    #         sequences.append(seq)
    #         durations.append(group.iloc[i][duration_col].values)
    #         events.append(group.iloc[i][event_col].values)

    # X = np.array(sequences)
    # y_duration = np.array(durations).squeeze()
    # y_event = np.array(events).squeeze()
    # y = (y_duration, y_event)
    
    return X, y

def underbalance_data(df, event_col, cluster_col, event_focus=1, sampling_strategy=0.2, n_jobs=10, method='NearMiss', random_state=12345):
    assert len(event_col) == 1
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    df['_original_index'] = df.index
    X = df.drop(columns=[event_col[0], cluster_col[0]])
    y = df[event_col]
    
    if method == 'NearMiss':
        sampler = NearMiss(sampling_strategy=sampling_strategy, version=3, n_jobs=n_jobs)
    elif method == 'cluster':
        min_cluster = df[event_col].value_counts().min()
        cuml_kmeans = cuKMeans(n_clusters=min_cluster, random_state=random_state)
        sampler = ClusterCentroids(sampling_strategy=sampling_strategy, estimator=cuml_kmeans, random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    # Retrieve the original indices of the resampled data
    original_indices = X_res['_original_index'].values
    df_balanced = df.loc[original_indices]
   
    return df_balanced

### DeepHit
def dh_define_medoid(df, feature_col, event_col, n_neighbors=50):
    # Ensure event_col is a string
    if isinstance(event_col, list) and len(event_col) == 1:
        event_col = event_col[0]
    
    import numba.cuda
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
    numba.cuda.cudadrv.devices.get_context() 
    cp.get_default_memory_pool().free_all_blocks()
    
    # isolate the minority rows
    # df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    df_minor = df[df[event_col] != 0].copy()
    df_major = df[df[event_col] == 0].copy()
    
    # use number of rws in minority group as the number of clusters
    n_clusters = min(len(df_major), len(df_minor))
    print(f"Setting n_neighbors to {n_clusters} (size of minority class)")
    # Initialize and fit NearestNeighbors
    df_major_fea = df_major[feature_col].copy()
    
    # nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto', n_jobs=-1)  # +1 because the first neighbor is the point itself
    nn = cuNN(n_neighbors=n_neighbors, algorithm='auto')  # +1 because the first neighbor is the point itself
    nn.fit(df_major_fea)
    # Compute the distances to the n_neighbors-th nearest neighbor for each point
    # Suppose medoid should be the row that have lowest distances with all other point
    distances, _ = nn.kneighbors(df_major_fea)
    total_distance = distances.sum(axis=1)
    cluster_center = np.argsort(total_distance)[:n_clusters]
    df_medoid = df_major.iloc[cluster_center].copy()
    
    assert sorted(df_medoid.columns) == sorted(df_minor.columns)
    
    # Combine the medoid of major group and the minority group
    df_final = pd.concat([df_medoid, df_minor])
    df_final = df_final.sample(frac=1.0)
    
    # Reamining dataframe (both major and minor group) after slicing out the medoid rows
    df_remain = df.drop(df_major.index[cluster_center])
    gc.collect()
    return df_final, df_remain

def dh_underbalance_data(df, event_col, cluster_col, sampling_strategy=0.2, n_jobs=10, method='NearMiss', random_state=12345):
    assert len(event_col) == 1
    df['_original_index'] = df.index
    X = df.drop(columns=[event_col[0], cluster_col[0]])
    y = df[event_col]
    
    if method == 'NearMiss':
        sampler = NearMiss(sampling_strategy={0: int((1 / sampling_strategy) * ((df[event_col] == 1).sum().iloc[0] + (df[event_col] == 2).sum().iloc[0]))})
    elif method == 'cluster':
        min_cluster = df[event_col].value_counts().min()
        cuml_kmeans = cuKMeans(n_clusters=min_cluster, random_state=random_state)
        sampler = ClusterCentroids(sampling_strategy=sampling_strategy, estimator=cuml_kmeans, random_state=random_state)
    X_res, y_res = sampler.fit_resample(X, y)
    # Retrieve the original indices of the resampled data
    original_indices = X_res['_original_index'].values
    df_balanced = df.loc[original_indices]
    return df_balanced

def dh_rebalance_data(df, event_col, cluster_col, cat_features, sampling_strategy=0.2, random_state=12345, method='NearMiss', version=2):
    if method == 'NearMiss':
        sampler = NearMiss(sampling_strategy={0: int((1 / sampling_strategy) * ((df[event_col] == 1).sum().iloc[0] + (df[event_col] == 2).sum().iloc[0]))}, version=version)
    
    X = df.drop(columns=[event_col[0], cluster_col[0]])
    y = df[event_col]
    assert len(X) == len(y)
    assert len(event_col) == 1
        
    X_res, y_res = sampler.fit_resample(X, y)
    
    assert len(X_res) == len(y_res)
    X_res.loc[:, event_col] = y_res
    
    # Check for NaN values and impute if necessary
    if X_res.isnull().values.any():
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=random_state,
                                   initial_strategy='mean', n_nearest_features=None, min_value=1e-6,
                                   imputation_order='ascending')
        X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
    else:
        imputer = None
    
    assert not X_res.isnull().values.any(), "NaN values are still present after imputation"
    X_res[cat_features] = round(X_res[cat_features])
    print(f'Dataframe rebalanced.')
    del sampler, df, y, X, imputer
    gc.collect()
    return X_res
