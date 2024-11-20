import os
import gc
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import NearMiss, ClusterCentroids
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from cuml.cluster import KMeans as cuKMeans
from cuml.neighbors import NearestNeighbors as cuNN
import cupy as cp
import numba.cuda
from dataloader2 import stack_sequences, prep_tensor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_cuda_environment():
    """
    Sets up CUDA environment variables and ensures that GPU memory is cleared.
    """
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64'
    numba.cuda.cudadrv.devices.get_context()
    cp.get_default_memory_pool().free_all_blocks()
    logging.info("CUDA environment set up and GPU memory cleared.")

def df_event_focus(df, event_col, event_focus=1):
    """
    Creates a copy of the DataFrame where the event column is set to 1 for rows matching the event_focus value, else 0.

    Args:
        df (pd.DataFrame): Input DataFrame.
        event_col (str): Column representing the event.
        event_focus (int, optional): Event value to focus on. Defaults to 1.

    Returns:
        pd.DataFrame: Modified DataFrame with updated event column.
    """
    df2 = df.copy()
    df2[event_col] = np.where(df2[event_col] == event_focus, 1, 0)
    logging.info(f"Event column '{event_col}' updated with focus on event value {event_focus}.")
    return df2

def rebalance_data(df, event_col, event_focus, cat_features, params, random_state, method='ENN', n_jobs=10):
    """
    Rebalances the dataset using specified resampling techniques (SMOTEENN or SMOTETomek).

    Args:
        df (pd.DataFrame): Input DataFrame.
        event_col (str): Event column name.
        event_focus (int): Event value to focus on.
        cat_features (list): List of categorical features.
        params (dict): Parameters for sampling strategy.
        random_state (int): Random state for reproducibility.
        method (str, optional): Resampling method ('ENN' or 'Tomek'). Defaults to 'ENN'.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 10.

    Returns:
        pd.DataFrame: Rebalanced DataFrame.
    """
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    y = df[event_col].astype(int).values
    X = df.drop(columns=event_col)
    
    if method == 'ENN':
        sampler = SMOTEENN(sampling_strategy=params['sampling_strategy'], random_state=random_state, n_jobs=n_jobs)
    elif method == 'Tomek':
        sampler = SMOTETomek(sampling_strategy=params['sampling_strategy'], random_state=random_state, n_jobs=n_jobs)
    else:
        raise ValueError("Unsupported method. Use 'ENN' or 'Tomek'.")
    
    X_res, y_res = sampler.fit_resample(X, y)
    X_res[event_col] = y_res
    
    # Check for NaN values and impute if necessary
    if X_res.isnull().values.any():
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=random_state, initial_strategy='mean')
        X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
        logging.info("Missing values imputed using IterativeImputer.")
    
    X_res[cat_features] = X_res[cat_features].round()
    logging.info(f"Dataframe rebalanced with SMOTE and {method}.")
    
    return X_res

def define_medoid_general(df, feature_col, event_col, event_focus=1, n_neighbors=50, model_type='deepsurv'):
    """
    Defines the medoid by calculating the total distance from each point to its neighbors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_col (list): Feature columns for clustering.
        event_col (str): Event column name.
        event_focus (int, optional): Event value to focus on. Defaults to 1.
        n_neighbors (int, optional): Number of neighbors for distance calculation. Defaults to 50.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.

    Returns:
        tuple: DataFrame containing medoids and DataFrame with remaining data.
    """
    setup_cuda_environment()

    if model_type == 'deephit':
        df_minor = df[df[event_col] != 0].copy()
        df_major = df[df[event_col] == 0].copy()
    else:
        df2 = df_event_focus(df, event_col=event_col, event_focus=event_focus)
        df_minor = df2[df2[event_col] == 1].copy()
        df_major = df2[df2[event_col] != 1].copy()

    n_clusters = min(len(df_major), len(df_minor))
    df_major_fea = df_major[feature_col].copy()

    nn = cuNN(n_neighbors=n_neighbors, algorithm='auto')
    nn.fit(df_major_fea)
    distances, _ = nn.kneighbors(df_major_fea)
    total_distance = distances.sum(axis=1)
    cluster_center = np.argsort(total_distance)[:n_clusters]
    df_medoid = df_major.iloc[cluster_center].copy()

    df_final = pd.concat([df_medoid, df_minor]).sample(frac=1.0)
    df_remain = df.drop(df_major.index[cluster_center])

    logging.info(f"Defined medoid for {model_type} model with {n_clusters} clusters.")
    return df_final, df_remain

def underbalance_data_general(df, event_col, cluster_col, params, n_jobs=10, version=3, random_state=12345):
    """
    Undersamples the dataset to balance the event classes.

    Args:
        df (pd.DataFrame): Input DataFrame.
        event_col (str): Event column name.
        cluster_col (str): Cluster column name.
        sampling_strategy (float, optional): Sampling strategy ratio. Defaults to 0.2.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 10.
        method (str, optional): Sampling method ('NearMiss' or 'cluster'). Defaults to 'NearMiss'.
        random_state (int, optional): Random state for reproducibility. Defaults to 12345.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.

    Returns:
        pd.DataFrame: Undersampled DataFrame.
    """
    model_type = params['model']
    method = params['balance_method']
    event_focus = params['endpoint']
    sampling_strategy = params['sampling_strategy']
    
    if model_type == 'deephit':
        df['_original_index'] = df.index
    else:
        df['_original_index'] = df.index
        df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
        

    X = df.drop(columns=[event_col, cluster_col])
    y = df[event_col]

    if method == 'NearMiss':
        if y.nunique() > 2:  # Multi-class case for DeepHit
            # Calculate the sampling strategy dict for multi-class undersampling
            non_major_class_count = y[y != 0].count()  # Sum of all non-major class rows
            majority_class_target = int((1 / sampling_strategy) * non_major_class_count)  # Target rows for the majority class
            sampling_strategy_dict = {label: count for label, count in y.value_counts().items() if label != 0}
            sampling_strategy_dict[0] = majority_class_target  # Set the majority class target count
            sampler = NearMiss(sampling_strategy=sampling_strategy_dict, version=version, n_jobs=n_jobs)
        else:  # Binary case for DeepSurv
            sampler = NearMiss(sampling_strategy=sampling_strategy, version=version, n_jobs=n_jobs)
    # elif method == 'clustering':
    #     min_cluster = df[event_col].value_counts().min()
    #     cuml_kmeans = cuKMeans(n_clusters=min_cluster, random_state=random_state)
    #     sampler = ClusterCentroids(sampling_strategy=sampling_strategy, estimator=cuml_kmeans, random_state=random_state)
    else:
        raise ValueError("Unsupported method. Use 'NearMiss'.")

    X_res, y_res = sampler.fit_resample(X, y)
    original_indices = X_res['_original_index'].values
    df_balanced = df.loc[original_indices]

    logging.info(f"Dataset for {model_type} model undersampled using method '{method}' with sampling strategy {sampling_strategy}.")
    return df_balanced

def medoid_cluster(df, feature_col, duration_col, event_col, event_focus, params, cluster_col, seq_length=1, n_neighbors=50, model_type='deepsurv'):
    """
    Clusters data using medoid definition and generates training sequences.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_col (list): Feature columns for clustering.
        duration_col (str): Duration column name.
        event_col (str): Event column name.
        event_focus (int): Event value to focus on.
        params (dict): Parameters for sampling strategy.
        cluster_col (str): Column representing clusters.
        seq_length (int, optional): Length of sequences. Defaults to 1.
        n_neighbors (int, optional): Number of neighbors for distance calculation. Defaults to 50.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.

    Returns:
        tuple: Feature tensor X and target tensor y.
    """
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    remaining_data = df.copy()
    goal = round(1 / params['sampling_strategy'])
    all_clusters = []
    
    for repeat_count in range(goal):
        if len(remaining_data) == 0:
            break
        logging.info(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        X_cluster, remaining_data = define_medoid_general(df=remaining_data, feature_col=feature_col, event_col=event_col, event_focus=event_focus, n_neighbors=n_neighbors, model_type=model_type)
        all_clusters.append(X_cluster)
    
    final_clusters = pd.concat(all_clusters, ignore_index=True).drop_duplicates(keep='first')
    
    if seq_length >= 2:
        X, y = stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, seq_length)
    else:
        X, y = prep_tensor(df=df, feature_col=feature_col, duration_col=duration_col, event_col=event_col)
    
    logging.info("Medoid clustering completed and training sequences generated.")
    return X, y

def dh_rebalance_data(df, event_col, cluster_col, cat_features, sampling_strategy=0.2, random_state=12345, method='NearMiss', version=2):
    """
    Rebalances the dataset for DeepHit model using NearMiss or cluster-based undersampling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        event_col (str): Event column name.
        cluster_col (str): Cluster column name.
        cat_features (list): List of categorical features.
        sampling_strategy (float, optional): Sampling strategy ratio. Defaults to 0.2.
        random_state (int, optional): Random state for reproducibility. Defaults to 12345.
        method (str, optional): Sampling method ('NearMiss' or 'cluster'). Defaults to 'NearMiss'.
        version (int, optional): Version of NearMiss to use. Defaults to 2.

    Returns:
        pd.DataFrame: Rebalanced DataFrame.
    """
    if method == 'NearMiss':
        sampler = NearMiss(sampling_strategy=sampling_strategy, version=version, n_jobs=-1)
    else:
        raise ValueError("Unsupported method. Use 'NearMiss'.")
    
    X = df.drop(columns=[event_col, cluster_col])
    y = df[event_col]
    
    X_res, y_res = sampler.fit_resample(X, y)
    X_res[event_col] = y_res
    
    # Check for NaN values and impute if necessary
    if X_res.isnull().values.any():
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=random_state, initial_strategy='mean')
        X_res = pd.DataFrame(imputer.fit_transform(X_res), columns=X_res.columns)
        logging.info("Missing values imputed using IterativeImputer.")
    
    X_res[cat_features] = X_res[cat_features].round()
    logging.info("Dataframe for DeepHit model rebalanced.")
    
    return X_res
