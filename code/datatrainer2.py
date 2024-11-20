import logging
import pandas as pd
import numpy as np
import math
import torch
import gc
from databalancer2 import define_medoid_general, df_event_focus, underbalance_data_general, dh_rebalance_data
from dataloader2 import preprocess_data, stack_sequences, dh_dataset_loader, dh_stack_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recursive_clustering(model, df, duration_col, event_col, params, val, callbacks, max_repeats, time_grid=None):
    """
    Recursively performs clustering on the given dataset and trains the model on each cluster.

    Args:
        model: The model to train.
        df (pd.DataFrame): Input DataFrame.
        event_focus (int): The event value to focus on.
        duration_col (str): The duration column.
        event_col (list): The event column.
        feature_col (list): The feature columns.
        params (dict): Training parameters.
        val: Validation data.
        callbacks (list): List of callbacks for training.
        max_repeats (int): Maximum number of repeats for clustering.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.

    Returns:
        tuple: Trained model and logs.
    """
    repeat_count = 0
    logs = []
    event_focus = params['endpoint']
    feature_col = params['features']
    model_type= params['model']
    
    remaining_data = df_event_focus(df=df, event_col=event_col, event_focus=event_focus) if model_type == 'deepsurv' else df.copy()
    df_minor = remaining_data[remaining_data[event_col] == event_focus].copy() if model_type == 'deepsurv' else remaining_data[remaining_data[event_col] != 0].copy()
    df_major = remaining_data[remaining_data[event_col] != event_focus].copy() if model_type == 'deepsurv' else remaining_data[remaining_data[event_col] == 0].copy()
    
    goal = round(len(df_major) / len(df_minor)) - 1 if max_repeats == -1 else round(1 / params['sampling_strategy'])

    while len(remaining_data) > 0 and repeat_count < goal:
        logging.info(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        if model_type == 'deepsurv':
            X_cluster, remaining_data = define_medoid_general(df=remaining_data, feature_col=feature_col, event_col=event_col)
            X_train_cluster, y_train_cluster = preprocess_data(df=X_cluster, feature_col=feature_col, duration_col=duration_col, event_col=event_col)
        elif  model_type == 'deephit':
            X_cluster, remaining_data = define_medoid_general(df=remaining_data, feature_col=feature_col, event_col=event_col, event_focus=event_focus, model_type=model_type)
            X_train_cluster, y_train_cluster = preprocess_data(df=X_cluster, feature_col=feature_col, duration_col=duration_col, event_col=event_col, time_grid=time_grid, discretize=True)
        log = model.fit(X_train_cluster, y_train_cluster, params['batch_size'], params['max_epochs'], callbacks, verbose=True, val_data=val)
        logs.append(log)
        gc.collect()

        # Early stopping check
        if callbacks and hasattr(callbacks[0], 'stopped_epoch') and callbacks[0].stopped_epoch > 0:
            logging.info(f"Early stopping at epoch {callbacks[0].stopped_epoch}")
            break

        repeat_count += 1

    return model, logs

def prepare_training_data(df, feature_col, duration_col, event_col, params, cluster_col, clustering_method='define_medoid', time_grid=None):
    """
    Prepares training data by clustering the given dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_col (list): The feature columns.
        duration_col (str): The duration column.
        event_col (list): The event column.
        event_focus (int): The event value to focus on.
        params (dict): Training parameters.
        cluster_col (str): Cluster column name.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.
        clustering_method (str, optional): Method for clustering. Defaults to 'define_medoid'.
        time_grid (list, optional): Time grid for sequences (for DeepHit). Defaults to None.

    Returns:
        tuple: Features and labels for training.
    """
    goal = round(1 / params['sampling_strategy'])
    repeat_count = 0
    all_clusters = []
    
    seq_length = params['seq_length']
    event_focus = params['endpoint']
    model_type = params['model']
    
    # df = df_event_focus(df, event_col=event_col, event_focus=event_focus) if model_type == 'deepsurv' else df.copy()
    remaining_data = df.copy()    
    
    if params['balance_method'] == 'clustering':
        while len(remaining_data) > 0 and repeat_count < goal:
            logging.info(f"Performing clustering iteration {repeat_count + 1} / {goal}")
            if model_type == 'deepsurv':
                X_cluster, remaining_data = define_medoid_general(df=remaining_data, feature_col=feature_col, event_col=event_col, event_focus=event_focus)
            elif model_type == 'deephit':
                X_cluster, remaining_data = define_medoid_general(df=remaining_data, feature_col=feature_col, event_col=event_col)
            all_clusters.append(X_cluster)
            repeat_count += 1
        final_clusters = pd.concat(all_clusters, ignore_index=False).drop_duplicates(keep='first')
        logging.info("Cluster data retrieved")
    elif params['balance_method'] == 'NearMiss':
        final_clusters = underbalance_data_general(df, event_col, cluster_col, params)
    if model_type == 'deepsurv':
        X, y = stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, seq_length)
    else:
        X, y = dh_stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, time_grid, seq_length)
    return X, y

def prepare_validation_data(df, feature_col, duration_col, event_col, params, cluster_col, model_type='deepsurv', time_grid=None):
    """
    Prepares validation data by stacking sequences.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_col (list): The feature columns.
        duration_col (str): The duration column.
        event_col (list): The event column.
        params (dict): Training parameters.
        cluster_col (str): Cluster column name.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.
        time_grid (list, optional): Time grid for sequences (for DeepHit). Defaults to None.

    Returns:
        tuple: Features and labels for validation.
    """
    seq_length = params['seq_length']
    event_focus = params['endpoint']
    if model_type == 'deepsurv':
        df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
        X, y = stack_sequences(df, df, feature_col, duration_col, event_col, cluster_col, seq_length)
    else:
        X, y = dh_stack_sequences(df, df, feature_col, duration_col, event_col, cluster_col, time_grid, seq_length)
    logging.info("Validation data retrieved")
    return X, y

def lstm_training(model, train_df, val_df, duration_col, event_col, cluster_col, params, callbacks, time_grid=None):
    """
    Trains an LSTM model with the given training and validation data.

    Args:
        model: The LSTM model to train.
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.
        event_focus (int): The event value to focus on.
        duration_col (str): The duration column.
        event_col (list): The event column.
        feature_col (list): The feature columns.
        cluster_col (str): Cluster column name.
        params (dict): Training parameters.
        callbacks (list): List of callbacks for training.
        model_type (str, optional): Type of model ('deepsurv' or 'deephit'). Defaults to 'deepsurv'.
        time_grid (list, optional): Time grid for sequences (for DeepHit). Defaults to None.

    Returns:
        tuple: Trained model and logs.
    """
    model_type = params['model']
    event_focus = params['endpoint']
    feature_col = params['features']
    
    gc.collect()
    X_train, y_train = prepare_training_data(train_df, feature_col, duration_col, event_col, params, cluster_col, model_type, time_grid)
    X_val, y_val = prepare_validation_data(val_df, feature_col, duration_col, event_col, params, cluster_col, model_type, time_grid)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = (torch.tensor(y_train[0], dtype=torch.float32), torch.tensor(y_train[1], dtype=torch.float32))

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = (torch.tensor(y_val[0], dtype=torch.float32), torch.tensor(y_val[1], dtype=torch.float32))
    val_data = (X_val_tensor, y_val_tensor)
    
    dataset_size = X_train_tensor.size()[0]
    print(X_val_tensor.size()[0])
    batch_size = min(params['batch_size'], dataset_size)
    if dataset_size % batch_size == 1:
        batch_size = math.ceil(dataset_size / (math.floor(dataset_size / batch_size) + 1))
    
    if model_type == 'deepsurv':
        logs = model.fit(X_train_tensor, y_train_tensor, batch_size, params['max_epochs'], callbacks, verbose=True, val_data=val_data, num_workers=10)
    elif model_type == 'deephit':
        logs = model.fit(X_train, y_train, batch_size, params['max_epochs'], callbacks, verbose=True, val_data=(X_val, y_val), num_workers=10)
    return model, logs
