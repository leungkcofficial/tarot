from databalancer import define_medoid, df_event_focus, underbalance_data, dh_define_medoid, dh_underbalance_data, dh_rebalance_data
from dataloader import prep_tensor, stack_sequences, dh_dataset_loader, dh_stack_sequences
import pandas as pd
import numpy as np
import math
import torch
import gc

def recursive_clustering(model, df, event_focus, duration_col, event_col, feature_col, params, val, callbacks, max_repeats):
    repeat_count = 0
    logs = []
    
    remaining_data = df_event_focus(df=df, event_col=event_col, event_focus=event_focus)
    # remaining_data = df.copy()
    df_minor = remaining_data[remaining_data[event_col[0]] == event_focus].copy()
    df_major = remaining_data[remaining_data[event_col[0]] != event_focus].copy()
    
    if max_repeats == -1:
        goal = round(len(df_major) / len(df_minor)) - 1
    else:
        goal = round(1 / params['sampling_strategy'])

    while len(remaining_data) > 0 and repeat_count < goal:
        print(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        X_cluster, remaining_data = define_medoid(df=remaining_data, feature_col=feature_col, event_col=event_col)

        X_train_cluster, y_train_cluster = prep_tensor(df=X_cluster, feature_col=feature_col, duration_col=duration_col, event_col=event_col)

        log = model.fit(X_train_cluster, y_train_cluster, params['batch_size'], params['max_epochs'], callbacks, verbose=True, val_data=val)
        logs.append(log)
        gc.collect()

        # Early stopping check
        if callbacks and hasattr(callbacks[0], 'stopped_epoch'):
            if callbacks[0].stopped_epoch > 0:
                print(f"Early stopping at epoch {callbacks[0].stopped_epoch}")
                break

        repeat_count += 1

        if remaining_data.empty:
            break

    return model, logs

def lstm_prepare_training_data(df, feature_col, duration_col, event_col, event_focus, params, cluster_col):
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    remaining_data = df.copy()
    goal = round(1 / params['sampling_strategy'])
    repeat_count = 0
    all_clusters = []
    seq_length = params['seq_length']
    
    while len(remaining_data) > 0 and repeat_count < goal:
        print(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        X_cluster, remaining_data = define_medoid(df=remaining_data, feature_col=feature_col, event_col=event_col)
        # Append the current X_cluster to the list
        all_clusters.append(X_cluster)
        # Increment the repeat count
        repeat_count += 1
    # After the loop, concatenate all X_cluster DataFrames
    concatenated_clusters = pd.concat(all_clusters, ignore_index=False)
    # Remove duplicated rows, keeping only the first occurrence
    final_clusters = concatenated_clusters.drop_duplicates(keep='first')
    # final_clusters = underbalance_data(df, event_col=event_col,
    #                                    cluster_col=cluster_col, sampling_strategy=params['sampling_strategy'])
    print('Cluster data retrieved')
    X, y = stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, seq_length)
    return X, y
    
def lstm_prepare_validation_data(df, feature_col, duration_col, event_col, event_focus, params, cluster_col):
    df = df_event_focus(df, event_col=event_col, event_focus=event_focus)
    seq_length = params['seq_length']
    X, y = stack_sequences(df, df, feature_col, duration_col, event_col, cluster_col, seq_length)
    return X, y

def lstm_training(model, train_df, val_df, event_focus, duration_col, event_col, feature_col, cluster_col, params, callbacks):
    gc.collect()
    X_train, y_train = lstm_prepare_training_data(df=train_df, feature_col=feature_col, duration_col=duration_col, event_col=event_col, 
                                                    event_focus=event_focus, params=params, cluster_col=cluster_col)
    X_val, y_val = lstm_prepare_validation_data(df=val_df, feature_col=feature_col, duration_col=duration_col, event_col=event_col, 
                                                event_focus=event_focus, params=params, cluster_col=cluster_col)
    gc.collect()
    torch.cuda.empty_cache()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = (torch.tensor(y_train[0], dtype=torch.float32), torch.tensor(y_train[1], dtype=torch.float32))

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = (torch.tensor(y_val[0], dtype=torch.float32), torch.tensor(y_val[1], dtype=torch.float32))
    val_data = (X_val_tensor, y_val_tensor)
    
    # Determine batch size
    dataset_size = X_train_tensor.size()[0]
    batch_size = min(params['batch_size'], dataset_size)
    # Adjust batch size to avoid single-item last batch
    if dataset_size % batch_size == 1:
        batch_size = math.ceil(dataset_size / (math.floor(dataset_size / batch_size) + 1))
    logs = model.fit(X_train_tensor, y_train_tensor, batch_size, params['max_epochs'], callbacks, verbose=True, val_data=val_data, num_workers=10)
    
    return model, logs

### DeepHit
def dh_recursive_clustering(model, df, duration_col, event_col, feature_col, params, val, callbacks, max_repeats):
    repeat_count = 0
    logs = []
    
    remaining_data = df.copy()
    df_minor = remaining_data[remaining_data[event_col[0]] != 0].copy()
    df_major = remaining_data[remaining_data[event_col[0]] == 0].copy()
    
    if max_repeats == -1:
        goal = round(len(df_major) / len(df_minor)) - 1
    else:
        goal = round(1 / params['sampling_strategy'])

    while len(remaining_data) > 0 and repeat_count < goal:
        print(f"Performing clustering iteration {repeat_count + 1} / {goal}")
        X_cluster, remaining_data = dh_define_medoid(df=remaining_data, feature_col=feature_col, event_col=event_col)

        X_train_cluster, y_train_cluster = dh_dataset_loader(X_cluster, duration_col, event_col, feature_col)

        log = model.fit(X_train_cluster, y_train_cluster, params['batch_size'], params['max_epochs'], callbacks, verbose=True, val_data=val)
        logs.append(log)
        gc.collect()

        # Early stopping check
        if callbacks and hasattr(callbacks[0], 'stopped_epoch'):
            if callbacks[0].stopped_epoch > 0:
                print(f"Early stopping at epoch {callbacks[0].stopped_epoch}")
                break

        repeat_count += 1

        if remaining_data.empty:
            break

    return model, logs

def dh_lstm_prepare_training_data(df, feature_col, duration_col, event_col, params, cluster_col, time_grid, method = 'cluster'):
    remaining_data = df.copy()
    goal = round(1 / params['sampling_strategy'])
    repeat_count = 0
    all_clusters = []
    seq_length = params['seq_length']

    if method == 'cluster':
        # Clustering iterations to reduce data into meaningful clusters
        while len(remaining_data) > 0 and repeat_count < goal:
            print(f"Performing clustering iteration {repeat_count + 1} / {goal}")
            X_cluster, remaining_data = dh_define_medoid(df=remaining_data, feature_col=feature_col, event_col=event_col)
            all_clusters.append(X_cluster)
            repeat_count += 1

        # Concatenate all clusters and remove duplicates
        concatenated_clusters = pd.concat(all_clusters, ignore_index=False)
        final_clusters = concatenated_clusters.drop_duplicates(keep='first')
    else:
        final_clusters = dh_underbalance_data(df, event_col, cluster_col, params['sampling_strategy'])

    # Stack sequences for the LSTM input
    X, y = dh_stack_sequences(df, final_clusters, feature_col, duration_col, event_col, cluster_col, time_grid, seq_length)
    
    return X, y

def dh_lstm_prepare_validation_data(df, feature_col, duration_col, event_col, params, cluster_col, time_grid):
    seq_length = params['seq_length']
    X, y = dh_stack_sequences(df, df, feature_col, duration_col, event_col, cluster_col, time_grid, seq_length)
    return X, y

def dh_lstm_training(model, train_df, val_df, feature_col, duration_col, event_col, cluster_col, params, callbacks, time_grid, method='NearMiss'):
    gc.collect()
    X_train, y_train = dh_lstm_prepare_training_data(df=train_df, feature_col=feature_col, duration_col=duration_col, event_col=event_col, 
                                                    params=params, cluster_col=cluster_col, time_grid=time_grid, method=method)
    X_val, y_val = dh_lstm_prepare_validation_data(df=val_df, feature_col=feature_col, duration_col=duration_col, event_col=event_col, 
                                                    params=params, cluster_col=cluster_col, time_grid=time_grid)
    val_data = (X_val, y_val)
    gc.collect()
    torch.cuda.empty_cache()
    
    # Determine batch size
    dataset_size = np.shape(X_train)[0]
    batch_size = min(params['batch_size'], dataset_size)
    # Adjust batch size to avoid single-item last batch
    if dataset_size % batch_size == 1:
        batch_size = math.ceil(dataset_size / (math.floor(dataset_size / batch_size) + 1))
    logs = model.fit(X_train, y_train, batch_size, params['max_epochs'], callbacks, verbose=True, val_data=val_data)
    
    return model, logs

def dh_ann_training(model, train_df, val_df, feature_col, duration_col, event_col, cluster_col, cat_features, params, callbacks, time_grid, method='NearMiss'):
    gc.collect()
    version = params['version']
    
    train_df = dh_rebalance_data(train_df, event_col, cluster_col, cat_features, sampling_strategy=params['sampling_strategy'], method=method, version=version)
    print(train_df[event_col].value_counts())
    X_train, y_train = dh_dataset_loader(train_df, duration_col, event_col, feature_col, time_grid=time_grid)
    
    X_val, y_val = dh_dataset_loader(val_df, duration_col, event_col, feature_col, time_grid=time_grid)
    val_data = (X_val, y_val)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Determine batch size
    dataset_size = np.shape(X_train)[0]
    batch_size = min(params['batch_size'], dataset_size)
    # Adjust batch size to avoid single-item last batch
    if dataset_size % batch_size == 1:
        batch_size = math.ceil(dataset_size / (math.floor(dataset_size / batch_size) + 1))
    logs = model.fit(X_train, y_train, batch_size, params['max_epochs'], callbacks, verbose=True, val_data=val_data)
    
    return model, logs