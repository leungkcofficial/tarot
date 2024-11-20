import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.utils import shuffle
from pycox.preprocessing.label_transforms import LabTransDiscreteTime
import gc

def read_imputed_datasets_hdf5(base_filename):
    datasets = {}
    with pd.HDFStore(f"{base_filename}.h5", 'r') as store:
        for key in store.keys():
            # Split the key to get estimator_name and dataset identifier
            parts = key.strip('/').split('/')
            estimator_name = parts[0]
            dataset_name = '/'.join(parts[1:])
            if estimator_name not in datasets:
                datasets[estimator_name] = []
            datasets[estimator_name].append(store[key])
    return datasets

class RowFilter(BaseEstimator, TransformerMixin):
    def __init__(self, column, condition):
        self.column = column
        self.condition = condition
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()  # To avoid modifying the original DataFrame
        if self.column in X.columns:
            X = X[self.condition(X[self.column])]
        return X.reset_index(drop=True)

# Custom transformer for log transformation
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1e-6):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)  # Ensure X is a DataFrame
        # Apply log transformation column-wise
        X = X.apply(lambda col: np.log(col + self.offset) if np.issubdtype(col.dtype, np.number) else col)
        return X

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_names, dtypes=None):
        self.column_names = column_names
        self.dtypes = dtypes or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.column_names)
        for column, dtype in self.dtypes.items():
            X[column] = X[column].astype(dtype)
        return X
    
class DataFrameShuffler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return shuffle(X).reset_index(drop=True)

def create_pipeline(cat_features, log_features, standard_features, passthrough_features):
    impute_features = cat_features + log_features + standard_features
    dtypes = {
        **{name: 'float' for name in log_features + standard_features},
        **{name: 'category' for name in cat_features},
        'key': 'int'
    }

    pipeline = Pipeline([
        ('impute', ColumnTransformer([
            ('imputer', IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42,
                                         initial_strategy='mean', n_nearest_features=None, min_value=1e-6,
                                       imputation_order='ascending'), impute_features),
            ('passthrough', 'passthrough', passthrough_features)
        ], remainder='drop')),
        ('to_df', DataFrameTransformer(impute_features + passthrough_features, dtypes)),
        ('process', ColumnTransformer([
            ('categorical', Pipeline([
                ('encode', OrdinalEncoder())
            ]), cat_features),
            ('log_and_scale', Pipeline([
                ('log', LogTransformer()),
                ('scaler', StandardScaler()),
                ('minmax', MinMaxScaler(feature_range=(1e-6, 1)))
            ]), log_features),
            ('standard_and_minmax', Pipeline([
                ('standard', StandardScaler()),
                ('minmax', MinMaxScaler(feature_range=(1e-6, 1)))
            ]), [col for col in standard_features if col not in log_features]),
            ('passthrough', 'passthrough', passthrough_features)
        ], remainder='drop')),
        ('to_df2', DataFrameTransformer(impute_features + passthrough_features, dtypes)),
        ('row_filter', RowFilter('date_from_sub_60', lambda x: x <= 1825)),
        ('shuffle', DataFrameShuffler())
    ])
    return pipeline

def load_and_transform_data(base_filename, cat_features, log_features, standard_features, passthrough_features):
    pipeline = create_pipeline(cat_features, log_features, standard_features, passthrough_features)
    X_load = read_imputed_datasets_hdf5(base_filename)
    X_train = X_load['X_train_main'][0]
    X_test = X_load['X_test_main'][0]
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    # Clean up
    del X_load, X_train, X_test
    gc.collect()
    
    return X_train_transformed, X_test_transformed

def prep_tensor(df, feature_col, duration_col, event_col):
    """
    Prepares the tensors for training.
    """
    X = df[feature_col].values.astype('float32')
    duration = df[duration_col].values.astype('float32').squeeze()
    event = df[event_col].values.astype('int').squeeze()
    y = (duration, event)
    return X, y

def stack_sequences(original_df, target_df, feature_col, duration_col, event_col, cluster_col, seq_length=5):
    num_samples = len(target_df)
    num_features = len(feature_col)
    
    sequences = np.zeros((num_samples, seq_length, num_features))
    durations = target_df[duration_col].values
    events = target_df[event_col].values

    # Map cluster_col to their index in target_df
    cluster_map = original_df.groupby(cluster_col).apply(lambda x: x.sort_values(by=duration_col).index.values).to_dict()

    for i, idx in enumerate(target_df.index):
        key = target_df.loc[idx, cluster_col]
        
        # Ensure key is a scalar value, not a Series
        if isinstance(key, pd.Series):
            key = key.iloc[0]
        elif isinstance(key, np.ndarray):
            key = key[0]

        current_idx = np.where(cluster_map[key] == idx)[0][0]

        # Get the indices for the sequence
        seq_indices = cluster_map[key][max(0, current_idx-seq_length+1):current_idx+1]

        # Retrieve the rows from original_df based on the indices
        seq_data = original_df.loc[seq_indices, feature_col].values

        # Handle padding if the sequence is shorter than seq_length
        if len(seq_data) < seq_length:
            sequences[i, -len(seq_data):, :] = seq_data
        else:
            sequences[i, :, :] = seq_data

    y = (durations.squeeze(), events.squeeze())
    
    return sequences, y

### DeepHit
def discretize_durations_and_events(durations, events, cut_points):
    """
    Discretizes durations based on the cut points and assigns event labels.
    
    Arguments:
    durations -- array-like, list or np.array of durations (in days)
    events -- array-like, list or np.array of event indicators 
              (0 = censored, 1 = start RRT / eGFR < 15, 2 = all-cause mortality)
    cut_points -- np.array of cut points to define time intervals (in days)

    Returns:
    idx_durations -- np.array of indices representing the time bin each duration falls into
    transformed_events -- np.array of event indicators corresponding to the bins
    """
    # Ensure cut points are sorted
    cut_points = np.sort(cut_points)
    
    # Find the index of the bin each duration belongs to
    idx_durations = np.digitize(durations, bins=cut_points, right=True).ravel().astype('int')

    # Keep the original event labels (1 = RRT, 2 = all-cause mortality), 0 for censored
    transformed_events = np.where(events == 0, 0, events).ravel().astype('int')

    return idx_durations, transformed_events

def dh_dataset_loader(df, duration_col, event_col, feature_col, time_grid=None):
    """
    Prepares the data by extracting and transforming the duration, event,
    and feature columns and discretizing them based on the given time grid.

    Args:
    - df (DataFrame): The DataFrame.
    - duration_col (str): The column name for the duration.
    - event_col (str): The column name for the event.
    - feature_col (list of str): The list of feature columns.
    - time_grid (array, optional): The time grid for discretization (default is 1-5 years).

    Returns:
    - X (np.array): Processed training features.
    - y (tuple): Discretized durations and events for training.
    """
    if time_grid is None:
        # Default time grid (in days): 1 year, 2 years, 3 years, 4 years, 5 years
        time_grid = np.array([i * 365 for i in range(0,6)])
    time_grid = time_grid.astype('int64')
    df_durations = df[duration_col].values.astype('int64')
    X = df[feature_col].values.astype('float32')
    df_event = df[event_col].values.astype('int64')
    binary_events = np.where(df_event > 0, 1, 0)
    labtrans = LabTransDiscreteTime(time_grid)
    assert time_grid.dtype == df_durations.dtype, f"dtype mismatch: time_grid is {time_grid.dtype}, df_durations is {df_durations.dtype}"
    assert df_durations.size > 0, "Error: df_durations is an empty array after processing."
    assert binary_events.size > 0, "Error: binary_events is an empty array after processing."
    durations, events = labtrans.transform(df_durations, binary_events)
    y = (durations.astype('int').squeeze(), df_event.astype('int').squeeze())
    assert len(X) == len(y[0]) == len(y[1])
    return X, y

def dh_stack_sequences(original_df, target_df, feature_col, duration_col, event_col, cluster_col, time_grid, seq_length=5):
    num_samples = len(target_df)
    num_features = len(feature_col)
    time_grid = time_grid.astype('int64')

    # Map cluster_col to their index in target_df
    cluster_map = original_df.groupby(cluster_col).apply(lambda x: x.sort_values(by=duration_col).index.values).to_dict()
    df_durations = target_df[duration_col].values.astype('int64')
    df_event = target_df[event_col].values.astype('int64')
    
    # Binary event for discrete time model
    binary_events = np.where(df_event > 0, 1, 0)

    # Transform durations and events using LabTransDiscreteTime
    labtrans = LabTransDiscreteTime(time_grid)
    durations, events = labtrans.transform(df_durations, binary_events)

    sequences = np.zeros((num_samples, seq_length, num_features), dtype=np.float32)

    for i, idx in enumerate(target_df.index):
        key = target_df.loc[idx, cluster_col]

        # Ensure key is a scalar value, not a Series
        if isinstance(key, pd.Series):
            key = key.iloc[0]
        elif isinstance(key, np.ndarray):
            key = key[0]

        current_idx = np.where(cluster_map[key] == idx)[0][0]

        # Get the indices for the sequence
        seq_indices = cluster_map[key][max(0, current_idx - seq_length + 1):current_idx + 1]

        # Retrieve the rows from original_df based on the indices
        seq_data = original_df.loc[seq_indices, feature_col].values

        # Handle padding if the sequence is shorter than seq_length
        if len(seq_data) < seq_length:
            sequences[i, -len(seq_data):, :] = seq_data
        else:
            sequences[i, :, :] = seq_data

    # y is the (discretized) durations and events
    y = (durations.astype('int64').squeeze(), df_event.astype('int').squeeze())
    
    return sequences, y