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
from typing import Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_imputed_datasets_hdf5(base_filename: str) -> dict:
    """
    Reads imputed datasets from an HDF5 file.
    
    Args:
        base_filename (str): Base name of the HDF5 file (without extension).
        
    Returns:
        dict: A dictionary containing datasets grouped by estimator names.
    """
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
    """
    Custom transformer to filter rows based on a condition applied to a specific column.
    
    Args:
        column (str): Name of the column to apply the filter on.
        condition (callable): A callable that takes a pandas Series and returns a boolean mask.
    """
    def __init__(self, column: str, condition):
        self.column = column
        self.condition = condition
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()  # To avoid modifying the original DataFrame
        if self.column in X.columns:
            X = X[self.condition(X[self.column])]
        return X.reset_index(drop=True)

class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for log transformation of numerical columns.
    
    Args:
        offset (float): A small value added to avoid log of zero.
    """
    def __init__(self, offset: float = 1e-6):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)  # Ensure X is a DataFrame
        # Apply log transformation column-wise
        X = X.apply(lambda col: np.log(col + self.offset) if np.issubdtype(col.dtype, np.number) else col)
        return X

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to convert numpy arrays back to DataFrames with specific column names and data types.
    
    Args:
        column_names (list): List of column names.
        dtypes (dict, optional): Dictionary of column names and their respective data types.
    """
    def __init__(self, column_names: List[str], dtypes: dict = None):
        self.column_names = column_names
        self.dtypes = dtypes or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.column_names)
        for column, dtype in self.dtypes.items():
            X[column] = X[column].astype(dtype)
        return X

class DataFrameShuffler(BaseEstimator, TransformerMixin):
    """
    Transformer to shuffle the rows of a DataFrame.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return shuffle(X).reset_index(drop=True)

def create_pipeline(cat_features: List[str], log_features: List[str], standard_features: List[str], passthrough_features: List[str]) -> Pipeline:
    """
    Creates a data preprocessing pipeline with imputation, transformation, and scaling steps.
    
    Args:
        cat_features (list): List of categorical feature names.
        log_features (list): List of features to be log-transformed.
        standard_features (list): List of features to be standardized.
        passthrough_features (list): List of features to be passed through without transformation.
        
    Returns:
        Pipeline: A scikit-learn pipeline for data preprocessing.
    """
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

def load_and_transform_data(base_filename: str, cat_features: List[str], log_features: List[str], standard_features: List[str], passthrough_features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and transforms data using the specified pipeline.
    
    Args:
        base_filename (str): Base name of the HDF5 file containing the datasets.
        cat_features (list): List of categorical feature names.
        log_features (list): List of features to be log-transformed.
        standard_features (list): List of features to be standardized.
        passthrough_features (list): List of features to be passed through without transformation.
        
    Returns:
        tuple: Transformed training and test DataFrames.
    """
    pipeline = create_pipeline(cat_features, log_features, standard_features, passthrough_features)
    X_load = read_imputed_datasets_hdf5(base_filename)
    X_train = X_load['X_train_main'][0]
    X_test = X_load['X_test_main'][0]
    logging.info("Transforming training data...")
    X_train_transformed = pipeline.fit_transform(X_train)
    logging.info("Transforming test data...")
    X_test_transformed = pipeline.transform(X_test)
    
    # Clean up
    del X_load, X_train, X_test
    gc.collect()
    
    return X_train_transformed, X_test_transformed

def prepare_features(df: pd.DataFrame, feature_col: List[str]) -> np.ndarray:
    """
    Extracts feature columns from a DataFrame.
    
    Args:
        df (DataFrame): The DataFrame containing the data.
        feature_col (list): The list of feature column names.
    
    Returns:
        np.ndarray: Numpy array containing the feature values.
    """
    return df[feature_col].values.astype('float32')

def prepare_labels(df: pd.DataFrame, duration_col: str, event_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts duration and event columns from a DataFrame.
    
    Args:
        df (DataFrame): The DataFrame containing the data.
        duration_col (str): The column name for the duration.
        event_col (str): The column name for the event.
    
    Returns:
        tuple: Duration and event arrays.
    """
    duration = df[duration_col].values.astype('float32').squeeze()
    event = df[event_col].values.astype('int').squeeze()
    return duration, event

def preprocess_data(df: pd.DataFrame, feature_col: List[str], duration_col: str, event_col: str, time_grid: np.ndarray = None, discretize: bool = False) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares the data by extracting and optionally discretizing the duration, event, and feature columns.
    
    Args:
        df (DataFrame): The DataFrame containing the data.
        feature_col (list of str): The list of feature columns.
        duration_col (str): The column name for the duration.
        event_col (str): The column name for the event.
        time_grid (array, optional): The time grid for discretization.
        discretize (bool, optional): Whether to discretize the durations (default is False).
    
    Returns:
        tuple: Processed training features (X) and target tensor (y).
    """
    X = prepare_features(df, feature_col)
    duration, event = prepare_labels(df, duration_col, event_col)
    
    if discretize and time_grid is not None:
        labtrans = LabTransDiscreteTime(time_grid)
        binary_events = np.where(event > 0, 1, 0)
        durations, events = labtrans.transform(duration.astype('int64'), binary_events)
        y = (durations.astype('int').squeeze(), event.astype('int').squeeze())
    else:
        y = (duration, event)
    
    return X, y

def stack_sequences_generic(
    original_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_col: List[str],
    duration_col: str,
    event_col: str,
    cluster_col: str,
    seq_length: int = 5,
    time_grid: np.ndarray = None,
    discretize: bool = False
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stacks sequences of feature data for model training.
    
    Args:
        original_df (DataFrame): Original DataFrame containing all data.
        target_df (DataFrame): Target DataFrame containing data to be used for sequence stacking.
        feature_col (list): List of feature column names.
        duration_col (str): Name of the duration column.
        event_col (str): Name of the event column.
        cluster_col (str): Name of the column used to group data for sequence stacking.
        seq_length (int, optional): Length of the sequences to be stacked (default is 5).
        time_grid (array, optional): The time grid for discretization.
        discretize (bool, optional): Whether to discretize the durations (default is False).
    
    Returns:
        tuple: Sequences of feature data and target tensor (durations and events).
    """
    num_samples = len(target_df)
    num_features = len(feature_col)
    
    sequences = np.zeros((num_samples, seq_length, num_features))
    durations, events = prepare_labels(target_df, duration_col, event_col)

    # Map cluster_col to their index in target_df
    cluster_map = original_df.groupby(cluster_col).apply(lambda x: x.sort_values(by=duration_col).index.values).to_dict()

    for i, idx in enumerate(target_df.index):
        key = target_df.loc[idx, cluster_col]
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

    if discretize and time_grid is not None:
        labtrans = LabTransDiscreteTime(time_grid)
        binary_events = np.where(events > 0, 1, 0)
        durations, events = labtrans.transform(durations.astype('int64'), binary_events)
    
    y = (durations.squeeze(), events.squeeze())
    
    return sequences, y


def prep_tensor(df: pd.DataFrame, feature_col: List[str], duration_col: str, event_col: str) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares the tensors for training.
    
    Args:
        df (DataFrame): DataFrame containing the features, duration, and event columns.
        feature_col (list): List of feature column names.
        duration_col (str): Name of the duration column.
        event_col (str): Name of the event column.
        
    Returns:
        tuple: Features tensor (X) and target tensor (y) containing durations and events.
    """
    X = df[feature_col].values.astype('float32')
    duration = df[duration_col].values.astype('float32').squeeze()
    event = df[event_col].values.astype('int').squeeze()
    y = (duration, event)
    return X, y

def stack_sequences(original_df: pd.DataFrame, target_df: pd.DataFrame, feature_col: List[str], duration_col: str, event_col: str, cluster_col: str, seq_length: int = 5) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stacks sequences of feature data for LSTM training.
    
    Args:
        original_df (DataFrame): Original DataFrame containing all data.
        target_df (DataFrame): Target DataFrame containing data to be used for sequence stacking.
        feature_col (list): List of feature column names.
        duration_col (str): Name of the duration column.
        event_col (str): Name of the event column.
        cluster_col (str): Name of the column used to group data for sequence stacking.
        seq_length (int, optional): Length of the sequences to be stacked (default is 5).
        
    Returns:
        tuple: Sequences of feature data and target tensor (durations and events).
    """
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

# Add similar improvements for the rest of the DeepHit related functions, ensuring modularization, error handling, and documentation throughout.
def discretize_durations_and_events(durations: np.ndarray, events: np.ndarray, cut_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Discretizes durations based on the cut points and assigns event labels.
    
    Args:
        durations (np.ndarray): Array of durations (in days).
        events (np.ndarray): Array of event indicators (0 = censored, 1 = start RRT / eGFR < 15, 2 = all-cause mortality).
        cut_points (np.ndarray): Array of cut points to define time intervals (in days).
        
    Returns:
        tuple: Indices representing the time bin each duration falls into and transformed event indicators.
    """
    # Ensure cut points are sorted
    cut_points = np.sort(cut_points)
    
    # Find the index of the bin each duration belongs to
    idx_durations = np.digitize(durations, bins=cut_points, right=True).ravel().astype('int')

    # Keep the original event labels (1 = RRT, 2 = all-cause mortality), 0 for censored
    transformed_events = np.where(events == 0, 0, events).ravel().astype('int')

    return idx_durations, transformed_events

def dh_dataset_loader(df: pd.DataFrame, duration_col: str, event_col: str, feature_col: List[str], time_grid: np.ndarray = None) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Prepares the data by extracting and transforming the duration, event, and feature columns and discretizing them based on the given time grid.
    
    Args:
        df (DataFrame): The DataFrame containing the data.
        duration_col (str): The column name for the duration.
        event_col (str): The column name for the event.
        feature_col (list of str): The list of feature columns.
        time_grid (array, optional): The time grid for discretization (default is 1-5 years).
        
    Returns:
        tuple: Processed training features (X) and target tensor (y) containing discretized durations and events.
    """
    if time_grid is None:
        # Default time grid (in days): 1 year, 2 years, 3 years, 4 years, 5 years
        time_grid = np.array([i * 365 for i in range(0, 6)])
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

def dh_stack_sequences(original_df: pd.DataFrame, target_df: pd.DataFrame, feature_col: List[str], duration_col: str, event_col: str, cluster_col: str, time_grid: np.ndarray, seq_length: int = 5) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Stacks sequences of feature data for DeepHit model training.
    
    Args:
        original_df (DataFrame): Original DataFrame containing all data.
        target_df (DataFrame): Target DataFrame containing data to be used for sequence stacking.
        feature_col (list): List of feature column names.
        duration_col (str): Name of the duration column.
        event_col (str): Name of the event column.
        cluster_col (str): Name of the column used to group data for sequence stacking.
        time_grid (array): The time grid for discretization.
        seq_length (int, optional): Length of the sequences to be stacked (default is 5).
        
    Returns:
        tuple: Sequences of feature data and target tensor (discretized durations and events).
    """
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
