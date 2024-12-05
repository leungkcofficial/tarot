import os
import json
import h5py
import numpy as np
import pandas as pd
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

import modeleval
from modeleval import (
    dh_test_model, nam_dagostino_chi2, get_baseline_hazard_at_timepoints, combined_test_model, evaluate_cif_predictions
)

def load_bootstrap_predictions(directory_path):
    """
    Load bootstrap predictions, durations, and events from multiple HDF5 files.

    Args:
        directory_path (str): Path to the directory containing HDF5 files.

    Returns:
        dict: A dictionary containing all the bootstrap results.
            - "predictions": A dictionary of CIF predictions for each bootstrap iteration.
            - "durations": A list of duration arrays for each iteration.
            - "events": A list of event arrays for each iteration.
    """
    bootstrap_results = {
        "predictions": {},
        "durations": [],
        "events": []
    }

    # Iterate over all HDF5 files in the directory
    for file_name in sorted(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, file_name)
        
        if file_name.endswith(".h5"):
            print(f"Loading {file_path}...")
            with h5py.File(file_path, "r") as hdf:
                # Extract iteration key from file name or structure
                iteration_key = file_name.replace("bootstrap_iteration_", "").replace(".h5", "")
                
                # Load predictions
                predictions = {}
                for model_key in hdf["predictions"].keys():
                    predictions[model_key] = hdf["predictions"][model_key][:]
                bootstrap_results["predictions"][iteration_key] = predictions

                # Load durations and events
                durations = hdf["durations"][:]
                events = hdf["events"][:]
                bootstrap_results["durations"].append(durations)
                bootstrap_results["events"].append(events)
    
    print(f"Loaded bootstrap iterations from {directory_path}.")
    return bootstrap_results

def generate_combinations(labels):
    """
    Generate all possible combinations of labels from 1 to max length.

    Args:
        labels (list): List of CIF labels.

    Returns:
        list: List of all combinations.
    """
    all_combinations = []
    for r in range(1, len(labels) + 1):
        all_combinations.extend(combinations(labels, r))
    return all_combinations

def get_cif_for_combination(predictions, combination):
    """
    Retrieve CIF arrays for a specific combination of CIF labels.

    Args:
        predictions (dict): Dictionary containing CIF arrays for all labels.
        combination (tuple): A tuple of CIF labels to combine.

    Returns:
        dict: A dictionary where keys are the CIF labels and values are their corresponding arrays.
    """
    cif_arrays = {}
    for label in combination:
        if label in predictions:
            cif_arrays[label] = predictions[label]
        else:
            raise KeyError(f"Label '{label}' not found in predictions.")
    return cif_arrays

def ensemble_cif_arrays(cif_arrays):
    """
    Ensemble CIF arrays by averaging across all provided CIF arrays.

    Args:
        cif_arrays (dict): A dictionary where keys are model labels, 
                           and values are their corresponding CIF arrays.

    Returns:
        np.ndarray: A final CIF array of shape (2, 6, N) where N is the number of rows in the input CIF arrays.
    """
    # Ensure there are CIF arrays to process
    if not cif_arrays:
        raise ValueError("The cif_arrays dictionary is empty.")

    # Initialize the final CIF array with zeros (same shape as any CIF array)
    sample_shape = next(iter(cif_arrays.values())).shape  # Get the shape from the first array
    final_cif_array = np.zeros(sample_shape, dtype=np.float32)

    # Compute the mean across all arrays for each outcome
    for label, cif_array in cif_arrays.items():
        final_cif_array[0] += cif_array[0]  # Add up CIFs for event 1
        final_cif_array[1] += cif_array[1]  # Add up CIFs for event 2

    # Divide by the number of arrays to get the mean
    num_arrays = len(cif_arrays)
    final_cif_array[0] /= num_arrays
    final_cif_array[1] /= num_arrays

    return final_cif_array

def save_metrics_to_individual_json(output_dir, bootstrap_idx, combo, metrics):
    """
    Save metrics for a specific bootstrap index and combination to a separate JSON file.

    Args:
        output_dir (str): Directory where JSON files will be saved.
        bootstrap_idx (int): The bootstrap iteration index.
        combo (tuple): Combination of models.
        metrics (dict): Calculated metrics for the combination.
    """
    # Ensure the directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique file name for this bootstrap iteration
    file_name = f"bootstrap_{bootstrap_idx}_combo_{'_'.join(combo)}.json"
    file_path = os.path.join(output_dir, file_name)

    # Convert all non-serializable objects in metrics to JSON-serializable formats
    def make_serializable(obj):
        if isinstance(obj, (np.ndarray, pd.Series)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        else:
            return obj

    metrics_serializable = make_serializable(metrics)

    # Save metrics to the file
    with open(file_path, "w") as outfile:
        json.dump({str(combo): metrics_serializable}, outfile, indent=4)
    print(f"Metrics for Bootstrap {bootstrap_idx}, Combination {combo} saved to {file_path}.")

def get_processed_combinations(output_dir):
    """
    Retrieve processed bootstrap indices and combinations from existing JSON files.

    Args:
        output_dir (str): Directory containing JSON files.

    Returns:
        set: A set of tuples containing (bootstrap_idx, combo).
    """
    processed_combinations = set()

    # Iterate over all JSON files in the output directory
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)

            try:
                # Read the JSON file
                with open(file_path, "r") as infile:
                    data = json.load(infile)

                # Extract the key (combination) and ensure it's interpreted as a tuple
                for combo_str, _ in data.items():
                    combo = eval(combo_str)  # Safely evaluate the string to a tuple
                    bootstrap_idx = int(file_name.split("_")[1])  # Extract bootstrap_idx from filename
                    processed_combinations.add((bootstrap_idx, combo))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return processed_combinations


def process_combination_worker(args):
    """
    Worker function for processing a combination.

    Args:
        args (tuple): A tuple containing:
            - bootstrap_idx (int): Bootstrap iteration index.
            - combo (tuple): Combination of CIF labels.
            - itr_predictions (dict): CIF predictions for this iteration.
            - itr_durations (np.ndarray): Durations for this iteration.
            - itr_events (np.ndarray): Events for this iteration.
            - output_dir (str): Directory to save metrics JSON files.

    Returns:
        str: Status of the processing.
    """
    bootstrap_idx, combo, itr_predictions, itr_durations, itr_events, output_dir = args
    try:
        cif_arrays = get_cif_for_combination(itr_predictions, combo)
        final_cif = ensemble_cif_arrays(cif_arrays)
        metrics = evaluate_cif_predictions(final_cif, itr_durations, itr_events, np.array([0, 1, 2, 3, 4, 5]), nam_dagostino_chi2)
        save_metrics_to_individual_json(output_dir, bootstrap_idx, combo, metrics)
        return f"Processed Bootstrap {bootstrap_idx}, Combination {combo}"
    except KeyError as e:
        return f"KeyError for Combination {combo}: {e}"
    except Exception as e:
        return f"Error in Bootstrap {bootstrap_idx}, Combination {combo}: {e}"

def process_combination(bootstrap_idx, combo, itr_predictions, itr_durations, itr_events, output_dir):
    """
    Process a single combination of CIFs and save metrics.
    """
    try:
        cif_arrays = get_cif_for_combination(itr_predictions, combo)
        final_cif = ensemble_cif_arrays(cif_arrays)

        # Evaluate metrics
        metrics = evaluate_cif_predictions(
            final_cif, itr_durations, itr_events, np.array([0, 1, 2, 3, 4, 5]), nam_dagostino_chi2
        )

        save_metrics_to_individual_json(output_dir, bootstrap_idx, combo, metrics)
        return f"Processed bootstrap {bootstrap_idx}, combination {combo}"
    except Exception as e:
        return f"Error in bootstrap {bootstrap_idx}, combination {combo}: {e}"

def process_combination_worker(task):
    """
    Wrapper for multiprocessing.
    """
    return process_combination(*task)

