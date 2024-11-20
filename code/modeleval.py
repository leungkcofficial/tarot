import numpy as np
import matplotlib.pyplot as plt
from pycox.evaluation import EvalSurv
from dataloader import prep_tensor, dh_dataset_loader
from datatrainer import lstm_prepare_validation_data, dh_lstm_prepare_validation_data
from databalancer import df_event_focus
from lifelines import KaplanMeierFitter, AalenJohansenFitter
from sklearn.calibration import calibration_curve
from scipy.stats import chi2
import gc
import torch
import pandas as pd

def get_baseline_hazard_at_timepoints(hazard, time_grid):
    baseline_times = hazard.index.values
    
    def find_closest_day(day):
        if day in baseline_times:
            return day
        else:
            closest_index = np.searchsorted(baseline_times, day, side="left")
            if closest_index == 0:
                return baseline_times[0]
            if closest_index == len(baseline_times):
                return baseline_times[-1]
            before = baseline_times[closest_index - 1]
            after = baseline_times[closest_index]
            if after - day < day - before:
                return after
            else:
                return before
    
    closest_days = np.array([find_closest_day(day) for day in time_grid])
    hazards_at_timepoints = hazard.loc[closest_days]
    
    return hazards_at_timepoints, closest_days

def calculate_aic(model, x_val, y_val):
    # Ensure the input is a tensor
    if isinstance(x_val, np.ndarray):
        x_val = torch.tensor(x_val, dtype=torch.float32).to(model.device)
    if isinstance(y_val, tuple):
        y_val = (torch.tensor(y_val[0], dtype=torch.float32).to(model.device),
                 torch.tensor(y_val[1], dtype=torch.float32).to(model.device))
    # Get the number of parameters
    num_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
    
    # Calculate the log-likelihood
    with torch.no_grad():
        phi = model.net(x_val)
        log_likelihood = -model.loss(phi, y_val[0], y_val[1]).item()
    
    # Calculate AIC
    aic = 2 * num_params - 2 * log_likelihood
    
    return aic

def plot_brier_scores(time_grid, brier_scores_list):
    brier_scores_array = np.array(brier_scores_list)
    mean_brier_scores = brier_scores_array.mean(axis=0)
    std_brier_scores = brier_scores_array.std(axis=0)
    ci_lower = mean_brier_scores - 1.96 * std_brier_scores / np.sqrt(brier_scores_array.shape[0])
    ci_upper = mean_brier_scores + 1.96 * std_brier_scores / np.sqrt(brier_scores_array.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, mean_brier_scores, label='Mean Brier Score')
    plt.fill_between(time_grid, ci_lower, ci_upper, color='b', alpha=0.1, label='95% CI')
    plt.xlabel('Time')
    plt.ylabel('Brier Score')
    plt.title('Brier Score with 95% Confidence Interval')
    plt.legend()
    plt.show()
    
def calculate_nri(old_risk, new_risk, true_outcomes, event_focus=1, cut_points=[0.2, 0.5, 0.8]):
    old_risk = np.concatenate(old_risk)
    new_risk = np.concatenate(new_risk)
    durations, events = zip(*true_outcomes)
    durations = np.concatenate(durations)
    events = np.concatenate(events)

    nri_event = 0
    nri_nonevent = 0

    for lower, upper in zip([0] + cut_points, cut_points + [1]):
        old_reclass_event = ((old_risk >= lower) & (old_risk < upper)).astype(int)
        new_reclass_event = ((new_risk >= lower) & (new_risk < upper)).astype(int)
        
        old_reclass_nonevent = ((old_risk < lower) | (old_risk >= upper)).astype(int)
        new_reclass_nonevent = ((new_risk < lower) | (new_risk >= upper)).astype(int)
        
        nri_event += (new_reclass_event - old_reclass_event)[events == event_focus].sum()
        nri_nonevent += (new_reclass_nonevent - old_reclass_nonevent)[events != event_focus].sum()

    nri_event /= events.sum()
    nri_nonevent /= (1 - events).sum()

    nri = nri_event + nri_nonevent

    return nri, nri_event, nri_nonevent

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Calibration Curve')
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title('Calibration Curve')
    plt.legend()
    plt.show()

def nam_dagostino_chi2(df, duration_col, event_col, surv, time, event_focus=1):
    predicted_probs = 1 - surv.loc[time].values  # Predicted risk at the specific time point

    # Create a DataFrame for calculating Nam-D'Agostino chi-squared statistic
    prob_df = pd.DataFrame({
        'predicted_probs': predicted_probs,
        'observed_probs': np.nan  # Placeholder for observed probabilities
    }, index=df.index)

    # Divide into quintiles
    prob_df['quantile'] = pd.qcut(prob_df['predicted_probs'], 5, labels=False, duplicates='drop')
    # Calculate observed probabilities for each quintile using Aalen-Johansen estimator
    observed_probs = []
    for q in prob_df['quantile'].unique():
        mask = prob_df['quantile'] == q
        durations = pd.Series(df.loc[mask, duration_col].values.ravel(), index=df.loc[mask, duration_col].index)
        event_observed = pd.Series(df.loc[mask, event_col].values.ravel(), index=df.loc[mask, event_col].index)
        ajf = AalenJohansenFitter()
        ajf.fit(durations=durations, event_observed=event_observed, event_of_interest=event_focus)
        # Find the closest time point if `time` is not in the index
        if time not in ajf.cumulative_density_.index:
            closest_idx = ajf.cumulative_density_.index.get_indexer([time], method='nearest')[0]
            closest_time = ajf.cumulative_density_.index[closest_idx]
        else:
            closest_time = time
        observed_probs.append(ajf.cumulative_density_.loc[closest_time].values[0])
            
    # Map observed probabilities back to the DataFrame
    prob_df['observed_probs'] = prob_df['quantile'].map(dict(zip(prob_df['quantile'].unique(), observed_probs)))
    grouped = prob_df.groupby('quantile')
    observed_events = grouped['observed_probs'].mean()
    expected_events = grouped['predicted_probs'].mean()
    # mean_p_g = grouped['predicted_probs'].mean()
    n = grouped.size()
        
    # chi2_stat = np.sum(((observed_events - expected_events) ** 2) / (n * mean_p_g * (1 - mean_p_g)))
    chi2_stat = np.sum(((observed_events - expected_events) ** 2) / (expected_events * (1 - expected_events/ n)))

    # Degrees of freedom
    dof = len(n) - 1
    # Calculate p-value
    p_value = 1 - chi2.cdf(chi2_stat, dof)
    
    return chi2_stat, p_value, observed_events, expected_events, n, prob_df
    
def test_model(model, df, feature_col, duration_col, event_col, event_interest, time_grid, params=None, cluster_col=None):
    df_test = df.copy()
    df_test = df_event_focus(df=df_test, event_col=event_col, event_focus=event_interest)
    df_test_2 = df_test.copy()
    # df_test_2[event_col] = df_test_2[event_col].replace({event_interest: 1})
    X_test, y_test = prep_tensor(df_test_2, feature_col=feature_col, duration_col=duration_col, event_col=event_col)
    # X_test, y_test =  lstm_prepare_validation_data(df_test, feature_col=feature_col, duration_col=duration_col, 
                                                #    event_col=event_col, event_focus=event_interest, params=params, cluster_col=cluster_col)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    # y_test = (torch.tensor(y_test[0], dtype=torch.float32), torch.tensor(y_test[1], dtype=torch.float32))
    model.compute_baseline_hazards()
    _, time_grid = get_baseline_hazard_at_timepoints(model.compute_baseline_hazards(), time_grid)
    
    surv = model.predict_surv_df(X_test_tensor)
    ev = EvalSurv(surv, y_test[0], y_test[1], censor_surv='km')
    concordance_index = ev.concordance_td()
            
    brier_series = ev.brier_score(time_grid)
    integrated_brier_score = ev.integrated_brier_score(time_grid)
    neg_log_likelihood = ev.integrated_nbll(time_grid)
    aic = calculate_aic(model, X_test, y_test)
    
    # Initialize lists to store results    
    results = []

    for time in time_grid:
        chi2_stat, p_value, observed_events, expected_events, n, prob_df = nam_dagostino_chi2(df=df_test, duration_col=duration_col, event_col=event_col, 
                                                                                              surv=surv, time=time, event_focus=event_interest)
        # Store the results for each time point
        results.append({
            'Year': round(time / 365),
            'Chi2_Stat': chi2_stat,
            'P_Value': p_value,
            'Observed_Events': observed_events.tolist(), 
            'Expected_Events': expected_events.tolist(), 
            'Sample_Size': n.tolist() 
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)  
        
    print(f"AIC: {aic} \t concordance index: {concordance_index} \t Brier's score: {integrated_brier_score} \t Negative Log Likelihood: {neg_log_likelihood}") 
    print(f"Nam and D'Agostino Chi2 statistic:")
    print(results_df)
    print(prob_df)
    
    return aic, concordance_index, integrated_brier_score, neg_log_likelihood, brier_series, results

### DeepHit

def dh_test_model(model, df, feature_col, duration_col, event_col, time_grid, params=None, cluster_col=None):
    df_test = df.copy()
    event_interests = np.unique(df_test[event_col].values.astype('int64'))

    # Initialize dictionaries to store results for each event type
    concordance_indices = {}
    integrated_brier_scores = {}
    neg_log_likelihoods = {}
    brier_series = {}
    nam_dagostino_results = []
    
    cif_all = []
    # Prepare the input data only once, as it will be used for all event predictions
    if params == None:
        X_test, y_test = dh_dataset_loader(df_test, duration_col, event_col, feature_col, time_grid=time_grid)
    else:
        X_test, y_test = dh_lstm_prepare_validation_data(df_test, feature_col, duration_col, event_col, params, cluster_col, time_grid=time_grid)
    
    # Predict cumulative incidence functions (CIF) for all competing risks
    cif_total = model.predict_cif(X_test)
    
    # Loop over all event types in event_interests (a list of event outcomes)
    for i in range(0, event_interests.max()):
        event_interest = i + 1
        cif = pd.DataFrame(cif_total[i], model.duration_index)
        ev = EvalSurv(1-cif, y_test[0], y_test[1] == event_interest, censor_surv='km')
        concordance_indices[f"Event_{event_interest}"] = ev.concordance_td()
        brier_series[f"Event_{event_interest}"] = ev.brier_score(time_grid)
        integrated_brier_scores[f"Event_{event_interest}"] = ev.integrated_brier_score(time_grid)
        neg_log_likelihoods[f"Event_{event_interest}"] = ev.integrated_nbll(time_grid)

        # Nam and D'Agostino Chi2 statistic for calibration
        for time in time_grid:
            chi2_stat, p_value, observed_events, expected_events, n, prob_df = nam_dagostino_chi2(
                df=df_test, 
                duration_col=duration_col, 
                event_col=event_col,
                surv=(1-cif), 
                time=time, 
                event_focus=event_interest
            )
            nam_dagostino_results.append({
                'Event': event_interest,
                'Year': round(time / 365),
                'Chi2_Stat': chi2_stat,
                'P_Value': p_value,
                'Observed_Events': observed_events.tolist(),
                'Expected_Events': expected_events.tolist(),
                'Sample_Size': n.tolist()
            })

    # Convert Nam and D'Agostino results to a DataFrame
    nam_dagostino_results_df = pd.DataFrame(nam_dagostino_results)

    # Print results
    for i in range(0, event_interests.max()):
        event_interest = i + 1
        print(f"Concordance index Event {event_interest}: {concordance_indices[f'Event_{event_interest}']} \t Brier's score Event {event_interest}: {integrated_brier_scores[f'Event_{event_interest}']} \t Negative Log Likelihood Event {event_interest}: {neg_log_likelihoods[f'Event_{event_interest}']}")

    print(f"Nam and D'Agostino Chi2 statistic:")
    print(nam_dagostino_results_df)

    return {
        'Concordance_Indices': concordance_indices,
        'Integrated_Brier_Scores': integrated_brier_scores,
        'Negative_Log_Likelihoods': neg_log_likelihoods,
        'Brier_Series': brier_series,
        'Nam_Dagostino_Results': nam_dagostino_results_df,
        'CIF_All': cif_all
    }
    
def combined_test_model(model, df, feature_col, duration_col, event_col, time_grid, params=None, cluster_col=None, model_type='deepsurv'):
    """
    Evaluate the DeepSurv or DeepHit model using the test dataset.

    Args:
        model (torch.nn.Module): Trained DeepSurv or DeepHit model.
        df (pd.DataFrame): Test dataset.
        feature_col (list): List of feature column names.
        duration_col (str): Column name for duration.
        event_col (str): Column name for event.
        time_grid (list): List of time points to evaluate.
        params (dict, optional): Parameters for LSTM preparation.
        cluster_col (str, optional): Column name for clustering.
        model_type (str): Model type, either 'deepsurv' or 'deephit'.

    Returns:
        dict or tuple: Evaluation results depending on the model type.
    """
    df_test = df.copy()

    if model_type == 'deepsurv':
        event_interest = 1
        df_test = df_event_focus(df=df_test, event_col=event_col, event_focus=event_interest)
        X_test, y_test = prep_tensor(df_test, feature_col=feature_col, duration_col=duration_col, event_col=event_col)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        model.compute_baseline_hazards()
        _, time_grid = get_baseline_hazard_at_timepoints(model.compute_baseline_hazards(), time_grid)

        surv = model.predict_surv_df(X_test_tensor)
        ev = EvalSurv(surv, y_test[0], y_test[1], censor_surv='km')
        concordance_index = ev.concordance_td()
        brier_series = ev.brier_score(time_grid)
        integrated_brier_score = ev.integrated_brier_score(time_grid)
        neg_log_likelihood = ev.integrated_nbll(time_grid)
        aic = calculate_aic(model, X_test, y_test)

        results = []
        for time in time_grid:
            chi2_stat, p_value, observed_events, expected_events, n, prob_df = nam_dagostino_chi2(
                df=df_test, duration_col=duration_col, event_col=event_col, surv=surv, time=time, event_focus=event_interest)
            results.append({
                'Year': round(time / 365),
                'Chi2_Stat': chi2_stat,
                'P_Value': p_value,
                'Observed_Events': observed_events.tolist(),
                'Expected_Events': expected_events.tolist(),
                'Sample_Size': n.tolist()
            })

        results_df = pd.DataFrame(results)
        print(f"AIC: {aic} \t concordance index: {concordance_index} \t Brier's score: {integrated_brier_score} \t Negative Log Likelihood: {neg_log_likelihood}")
        print(f"Nam and D'Agostino Chi2 statistic:")
        print(results_df)
        return aic, concordance_index, integrated_brier_score, neg_log_likelihood, brier_series, results

    elif model_type == 'deephit':
        event_interests = np.unique(df_test[event_col].values.astype('int64'))

        concordance_indices = {}
        integrated_brier_scores = {}
        neg_log_likelihoods = {}
        brier_series = {}
        nam_dagostino_results = []

        if params is None:
            X_test, y_test = dh_dataset_loader(df_test, duration_col, event_col, feature_col, time_grid=time_grid)
        else:
            X_test, y_test = dh_lstm_prepare_validation_data(df_test, feature_col, duration_col, event_col, params, cluster_col, time_grid=time_grid)

        cif_total = model.predict_cif(X_test)

        for i in range(event_interests.max()):
            event_interest = i + 1
            cif = pd.DataFrame(cif_total[i], model.duration_index)
            ev = EvalSurv(1 - cif, y_test[0], y_test[1] == event_interest, censor_surv='km')
            concordance_indices[f"Event_{event_interest}"] = ev.concordance_td()
            brier_series[f"Event_{event_interest}"] = ev.brier_score(time_grid)
            integrated_brier_scores[f"Event_{event_interest}"] = ev.integrated_brier_score(time_grid)
            neg_log_likelihoods[f"Event_{event_interest}"] = ev.integrated_nbll(time_grid)

            for time in time_grid:
                chi2_stat, p_value, observed_events, expected_events, n, prob_df = nam_dagostino_chi2(
                    df=df_test, duration_col=duration_col, event_col=event_col, surv=(1 - cif), time=time, event_focus=event_interest)
                nam_dagostino_results.append({
                    'Event': event_interest,
                    'Year': round(time / 365),
                    'Chi2_Stat': chi2_stat,
                    'P_Value': p_value,
                    'Observed_Events': observed_events.tolist(),
                    'Expected_Events': expected_events.tolist(),
                    'Sample_Size': n.tolist()
                })

        nam_dagostino_results_df = pd.DataFrame(nam_dagostino_results)

        for i in range(event_interests.max()):
            event_interest = i + 1
            print(f"Concordance index Event {event_interest}: {concordance_indices[f'Event_{event_interest}']} \t Brier's score Event {event_interest}: {integrated_brier_scores[f'Event_{event_interest}']} \t Negative Log Likelihood Event {event_interest}: {neg_log_likelihoods[f'Event_{event_interest}']}")

        print(f"Nam and D'Agostino Chi2 statistic:")
        print(nam_dagostino_results_df)

        return {
            'Concordance_Indices': concordance_indices,
            'Integrated_Brier_Scores': integrated_brier_scores,
            'Negative_Log_Likelihoods': neg_log_likelihoods,
            'Brier_Series': brier_series,
            'Nam_Dagostino_Results': nam_dagostino_results_df,
            'CIF_All': []
        }
    else:
        raise ValueError("Invalid model type. Please choose either 'deepsurv' or 'deephit'.")