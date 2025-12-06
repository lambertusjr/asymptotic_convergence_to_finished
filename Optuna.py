import optuna
from optuna.trial import Trial
import numpy as np
from models import ModelWrapper
from helper_functions import _get_model_instance, balanced_class_weights, check_study_existence, run_trial_with_cleanup
from utilities import FocalLoss
from training_funcs import train_and_validate
import pandas as pd
import os
from datetime import datetime

models = ['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN']

DEFAULT_EARLY_STOP = {
    "patience": 20,
    "min_delta": 1e-3,
}
EARLY_STOP_LOGGING = False

def _early_stop_args_from(source: dict) -> dict:
    """Build early stopping kwargs, falling back to defaults when keys are absent."""
    return {
        "patience": source.get("early_stop_patience", DEFAULT_EARLY_STOP["patience"]),
        "min_delta": source.get("early_stop_min_delta", DEFAULT_EARLY_STOP["min_delta"]),
        "log_early_stop": EARLY_STOP_LOGGING,
    }

def objective(trial, model, data, train_mask, val_mask, alpha_focal=None, sklearn_data=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # No try/finally with gc.collect here; run_trial_with_cleanup handles it.
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    gamma_focal = trial.suggest_float('gamma_focal', 0.1, 5.0)
    
    # Use pre-calculated alpha_focal if provided, else calculate (fallback)
    if alpha_focal is None:
        alpha_focal = balanced_class_weights(data.y[train_mask]).to(device)
        
    criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
    
    early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
    early_stop_min_delta = trial.suggest_float('early_stop_min_delta', 1e-4, 5e-3, log=True)
    trial_early_stop_args = _early_stop_args_from({
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta
    })
    
    # Pass train_mask to _get_model_instance for XGB leakage prevention
    model_instance = _get_model_instance(trial, model, data, device, train_mask=train_mask)   
    
    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']
    
    if model in sklearn_models:
        num_epochs = 50  
    elif model == "MLP":
        num_epochs = 50
    else: # GNNs
        num_epochs = 400
    
    if model in wrapper_models:
        optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
        model_wrapper.model.to(device)
        best_f1_model_wts, best_f1 = train_and_validate(
            model_wrapper,
            data,
            num_epochs,
            train_mask,
            val_mask,
            **trial_early_stop_args
        )
        return float(best_f1)
    
    elif model in sklearn_models:
        # Use pre-converted sklearn data if available
        if sklearn_data is not None:
            train_x = sklearn_data['train_x']
            train_y = sklearn_data['train_y']
            val_x = sklearn_data['val_x']
            val_y = sklearn_data['val_y']
        else:
            train_x = data.x[train_mask].cpu().numpy()
            train_y = data.y[train_mask].cpu().numpy()
            val_x = data.x[val_mask].cpu().numpy()
            val_y = data.y[val_mask].cpu().numpy()
            
        model_instance.fit(train_x, train_y)
        pred = model_instance.predict(val_x)
        f1_illicit = f1_score(val_y, pred, pos_label=1, average='binary')
        return float(f1_illicit)
        
    else:
        raise ValueError(f"Unknown model: {model}")
            
            

                 
def run_optimisation(models, data, data_for_optimisation, train_mask, val_mask):
    model_parameters = {model_name: [] for model_name in models}
    
    METRIC_KEYS = [
        'accuracy', 'precision', 'precision_illicit', 'recall', 'recall_illicit',
        'f1', 'f1_illicit', 'roc_auc', 'PRAUC', 'kappa'
    ]
    testing_results = {
        model_name: {key: [] for key in METRIC_KEYS} 
        for model_name in models
    }
    
    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']
    
    for model_name in tqdm(models, desc="Models", unit="model"):
        if model_name in wrapper_models:
            n_trials = 100
        else:
            n_trials = 50
        study_name = f'{model_name}_optimization on {data_for_optimisation} dataset'
        db_path = f'sqlite:///optimization_results_on_{data_for_optimisation}.db'
        
        if check_study_existence(model_name, data_for_optimisation): 
            study = optuna.load_study(study_name=study_name, storage=db_path)
        else:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                storage=db_path,
                load_if_exists=True
            )
            with tqdm(total=n_trials, desc=f"{model_name} trials", leave=False, unit="trial") as trial_bar:
                def _optuna_progress_callback(study_inner, trial):
                    trial_bar.update()
                
                # Pre-calculate alpha_focal
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                alpha_focal = balanced_class_weights(data.y[train_mask]).to(device)

                # Pre-calculate sklearn data if needed
                sklearn_data = None
                if model_name in sklearn_models:
                    sklearn_data = {
                        'train_x': data.x[train_mask].cpu().numpy(),
                        'train_y': data.y[train_mask].cpu().numpy(),
                        'val_x': data.x[val_mask].cpu().numpy(),
                        'val_y': data.y[val_mask].cpu().numpy()
                    }

                # Note: data, train_perf_eval, etc., are now the device tensors
                study.optimize(
                    lambda trial: run_trial_with_cleanup( 
                        objective, model_name, trial, model_name, data, train_mask, val_mask, alpha_focal=alpha_focal, sklearn_data=sklearn_data),
                    n_trials=n_trials,
                    callbacks=[_optuna_progress_callback]
                )
        
    return model_parameters

def run_final_evaluation(models, model_parameters, data, data_for_optimisation, train_mask, val_mask, test_mask):
    """
    Runs the final evaluation for all models using the best parameters found.
    """
    import os
    print(f"\nStarting FINAL EVALUATION for {data_for_optimisation} dataset...")
        
    for model_name in models:
        # Check if we have parameters for this model
        if model_name not in model_parameters or not model_parameters[model_name]:
            print(f"No parameters found for {model_name}, skipping evaluation.")
            continue
            
        # Get the best parameters (assuming last one is the best or list of them?)
        # run_optimisation returns a dict where values are lists of params. 
        # Line 155: model_parameters[model_name].append(study.best_params)
        # We should take the latest valid one.
        best_params = model_parameters[model_name][-1]
        
        print(f"Evaluating {model_name} on {data_for_optimisation}...")
        
        # Create directories for results if they don't exist
        os.makedirs(f"results/{data_for_optimisation}/pr_curves", exist_ok=True)
        os.makedirs(f"results/{data_for_optimisation}/metrics", exist_ok=True)
        
        evaluate_model_performance(
            model_name=model_name,
            best_params=best_params,
            data=data,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            dataset_name=data_for_optimisation,
            n_runs=5
        )

def evaluate_model_performance(model_name, best_params, data, train_mask, val_mask, test_mask, dataset_name, n_runs=5):
    """
    Retrains the model with best parameters multiple times (seeds) and saves metrics/PR curves.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']
    
    # Pre-calculate data/weights
    alpha_focal = balanced_class_weights(data.y[train_mask]).to(device)
    sklearn_data = None
    if model_name in sklearn_models:
        sklearn_data = {
            'train_x': data.x[train_mask].cpu().numpy(),
            'train_y': data.y[train_mask].cpu().numpy(),
            'val_x': data.x[val_mask].cpu().numpy(),
            'val_y': data.y[val_mask].cpu().numpy(),
            'test_x': data.x[test_mask].cpu().numpy(),
            'test_y': data.y[test_mask].cpu().numpy()
        }
        
    detailed_metrics = []
    
    # Fixed seeds for reproducibility of the 5 runs
    seeds = [42, 43, 44, 45, 46] 
    
    for i, seed in enumerate(seeds):
        set_seed(seed)
        run_id = f"run_{i+1}"
        print(f"  > Run {i+1}/{n_runs} (Seed {seed})")
        
        # 1. Instantiate Model
        # We need a dummy trial object since _get_model_instance uses trial.suggest_...
        # We'll use optuna.trial.FixedTrial logic or just patch _get_model_instance?
        # Actually _get_model_instance expects a 'trial' object to call suggest on.
        # We can use `optuna.trial.FixedTrial(best_params)`
        
        fixed_trial = optuna.trial.FixedTrial(best_params)
        model_instance = _get_model_instance(fixed_trial, model_name, data, device, train_mask=train_mask)
        
        # 2. Train and Predict
        y_true = data.y[test_mask].cpu().numpy()
        y_pred = None
        y_probs = None
        
        if model_name in wrapper_models:
            # Setup Training
             # Extract training hyperparameters from best_params
            learning_rate = best_params.get('learning_rate', 0.01)
            weight_decay = best_params.get('weight_decay', 5e-4)
            gamma_focal = best_params.get('gamma_focal', 2.0)
            patience = best_params.get('early_stop_patience', 20)
            min_delta = best_params.get('early_stop_min_delta', 1e-3)
            
            criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
            optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
            model_wrapper.model.to(device)
            
            # Determine epochs
            num_epochs = 400 if model_name != "MLP" else 50
            
            # Train
            best_wts, _ = train_and_validate(
                model_wrapper, data, num_epochs, train_mask, val_mask,
                patience=patience, min_delta=min_delta, log_early_stop=False
            )
            model_wrapper.model.load_state_dict(best_wts)
            
            # Predict (Evaluate on Test)
            logits_test = model_wrapper.model(data)[test_mask].detach().cpu()
            probs_test = torch.softmax(logits_test, dim=1)
            y_probs = probs_test.numpy()
            y_pred = torch.argmax(probs_test, dim=1).numpy()
            
        elif model_name in sklearn_models:
            # Sklearn models (SVM, XGB, RF)
            # Already instantiated with best params via FixedTrial
            
            model_instance.fit(sklearn_data['train_x'], sklearn_data['train_y'])
            
            if hasattr(model_instance, "predict_proba"):
                y_probs = model_instance.predict_proba(sklearn_data['test_x'])
            else:
                # SVM might not have predict_proba unless enabled. 
                # SVC(probability=True). logic in _get_model_instance uses SGDClassifier which uses 'loss'
                # SGDClassifier(loss='hinge') -> SVM. No predict_proba. 
                # If loss='log', then logistic regression.
                # If loss='modified_huber', proba available.
                # Standard SVM usage suggests using decision_function.
                # For compatibility, let's use decision function or calibrate.
                # Re-check _get_model_instance for SVM: SGDClassifier(loss='hinge'). This is Linear SVM.
                # Hinge loss doesn't provide probabilities naturally.
                # We can use CalibratedClassifierCV or just use decision_function.
                # BUT user requested PR curve.
                try:
                    y_probs = model_instance.predict_proba(sklearn_data['test_x'])
                except AttributeError:
                    # Fallback: decision_function and sigmoid/minmax? or skip?
                    # SGDClassifier(loss='hinge') does NOT support predict_proba.
                    # We will log a warning and skip PR curve for SVM if not available.
                    y_probs = None
            
            y_pred = model_instance.predict(sklearn_data['test_x'])
            
        # 3. Calculate Metrics
        # If probabilities are missing (SVM-Hinge), some metrics (ROC/PR) will fail/be empty.
        # calculate_metrics expects y_pred_prob.
        
        run_metrics = {}
        if y_probs is not None:
            run_metrics = calculate_metrics(y_true, y_pred, y_probs)
            # 4. Save PR Curve
            precision, recall, thresholds = calculate_pr_metrics_batched(
                torch.tensor(y_probs), torch.tensor(y_true)
            )
            pr_filename = f"results/{dataset_name}/pr_curves/{model_name}_run_{i+1}"
            pr_auc = save_pr_artifacts(precision, recall, thresholds, pr_filename)
        else:
             # Basic metrics without prob-dependent ones
             # Re-implement partial calculation or just wrap
             try:
                run_metrics = calculate_metrics(y_true, y_pred, y_probs) # This might fail
             except:
                from sklearn.metrics import accuracy_score, f1_score
                run_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'f1_illicit': f1_score(y_true, y_pred, pos_label=1, average='binary')
                }
        
        run_metrics['model'] = model_name
        run_metrics['run'] = i + 1
        detailed_metrics.append(run_metrics)
        
    # 5. Save Tables
    df = pd.DataFrame(detailed_metrics)
    
    # Save Detailed
    df.to_csv(f"results/{dataset_name}/metrics/{model_name}_detailed_metrics.csv", index=False)
    
    # Calculate Summary (Mean/Std) - Drop non-numeric columns
    numeric_df = df.drop(columns=['model', 'run'], errors='ignore')
    summary = numeric_df.agg(['mean', 'std']).transpose()
    summary['model'] = model_name
    
    # Append/Write Summary
    summary_file = f"results/{dataset_name}/metrics/summary_metrics.csv"
    if os.path.exists(summary_file):
        summary.to_csv(summary_file, mode='a', header=False)
    else:
        summary.to_csv(summary_file, mode='w', header=True)
        
    print(f"  > Completed. Metrics saved to results/{dataset_name}/metrics/")