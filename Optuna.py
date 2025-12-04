import optuna
from optuna.trial import Trial
import numpy as np
from models import *
import torch
import torch.nn as nn
from helper_functions import *
from helper_functions import _get_model_instance
from utilities import *
from tqdm import tqdm
from torch.optim import Adam
import gc
from training_funcs import *

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

def objective(trial, model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        
        gamma_focal = trial.suggest_float('gamma_focal', 0.1, 5.0)
        alpha_focal = balanced_class_weights(data.y[data.train_mask]).to(device)
        criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
        
        early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
        early_stop_min_delta = trial.suggest_float('early_stop_min_delta', 1e-4, 5e-3, log=True)
        trial_early_stop_args = _early_stop_args_from({
            "early_stop_patience": early_stop_patience,
            "early_stop_min_delta": early_stop_min_delta
        })
        model_instance = _get_model_instance(trial, model, data, device)   
        
        wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
        sklearn_models = ['SVM', 'XGB', 'RF']
        
        if model in sklearn_models:
            num_epochs = 50  
        elif model == "MLP":
            num_epochs = 50
        else: # GNNs
            num_epochs = 200
        
        if model in wrapper_models:
            optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
            model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
            model_wrapper.model.to(device)
            best_f1_model_wts, best_f1 = train_and_validate(
                model_wrapper,
                data,
                num_epochs,
                **trial_early_stop_args
            )
            return best_f1
        
        elif model in sklearn_models:
            train_x = data.x[data.train_mask].cpu().numpy()
            train_y = data.y[data.train_mask].cpu().numpy()
            val_x = data.x[data.val_mask].cpu().numpy()
            val_y = data.y[data.val_mask].cpu().numpy()
            model_instance.fit(train_x, train_y)
            pred = model_instance.predict(val_x)
            f1_illicit = f1_score(val_y, pred, pos_label=1, average='binary')
            return f1_illicit
        
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            

                 
def run_optimisation(models, data, data_for_optimisation):
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
                def _optuna_progress_callback(study, trial):
                    trial_bar.update()
                
                # Note: data, train_perf_eval, etc., are now the device tensors
                study.optimize(
                    lambda trial: run_trial_with_cleanup( 
                        objective, trial, model_name, data),
                    n_trials=n_trials,
                    callbacks=[_optuna_progress_callback]
                )
        
        model_parameters[model_name].append(study.best_params)
    return model_parameters