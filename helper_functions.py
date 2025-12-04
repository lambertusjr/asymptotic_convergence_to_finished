import torch
import numpy as np
from torchmetrics.classification import BinaryPrecisionRecallCurve
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score, auc
from torchmetrics.classification import BinaryAveragePrecision
import matplotlib.pyplot as plt
metric = BinaryAveragePrecision().to('cuda') if torch.cuda.is_available() else BinaryAveragePrecision()
from models import *
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import gc
from contextlib import contextmanager
def calculate_pr_metrics_batched(probs, labels, chunk_size=10000):
    # 1. Initialize metric on CPU to save GPU memory
    #    thresholds=None calculates exact curve (uses more RAM)
    #    thresholds=1000 uses fixed bins (uses constant minimal RAM)
    pr_curve = BinaryPrecisionRecallCurve(thresholds=None).cpu()

    n_samples = probs.size(0)

    # 2. Iterate manually in chunks (simulating batches)
    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            
            # SLICE AND MOVE TO CPU IMMEDIATELY
            # Essential: .detach() breaks the graph, .cpu() frees VRAM
            prob_chunk = probs[i:end].detach().cpu()
            label_chunk = labels[i:end].detach().cpu()
            
            # Accumulate stats
            pr_curve.update(prob_chunk, label_chunk)

    # 3. Compute final metric (on CPU)
    #    returns precision, recall, thresholds
    precision, recall, thresholds = pr_curve.compute()
    
    # Optional: Plotting
    # fig, ax = pr_curve.plot(score=True) 
    
    return precision, recall, thresholds

# Usage Example:
# precision, recall, _ = calculate_pr_metrics_batched(out, data.y)

def save_pr_artifacts(precision, recall, thresholds, filename_prefix):
    """
    Takes PRE-CALCULATED metrics and saves them to disk.
    Does NOT perform calculation.
    """
    # 1. Calculate AUC (Cheap)
    pr_auc = auc(recall.numpy(), precision.numpy())
    
    # 2. Save Raw Data
    np.savez_compressed(
        f"{filename_prefix}_pr_data.npz",
        precision=precision.numpy(),
        recall=recall.numpy(),
        thresholds=thresholds.numpy(),
        auc=pr_auc
    )

    # 3. Save Image
    try:
        fig, ax = plt.subplots()
        ax.plot(recall.numpy(), precision.numpy(), label=f'PRAUC = {pr_auc:.4f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        fig.savefig(f"{filename_prefix}_pr_curve.png", dpi=300)
    finally:
        plt.close(fig) # Prevent memory leak
        
    return pr_auc

def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) # illicit is class 1
    recall = recall_score(y_true, y_pred, average='weighted')
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    
    roc_auc = roc_auc_score(y_true, y_pred_prob[:,1])  # assuming class 1 is the positive class
    
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_illicit': precision_illicit,
        'recall': recall,
        'recall_illicit': recall_illicit,
        'f1': f1,
        'f1_illicit': f1_illicit,
        'roc_auc': roc_auc,
        'kappa': kappa,
    }
    
    return metrics

def balanced_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (sum to 1) for 1-D integer labels.

    Unlabelled entries (label < 0) are ignored.
    """
    if labels.ndim != 1:
        labels = labels.view(-1)
    labels = labels.detach()
    valid = labels >= 0
    if not torch.any(valid):
        return torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    filtered = labels[valid].to(torch.long).cpu()
    counts = torch.bincount(filtered, minlength=num_classes).clamp_min(1)
    inv = (1.0 / counts.float())
    inv = inv / inv.sum()
    return inv

def _get_model_instance(trial, model, data, device):
    """
    Helper function to suggest hyperparameters and instantiate a model
    based on the model name.
    """
    if model == 'MLP':
        from models import MLP
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        return MLP(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units, dropout_1=dropout_1, dropout_2=dropout_2)
    
    elif model == 'SVM':
        from sklearn.svm import SVC
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        degree = trial.suggest_int('degree', 2, 5)
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        return SGDClassifier(
            loss='hinge',
            alpha=1.0 / (C * data.num_nodes),  # Convert C to alpha
            penalty='l2',
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )

    elif model == 'XGB':
        max_depth = trial.suggest_int('max_depth', 5, 15)
        Gamma_XGB = trial.suggest_float('Gamma_XGB', 0, 5)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        learning_rate_XGB = trial.suggest_float('learning_rate_XGB', 0.005, 0.05, log=False) # XGB learning rate
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        return XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=calculate_scale_pos_weight(data),
            learning_rate=learning_rate_XGB,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            gamma=Gamma_XGB,
            subsample=subsample,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )

    elif model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')

    elif model == 'GCN':
        from models import GCN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.7)
        return GCN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, dropout=dropout)

    elif model == 'GAT':
        from models import GAT
        hidden_units = trial.suggest_int('hidden_units', 32, 128)
        num_heads = trial.suggest_int('num_heads', 1, 8)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        return GAT(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, num_heads=num_heads, dropout_1=dropout_1, dropout_2=dropout_2)

    elif model == 'GIN':
        from models import GIN
        hidden_units = trial.suggest_int('hidden_units', 32, 160)
        return GIN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units)

    else:
        raise ValueError(f"Unknown model: {model}")
    
def calculate_scale_pos_weight(data):
    """
    Calculate the scale_pos_weight for imbalanced datasets.
    """
    #train_mask = data[train_mask]
    y_train = data.y.cpu().numpy()
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg) / float(pos)

def check_study_existence(model_name, data_for_optimization):
    """
    Check if an Optuna study exists and has a sufficient number of trials (>= 50).
    
    If a study exists but has fewer than 50 trials, it is automatically
    deleted.
    
    Parameters
    ----------
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    data_for_optimization : str
        Name of the dataset used for optimization.
        
    Returns
    -------
    exists : bool
        True if a study exists with >= 50 trials, False otherwise.
    """
    import optuna
    study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
    storage_url = 'sqlite:///optimization_results.db'
    
    try:
        # 1. Attempt to load the study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # 2. Study exists, check the number of trials (runs)
        num_trials = len(study.trials)
        
        if num_trials < 50:
            # 3. Less than 50 runs: wipe the study and return False
            print(f"Study '{study_name}' found with only {num_trials} trials (< 50). Deleting study.")
            optuna.delete_study(study_name=study_name, storage=storage_url)
            return False
        else:
            # 4. 50 or more runs: study is valid, return True
            print(f"Study '{study_name}' found with {num_trials} trials (>= 50). Study is valid.")
            return True
            
    except KeyError:
        # 5. Study does not exist: return False
        print(f"Study '{study_name}' not found.")
        return False
    
def run_trial_with_cleanup(trial_func, model_name, *args, **kwargs):
    """
    Runs a trial function safely with:
      - Automatic no_grad() for CPU-based models.
      - GPU/CPU memory cleanup after each trial.
    
    Parameters
    ----------
    trial_func : callable
        The trial function to run (e.g., objective).
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    *args, **kwargs :
        Arguments to pass to trial_func.
        
    Returns
    -------
    result : Any
        The return value of the trial function.
    """
    try:
        with inference_mode_if_needed(model_name):
            result = trial_func(*args, **kwargs)
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
@contextmanager
def inference_mode_if_needed(model_name: str):
    """
    Context manager that disables gradient tracking if the model is CPU-based
    or if we are in evaluation mode.
    """
    if model_name in ["SVM", "XGB", "RF"]:
        with torch.no_grad():
            yield
    else:
        yield