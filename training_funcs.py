



def train_and_validate(
    model_wrapper,
    data,
    num_epochs,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    
    mdl_dev = next(model_wrapper.model.parameters()).device
    if not (data.x.device == mdl_dev and data.train_mask.device == mdl_dev and data.val_mask.device == mdl_dev):
        raise RuntimeError(
            f"Device mismatch: model={mdl_dev}, data.x={data.x.device}, "
            f"train_mask={data.train_mask.device}, val_mask={data.val_mask.device}"
        )
        
    metrics = {
        'accuracy': [],
        'precision_weighted': [],
        'precision_illicit': [],
        'recall': [],
        'recall_illicit': [],
        'f1': [],
        'f1_illicit': [],
        'roc_auc': [],
        'PRAUC': [],
        'kappa': [] 
    }
    
    best_f1 = -1
    epochs_without_improvement = 0
    best_epoch = -1
    
    for epoch in range(num_epochs):
        train_loss = model_wrapper.train_step(data)
        
        val_loss, f1_illicit = model_wrapper.evaluate(data, data.val_mask, full_metrics=False)
        
        current_f1 = f1_illicit
        
        improved = current_f1 > (best_f1 + min_delta)
        if improved:
            best_f1, best_f1_model_wts = update_best_weights(model_wrapper.model, best_f1, current_f1, best_f1_model_wts)
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if patience and epochs_without_improvement >= patience:
            if log_early_stop:
                print(f"Early stopping at epoch {epoch+1}. Best F1-illicit: {best_f1:.4f} at epoch {best_epoch+1}.")
            break
    
    return best_f1_model_wts, best_f1
        
        
import copy
def update_best_weights(model, best_f1, current_f1, best_f1_model_wts=None):
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_f1_model_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_f1, best_f1_model_wts