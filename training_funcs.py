



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
    best_f1_model_wts = None
    
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
import gc
import torch
import torch
import gc

def print_gpu_tensors():
    # Do not force gc.collect() here if you want to see current state including 
    # potential temporaries. Only use it if hunting for "unreachable" leaks.
    
    print(f"{'Shape':<25} {'Type':<15} {'Count':<10} {'Total Mem (MB)':<15}")
    print("-" * 70)
    
    tensor_groups = {}
    total_mem = 0
    
    # Iterate over all objects
    for obj in gc.get_objects():
        try:
            # Check if object is a tensor and on CUDA
            # We use type check first to avoid errors accessing attributes on non-tensors
            if torch.is_tensor(obj) and obj.is_cuda:
                shape = str(tuple(obj.size()))
                dtype = str(obj.dtype).replace('torch.', '')
                key = (shape, dtype)
                
                mem = obj.element_size() * obj.nelement() / (1024 ** 2)
                
                if key in tensor_groups:
                    tensor_groups[key]['count'] += 1
                    tensor_groups[key]['mem'] += mem
                else:
                    tensor_groups[key] = {'count': 1, 'mem': mem}
                
                total_mem += mem
        except Exception:
            # Squelch errors from fragile objects during iteration
            pass
            
    # Sort by memory usage
    sorted_groups = sorted(tensor_groups.items(), key=lambda x: x[1]['mem'], reverse=True)
    
    for (shape, dtype), stats in sorted_groups:
        print(f"{shape:<25} {dtype:<15} {stats['count']:<10} {stats['mem']:.2f}")
        
    print("-" * 70)
    print(f"Total GPU Memory Occupied by Tensors: {total_mem:.2f} MB")
    
    # Compare with reserved memory to see fragmentation/cache overhead
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Total GPU Memory Reserved by PyTorch: {reserved:.2f} MB")