import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()
        

            
        # device_type = 'cuda' if logits.is_cuda else 'cpu'
        with torch.amp.autocast('cuda', enabled=False):
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # prob of the true class
        
        # Use log_softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # More stable computation of pt
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0-1e-7)  # Prevent underflow/overflow
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha correctly depending on type
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
def extract_data_information(data):
    # Extracts masks from data object and recreates new data object to ensure no unnecessary attributes are included
    
    #Extract masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y = data.y
    x = data.x
    edge_index = data.edge_index
    del data
    #Recreate data object
    new_data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return new_data, train_mask, val_mask, test_mask