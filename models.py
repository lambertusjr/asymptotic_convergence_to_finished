import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from helper_functions import *
from sklearn.metrics import f1_score

#%% Modelwrapper

class ModelWrapper:
    def __init__(self, model, optimiser, criterion):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion

    def train_step(self, data):
        self.model.train()
        self.optimiser.zero_grad()
        out = self.model(data)[data.train_mask]
        loss = self.criterion(out, data.y[data.train_mask])
        loss.backward()
        self.optimiser.step()
        return loss.item()
    
    def evaluate(self, data, mask, full_metrics=False):
        self.model.eval()
        
        with torch.no_grad():
            # 1. Forward Pass (GPU)
            out = self.model(data)
            
            # 2. Slice specific nodes (GPU)
            # Note: We slice 'out' directly to avoid creating a second full-size tensor
            out_subset = out[mask]
            labels_subset = data.y[mask]
            
            loss = self.criterion(out_subset, labels_subset)

            logits_cpu = out_subset.detach().cpu()
            labels_cpu = labels_subset.detach().cpu()
            
            probs_cpu = F.softmax(logits_cpu, dim=1)
            preds_cpu = torch.argmax(probs_cpu, dim=1)
            
            if not full_metrics:
                f1_illicit = f1_score(labels_cpu.numpy(), preds_cpu.numpy(), pos_label=1, average='binary')
                return loss.item(), f1_illicit
            

            all_metrics = calculate_metrics(labels_cpu, preds_cpu, probs_cpu)
            
            prec, rec, thresh = calculate_pr_metrics_batched(probs_cpu, labels_cpu)
            pr_auc = save_pr_artifacts(prec, rec, thresh, "temp_pr_curve")
            
            all_metrics['PRAUC'] = pr_auc
            
            return loss.item(), all_metrics


            

#%% GCN
class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network model.
    """
    def __init__(self, num_node_features, num_classes, hidden_units, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_units)
        self.conv2 = GCNConv(hidden_units, num_classes)
        self.dropout = dropout

    def forward(self, data):
        # x: Node features [num_nodes, in_channels]
        # edge_index: Graph connectivity [2, num_edges]
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Output raw logits
        x = self.conv2(x, edge_index)
        return x

#%% GAT
class GAT(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, num_heads, dropout_1=0.6, dropout_2=0.5):
        super(GAT, self).__init__()
        # Keep the total latent size roughly equal to hidden_units while limiting per-head width
        per_head_dim = max(1, math.ceil(hidden_units / num_heads))
        total_hidden = per_head_dim * num_heads
        self.conv1 = GATConv(num_node_features, per_head_dim, heads=num_heads, dropout=dropout_1, add_self_loops=False)
        self.conv2 = GATConv(total_hidden, num_classes, heads=1, concat=False, dropout=dropout_2, add_self_loops=False)

    def forward(self, data):
        x = data.x
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x, inplace=True)
        x = self.conv2(x, edge_index)
        return x
    
#%% GIN
class GIN(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units):
        super(GIN, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(num_node_features, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_units, hidden_units)
        )
        self.conv2 = GINConv(nn2)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self, data):
        x, batch = data.x, data.batch
        edge_index = data.adj_t if hasattr(data, 'adj_t') else data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.fc(x)
        return x

#%% MLP
class MLP(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_units, dropout_1=0.6, dropout_2=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(num_node_features, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, num_classes)
        self.dropout_1 = dropout_1
        self.dropout_2 = dropout_2

    def forward(self, data):
        x = data.x  # only use node features, no graph structure
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_1, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout_2, training=self.training)
        x = self.fc3(x)
        return x