#Final restart of my code
from dependencies import *
from utilities import *

print("All dependencies imported successfully.")
print("Starting preparations for hyperparameter tuning...")

seeded_run = True
if seeded_run:
    set_seed(42)
    print("Seed set to 42 for reproducibility.")
else: 
    seed = np.random.SeedSequence().entropy
    

#%% Hyperparameter tuning
from Optuna import *
datasets = ["Elliptic", "IBM_AML_HiSmall", "IBM_AML_LiSmall", "IBM_AML_HiMedium", "IBM_AML_LiMedium", "AMLSim"]
for dataset in datasets:
    match dataset:
        case "Elliptic":
            from pre_processing import EllipticDataset
            data = EllipticDataset(root='/Users/Lambertus/Desktop/Datasets/Elliptic_dataset')[0]
            data_for_optimisation = "Elliptic"
        case "IBM_AML_HiSmall":
            from pre_processing import IBMAMLDataset_HiSmall
            data = IBMAMLDataset_HiSmall(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/HiSmall')[0]
            data_for_optimisation = "IBM_AML_HiSmall"
        case "IBM_AML_LiSmall":
            from pre_processing import IBMAMLDataset_LiSmall
            data = IBMAMLDataset_LiSmall(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/LiSmall')[0]
            data_for_optimisation = "IBM_AML_LiSmall"
        case "IBM_AML_HiMedium":
            from pre_processing import IBMAMLDataset_HiMedium
            data = IBMAMLDataset_HiMedium(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/HiMedium')[0]
            data_for_optimisation = "IBM_AML_HiMedium"
        case "IBM_AML_LiMedium":
            from pre_processing import IBMAMLDataset_LiMedium
            data = IBMAMLDataset_LiMedium(root='/Users/Lambertus/Desktop/Datasets/IBM_AML_dataset/LiMedium')[0]
            data_for_optimisation = "IBM_AML_LiMedium"
        case "AMLSim":
            from pre_processing import AMLSimDataset
            data = AMLSimDataset(root='/Users/Lambertus/Desktop/Datasets/AMLSim_dataset')[0]
            data_for_optimisation = "AMLSim"
    print(f"Dataset {dataset} loaded successfully for hyperparameter tuning.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    print(f"Data moved to device: {device}")
    
    print(f"Starting hyperparameter optimization for {data_for_optimisation} dataset...")
    
    model_parameter = run_optimisation(
        models=['MLP', 'SVM', 'XGB', 'RF', 'GCN', 'GAT', 'GIN'],
        data=data,
        data_for_optimisation=data_for_optimisation
    )