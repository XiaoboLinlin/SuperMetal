import gc
import math
import os
import numpy as np
import random
from argparse import Namespace, ArgumentParser, FileType
import torch.nn.functional as F
from functools import partial
import wandb
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataListLoader, DataLoader
from tqdm import tqdm
from datasets.pdbbind import PDBBind, NoiseTransform
from confidence.dataset import ConfidenceDataset
import csv
# torch.multiprocessing.set_sharing_strategy('file_system')

import yaml
from utils.utils import get_model
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from confidence.dataset import get_args
from sklearn.cluster import DBSCAN
from validation_matrix.utils.cluster_centroid import find_centroid
from validation_matrix.utils.find_zinc_pos import find_real_zinc_pos
from utils.nearest_point_dist import get_nearest_point_distances

parser = ArgumentParser()
parser.add_argument('--original_model_dir', type=str, default='workdir', help='Path to folder with trained model and hyperparameters')
parser.add_argument('--confidence_dir', type=str, default='workdir', help='Path to folder with trained confidence model and hyperparameters')
parser.add_argument('--split_test', type=str, default='data/splits/timesplit_test', help='Path of file defining the split')
parser.add_argument('--rmsd_classification_cutoff', type=float, default=2, help='RMSD value below which a prediction is considered a postitive. This can also be multiple cutoffs.')
parser.add_argument('--log_dir', type=str, default='workdir', help='')
parser.add_argument('--batch_size_preprocessing', type=int, default=4, help='Number of workers')
parser.add_argument('--prob_cutoff', type=float, default=0.5)
parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
parser.add_argument('--result_output_dir', type=str, default="validation_matrix/results")
args = parser.parse_args()

def evalulation(args, confidence_args, model, loader, filter=True, use_sigmoid = True):
    model.eval()
    n_centroids, n_real_zinc_pos, n_qualified_real_idx = 0, 0, 0
    results = []
    all_correct_dist = []
    for data in tqdm(loader, total=len(loader)):
        with torch.no_grad():
            pred = model(data)
        # if data[0].name == '6IU5':
        #     print("stop")
        labels = torch.cat([graph.y for graph in data]).to(device)
        labels = labels.flatten()
        if use_sigmoid:
            probabilities = torch.sigmoid(pred)
            pred_labels = (probabilities > args.prob_cutoff).float()
        else:
            pred_labels = (pred > 0).float()
        positions = torch.cat([graph['ligand'].pos for graph in data]).to(device)
        if filter:
            filtered_positions = positions[pred_labels == 1]
            filtered_positions = filtered_positions.cpu().numpy()
        else:
            filtered_positions = positions.cpu().numpy()
        try:
            ## calculating clusters
            # dbscan = DBSCAN(eps=args.rmsd_classification_cutoff, min_samples=2)
            dbscan = DBSCAN(eps=5, min_samples=2)
            clusters = dbscan.fit_predict(filtered_positions)
            centroids = find_centroid(filtered_positions, clusters) + data[0].original_center.numpy()
            real_zinc_pos = find_real_zinc_pos(os.path.join(confidence_args.data_dir, f"{data[0].name}/{data[0].name}_ligands.mol2"))
            centroids_dist_to_nearest_Zn, corresponding_real_zinc_indices = get_nearest_point_distances(centroids, real_zinc_pos)
            qualified_centroids, qualified_real_idx = [], []
            for i, (centroid_dist, idx) in enumerate(zip(centroids_dist_to_nearest_Zn, corresponding_real_zinc_indices)):
                if centroid_dist <= args.rmsd_classification_cutoff:
                    qualified_centroids.append(centroids[i])
                    qualified_real_idx.append(float(idx))
                    all_correct_dist.append(centroid_dist)
                
            accuracy = len(qualified_real_idx) / len(centroids) * 100
            coverage = len(set(qualified_real_idx)) / len(real_zinc_pos) * 100
        except Exception as e:
            centroids = []
            qualified_real_idx = []
            real_zinc_pos = find_real_zinc_pos(os.path.join(confidence_args.data_dir, f"{data[0].name}/{data[0].name}_ligands.mol2"))
            print(f"An error occurred on {data[0].name}", e)
            accuracy = None
            coverage = 0
        n_centroids += len(centroids)
        n_real_zinc_pos += len(real_zinc_pos)
        n_qualified_real_idx += len(set(qualified_real_idx))
        
        confidence_loss = F.binary_cross_entropy_with_logits(pred, labels)
        # predicted_labels = (pred > 0).float()
        
        confidence_classification_accuracy = torch.mean((labels == (pred > 0).float()).float())
        # roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
        try:
            roc_auc = roc_auc_score(labels.detach().cpu().numpy(), pred.detach().cpu().numpy())
        except ValueError as e:
            if 'Only one class present in y_true. ROC AUC score is not defined in that case.' in str(e):
                roc_auc = 0
            else:
                raise e
        results.append({"name": data[0].name, 
                        "accuracy": accuracy, 
                        "coverage": coverage, 
                        "confidence_loss": confidence_loss.cpu().detach().numpy(),
                        "confidence_classification_accuracy": confidence_classification_accuracy.cpu().detach().numpy(),
                        "roc_auc": roc_auc})
        print(f"name: {data[0].name}, accuracy: {accuracy}, coverage: {coverage}")
        
    total_accuracy = n_qualified_real_idx / n_centroids * 100
    total_coverage = n_qualified_real_idx / n_real_zinc_pos * 100
    print("total accuracy: ", total_accuracy, " total coverage: ", total_coverage)
    # Write results to CSV
    csv_file = f"{args.result_output_dir}/results_{args.prob_cutoff}.csv"
    csv_columns = ["name", 
                   "accuracy", 
                   "coverage", 
                   "confidence_loss", 
                   "confidence_classification_accuracy", 
                   "roc_auc"]
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)

    ## save all_correct_dist
    all_correct_dist = np.array(all_correct_dist)
    np.save(f"{args.result_output_dir}/correct_dist_{args.prob_cutoff}.npy", all_correct_dist)
    
def construct_loader_origin(args, score_model_args):
    ## the only difference compared to construct_loader is that we set batch_size = 1
    ## and we used DataLoader not DataLoaderList

    common_args = {'root': score_model_args.data_dir, 
                   'cache_path': score_model_args.cache_path, 
                   'limit_complexes': score_model_args.limit_complexes,
                   'receptor_radius': score_model_args.receptor_radius,
                   'c_alpha_max_neighbors': score_model_args.c_alpha_max_neighbors,
                   'remove_hs': score_model_args.remove_hs, 'max_lig_size': score_model_args.max_lig_size,
                   'popsize': score_model_args.matching_popsize, 'maxiter': score_model_args.matching_maxiter,
                   'num_workers': score_model_args.num_workers, 'all_atoms': score_model_args.all_atoms,
                   'atom_radius': score_model_args.atom_radius, 'atom_max_neighbors': score_model_args.atom_max_neighbors,
                   'esm_embeddings_path': score_model_args.esm_embeddings_path}    

    test_dataset = PDBBind(split_path=args.split_test, keep_original=True, **common_args)
    loader_class = DataLoader
    test_loader = loader_class(dataset=test_dataset, batch_size=args.batch_size_preprocessing, num_workers=args.num_workers, shuffle=False, pin_memory=score_model_args.pin_memory)
    return test_loader

def construct_loader_confidence(args, confidence_args, score_model_args, device):
    common_args = {'cache_path': confidence_args.cache_path, 'original_model_dir': confidence_args.original_model_dir, 'device': device,
                   'inference_steps': confidence_args.inference_steps, 'samples_per_complex': confidence_args.samples_per_complex,
                   'limit_complexes': confidence_args.limit_complexes, 'all_atoms': confidence_args.all_atoms, 'balance': confidence_args.balance,
                   'rmsd_classification_cutoff': confidence_args.rmsd_classification_cutoff, 'use_original_model_cache': confidence_args.use_original_model_cache,
                   'cache_creation_id': confidence_args.cache_creation_id, "cache_ids_to_combine": confidence_args.cache_ids_to_combine,
                   "model_ckpt": confidence_args.ckpt}
    loader_class = DataListLoader if torch.cuda.is_available() else DataLoader
    test_loader = construct_loader_origin(args, score_model_args)
    test_dataset = ConfidenceDataset(loader=test_loader, split=os.path.splitext(os.path.basename(args.split_test))[0], args=confidence_args, **common_args)
    test_loader = loader_class(dataset=test_dataset, batch_size=args.batch_size_preprocessing, shuffle=True)
    return test_loader

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    

if __name__ == '__main__':
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(f'{args.original_model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))
    with open(f'{args.confidence_dir}/model_parameters.yml') as f:
        confidence_args = Namespace(**yaml.full_load(f))

    # construct loader
    test_loader = construct_loader_confidence(args, confidence_args, score_model_args, device)
    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)
    # model = get_model(score_model_args if args.transfer_weights else args, device, t_to_sigma=None, confidence_mode=True)
    model = get_model(confidence_args, device, t_to_sigma=None, confidence_mode=True)

    # Load state_dict
    state_dict = torch.load(f'{args.confidence_dir}/best_model.pt', map_location='cpu')
    # Adjust for DataParallel wrapping
    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    # model.load_state_dict(state_dict)
    
    numel = sum([p.numel() for p in model.parameters()])
    print('Loading trained confidence model with', numel, 'parameters')

    args.device = device
    # evalulation(args, model, test_loader, run_dir)
    evalulation(args, confidence_args, model, test_loader)
