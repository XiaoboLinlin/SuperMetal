"""
SuperMetal Prediction Module
============================

Easy-to-use interface for predicting metal ion binding sites in proteins.

Example usage:
    >>> import supermetal
    >>> results = supermetal.predict("protein.pdb")
    >>> print(results['positions'])  # Predicted metal ion positions
    
    # Or from command line:
    # python predict.py --protein protein.pdb --output predictions/
"""

import os
import copy
import argparse
from functools import partial
from typing import Optional, Union, List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import yaml
from sklearn.cluster import DBSCAN

from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, set_time
from utils.sampling import sampling, randomize_position_new
from utils.inference_utils import InferenceDataset, get_sequences_from_pdbfile, compute_ESM_embeddings
from utils.visualise import ZincMolecule
from utils.nearest_point_dist import get_nearest_point_distances
from validation_matrix.utils.cluster_centroid import find_centroid
from validation_matrix.utils.find_zinc_pos import find_real_zinc_pos
from torch_geometric.loader import DataLoader
from esm import pretrained


# Default model paths
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "workdir/large_all_atoms_model")
DEFAULT_CONFIDENCE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "workdir/large_confidence_model")
HUGGINGFACE_REPO = "scofieldlinlin/SuperMetal"


class Args:
    """Default model arguments loaded from model_parameters.yml"""
    def __init__(self, config_path: Optional[str] = None):
        # Default values
        self.all_atoms = True
        self.atom_max_neighbors = 8
        self.atom_radius = 5
        self.c_alpha_max_neighbors = 24
        self.cross_distance_embed_dim = 64
        self.cross_max_distance = 80
        self.distance_embed_dim = 64
        self.dropout = 0.1
        self.dynamic_max_cross = True
        self.embedding_scale = 1000
        self.embedding_type = "sinusoidal"
        self.esm_embeddings_path = None
        self.inference_steps = 20
        self.max_radius = 5.0
        self.no_batch_norm = False
        self.ns = 40
        self.num_conv_layers = 3
        self.nv = 4
        self.receptor_radius = 15.0
        self.remove_hs = True
        self.scale_by_sigma = True
        self.sigma_embed_dim = 64
        self.tr_sigma_max = 20.0
        self.tr_sigma_min = 0.1
        self.use_second_order_repr = False
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                for key, value in config.items():
                    setattr(self, key, value)


def download_from_huggingface(model_name: str = "score_model") -> Tuple[str, str]:
    """Download model weights and config from HuggingFace Hub.
    
    Args:
        model_name: Either 'score_model' or 'confidence_model'
        
    Returns:
        Tuple of (model_checkpoint_path, config_path)
    """
    try:
        from huggingface_hub import hf_hub_download
        
        cache_dir = os.path.expanduser("~/.cache/supermetal")
        
        # Download checkpoint
        model_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=f"{model_name}/best_model.pt",
            cache_dir=cache_dir
        )
        
        # Download config
        config_path = hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=f"{model_name}/model_parameters.yml",
            cache_dir=cache_dir
        )
        
        return model_path, config_path
    except ImportError:
        raise ImportError("Please install huggingface_hub: pip install huggingface_hub")
    except Exception as e:
        raise RuntimeError(f"Failed to download model from HuggingFace: {e}")


def load_model(model_dir: Optional[str] = None, 
               device: Optional[torch.device] = None,
               use_huggingface: bool = False) -> tuple:
    """Load the SuperMetal model.
    
    Args:
        model_dir: Path to the model directory containing best_model.pt and model_parameters.yml
        device: torch device (auto-detected if None)
        use_huggingface: Download from HuggingFace if True
        
    Returns:
        Tuple of (model, args, t_to_sigma function)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine model path
    if use_huggingface:
        checkpoint_path, config_path = download_from_huggingface("score_model")
    elif model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
        config_path = os.path.join(model_dir, "model_parameters.yml")
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
    else:
        config_path = os.path.join(model_dir, "model_parameters.yml")
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at {checkpoint_path}")
        print("Attempting to download from HuggingFace...")
        checkpoint_path, config_path = download_from_huggingface("score_model")
    
    # Load configuration
    args = Args(config_path)
    
    # Setup t_to_sigma
    t_to_sigma = partial(t_to_sigma_compl, args=args)
    
    # Import model class
    from utils.utils import get_model
    model = get_model(args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, args, t_to_sigma


def load_confidence_model(confidence_model_dir: Optional[str] = None,
                          device: Optional[torch.device] = None,
                          use_huggingface: bool = False) -> tuple:
    """Load the SuperMetal confidence model for filtering predictions.
    
    Args:
        confidence_model_dir: Path to the confidence model directory
        device: torch device (auto-detected if None)
        use_huggingface: Download from HuggingFace if True
        
    Returns:
        Tuple of (confidence_model, confidence_args)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine model path
    if use_huggingface:
        checkpoint_path, config_path = download_from_huggingface("confidence_model")
    elif confidence_model_dir is None:
        confidence_model_dir = DEFAULT_CONFIDENCE_MODEL_DIR
        config_path = os.path.join(confidence_model_dir, "model_parameters.yml")
        checkpoint_path = os.path.join(confidence_model_dir, "best_model.pt")
    else:
        config_path = os.path.join(confidence_model_dir, "model_parameters.yml")
        checkpoint_path = os.path.join(confidence_model_dir, "best_model.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Confidence model not found at {checkpoint_path}")
        print("Attempting to download from HuggingFace...")
        try:
            checkpoint_path, config_path = download_from_huggingface("confidence_model")
        except Exception as e:
            print(f"Failed to download confidence model: {e}")
            return None, None
    
    # Load configuration
    confidence_args = Args(config_path)
    
    # Import model class
    from utils.utils import get_model
    confidence_model = get_model(confidence_args, device, t_to_sigma=None, 
                                  no_parallel=True, confidence_mode=True)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    confidence_model.load_state_dict(state_dict)
    confidence_model.eval()
    
    return confidence_model, confidence_args


def prepare_data(protein_path: str, 
                 args: Args,
                 num_metal_predictions: int = 40,
                 esm_embeddings: Optional[torch.Tensor] = None) -> list:
    """Prepare protein data for inference.
    
    Args:
        protein_path: Path to the protein PDB file
        args: Model arguments
        num_metal_predictions: Number of metal positions to predict
        esm_embeddings: Pre-computed ESM embeddings (computed if None)
        
    Returns:
        List of complex graphs ready for inference
    """
    complex_name = os.path.splitext(os.path.basename(protein_path))[0]
    out_dir = os.path.dirname(protein_path) or "."
    
    # Use a dummy zinc ligand to match expected input format
    ligand_smiles = "[Zn]"  # Single zinc atom
    
    # Compute or use provided ESM embeddings
    precomputed_embeddings = None
    if esm_embeddings is not None:
        precomputed_embeddings = [esm_embeddings]
    
    dataset = InferenceDataset(
        out_dir=out_dir,
        complex_names=[complex_name],
        protein_files=[protein_path],
        ligand_descriptions=[ligand_smiles],
        protein_sequences=[None],
        lm_embeddings=True,  # Compute ESM embeddings
        precomputed_lm_embeddings=precomputed_embeddings,
        receptor_radius=args.receptor_radius,
        c_alpha_max_neighbors=args.c_alpha_max_neighbors,
        remove_hs=args.remove_hs,
        all_atoms=args.all_atoms,
        atom_radius=args.atom_radius,
        atom_max_neighbors=args.atom_max_neighbors
    )
    
    complex_graph = dataset.get(0)
    
    if not complex_graph.get('success', True):
        raise RuntimeError(f"Failed to process protein: {protein_path}")
    
    return [complex_graph]


def predict(protein_path: Union[str, List[str]], 
            model_dir: Optional[str] = None,
            confidence_model_dir: Optional[str] = None,
            num_metal_predictions: int = 100,
            inference_steps: int = 20,
            output_dir: Optional[str] = None,
            save_pdb: bool = True,
            device: Optional[torch.device] = None,
            use_huggingface: bool = False,
            use_confidence: bool = True,
            confidence_threshold: float = 0.5,
            cluster_eps: float = 5.0,
            ground_truth_ligand_path: Optional[str] = None) -> Dict[str, Any]:
    """Predict metal ion binding sites in a protein structure.
    
    Pipeline:
    1. Generate num_metal_predictions positions using diffusion model
    2. Filter using confidence model (if use_confidence=True)
    3. Cluster nearby predictions using DBSCAN
    4. Return cluster centroids as final predictions
    
    Args:
        protein_path: Path to protein PDB file or list of paths
        model_dir: Path to score model directory (uses default if None)
        confidence_model_dir: Path to confidence model directory (uses default if None)
        num_metal_predictions: Number of initial metal positions to generate (default: 100)
        inference_steps: Number of diffusion steps (default: 20)
        output_dir: Directory to save output files (default: same as input)
        save_pdb: Whether to save predicted positions as PDB file
        device: Computation device (auto-detected if None)
        use_huggingface: Download model from HuggingFace if True
        use_confidence: Use confidence model to filter predictions (default: True)
        confidence_threshold: Confidence threshold for filtering (default: 0.5)
        cluster_eps: DBSCAN epsilon for clustering (default: 5.0 Angstroms)
        ground_truth_ligand_path: Path to ligand mol2 file for evaluation (optional)
        
    Returns:
        Dictionary containing:
            - 'name': complex name
            - 'raw_positions': all generated metal coordinates before filtering (N x 3)
            - 'filtered_positions': positions after confidence filtering (M x 3)
            - 'cluster_centroids': final clustered positions (K x 3)
            - 'confidences': confidence scores for each position (if use_confidence)
            - 'pdb_path': path to output PDB file (if save_pdb=True)
            - 'metrics': evaluation metrics (if ground_truth_ligand_path provided)
            
    Example:
        >>> results = predict("1ABC.pdb")
        >>> print(f"Found {len(results['cluster_centroids'])} metal binding sites")
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle single file or list
    if isinstance(protein_path, str):
        protein_paths = [protein_path]
    else:
        protein_paths = protein_path
    
    # Load score model
    print("Loading SuperMetal score model...")
    model, args, t_to_sigma = load_model(model_dir, device, use_huggingface)
    args.inference_steps = inference_steps
    
    # Load confidence model
    confidence_model, confidence_args = None, None
    if use_confidence:
        print("Loading confidence model...")
        confidence_model, confidence_args = load_confidence_model(
            confidence_model_dir, device, use_huggingface
        )
        if confidence_model is None:
            print("Warning: Confidence model not found, skipping confidence filtering")
            use_confidence = False
    
    results = []
    
    for pdb_path in protein_paths:
        print(f"\nProcessing: {pdb_path}")
        
        # Prepare data
        complex_graphs = prepare_data(pdb_path, args, num_metal_predictions)
        original_graph = copy.deepcopy(complex_graphs[0])
        
        # Setup for sampling
        from utils.diffusion_utils import get_t_schedule
        from utils.sampling import randomize_position_multiple
        t_schedule = get_t_schedule(inference_steps=args.inference_steps)
        
        # Randomize initial positions (100 metal positions)
        data_list = [copy.deepcopy(g) for g in complex_graphs]
        randomize_position_multiple(data_list, False, args.tr_sigma_max, metal_num=num_metal_predictions)
        
        # Run diffusion sampling
        print(f"Running diffusion with {num_metal_predictions} initial positions...")
        from utils.sampling import sampling_test1
        predictions_list, _ = sampling_test1(
            data_list=data_list,
            model=model,
            inference_steps=args.inference_steps,
            tr_schedule=t_schedule,
            device=device,
            t_to_sigma=t_to_sigma,
            model_args=args
        )
        
        # Extract predicted positions
        pred_graph = predictions_list[0]
        pred_positions = pred_graph['ligand'].pos.cpu().numpy()
        original_center = pred_graph.original_center.cpu().numpy()
        
        # Convert to original coordinate system
        raw_positions = pred_positions + original_center
        
        result = {
            'name': complex_graphs[0]['name'],
            'raw_positions': raw_positions,
            'filtered_positions': raw_positions,
            'cluster_centroids': None,
            'confidences': None,
            'pdb_path': None,
            'metrics': None
        }
        
        # Apply confidence filtering
        if use_confidence and confidence_model is not None:
            print("Applying confidence filtering...")
            # Prepare graph for confidence model
            conf_graph = copy.deepcopy(original_graph)
            conf_graph['ligand'].x = conf_graph['ligand'].x[-1].repeat(num_metal_predictions, 1)
            conf_graph['ligand'].pos = torch.from_numpy(pred_positions).float()
            conf_graph['ligand'].node_t = {'tr': torch.zeros(num_metal_predictions)}
            conf_graph['ligand'].batch = torch.zeros(num_metal_predictions, dtype=torch.long)
            conf_graph['receptor'].node_t = {'tr': torch.zeros(conf_graph['receptor'].num_nodes)}
            conf_graph['receptor'].batch = torch.zeros(conf_graph['receptor'].num_nodes, dtype=torch.long)
            if args.all_atoms:
                conf_graph['atom'].node_t = {'tr': torch.zeros(conf_graph['atom'].num_nodes)}
                conf_graph['atom'].batch = torch.zeros(conf_graph['atom'].num_nodes, dtype=torch.long)
            conf_graph.complex_t = {'tr': torch.zeros(1)}
            
            # Run confidence model
            conf_graph = conf_graph.to(device)
            with torch.no_grad():
                confidences = confidence_model(conf_graph)
                confidences = torch.sigmoid(confidences).cpu().numpy().flatten()
            
            result['confidences'] = confidences
            
            # Filter by confidence threshold
            mask = confidences > confidence_threshold
            filtered_positions = raw_positions[mask]
            result['filtered_positions'] = filtered_positions
            print(f"Confidence filtering: {len(raw_positions)} -> {len(filtered_positions)} positions")
        else:
            filtered_positions = raw_positions
        
        # Apply DBSCAN clustering
        if len(filtered_positions) > 0:
            print(f"Clustering with DBSCAN (eps={cluster_eps})...")
            dbscan = DBSCAN(eps=cluster_eps, min_samples=2)
            clusters = dbscan.fit_predict(filtered_positions)
            centroids = find_centroid(filtered_positions, clusters)
            
            if len(centroids) == 0:
                # If no clusters found, use all filtered positions
                centroids = filtered_positions
                print(f"No clusters found, using {len(centroids)} individual positions")
            else:
                print(f"Found {len(centroids)} cluster centroids")
            
            result['cluster_centroids'] = centroids
        else:
            result['cluster_centroids'] = np.array([])
            print("Warning: No positions after filtering")
        
        # Evaluate against ground truth if provided
        if ground_truth_ligand_path:
            try:
                real_zinc_pos = find_real_zinc_pos(ground_truth_ligand_path)
                if len(result['cluster_centroids']) > 0 and len(real_zinc_pos) > 0:
                    distances, indices = get_nearest_point_distances(
                        result['cluster_centroids'], real_zinc_pos
                    )
                    
                    # Calculate metrics
                    n_correct_2A = np.sum(distances < 2.0)
                    n_correct_5A = np.sum(distances < 5.0)
                    unique_matched = len(set(indices[distances < 5.0]))
                    
                    result['metrics'] = {
                        'ground_truth_count': len(real_zinc_pos),
                        'predicted_count': len(result['cluster_centroids']),
                        'correct_within_2A': int(n_correct_2A),
                        'correct_within_5A': int(n_correct_5A),
                        'coverage': unique_matched / len(real_zinc_pos) * 100,
                        'precision': n_correct_5A / len(result['cluster_centroids']) * 100 if len(result['cluster_centroids']) > 0 else 0,
                        'distances': distances.tolist()
                    }
                    print(f"\nEvaluation results:")
                    print(f"  Ground truth: {len(real_zinc_pos)} zinc sites")
                    print(f"  Predicted: {len(result['cluster_centroids'])} sites")
                    print(f"  Correct (<5Å): {n_correct_5A}")
                    print(f"  Coverage: {result['metrics']['coverage']:.1f}%")
                    print(f"  Precision: {result['metrics']['precision']:.1f}%")
            except Exception as e:
                print(f"Warning: Could not evaluate against ground truth: {e}")
        
        # Save PDB if requested
        if save_pdb and len(result['cluster_centroids']) > 0:
            if output_dir is None:
                output_dir = os.path.dirname(pdb_path) or "."
            os.makedirs(output_dir, exist_ok=True)
            
            # Save zinc-only predictions
            output_pdb = os.path.join(output_dir, f"{result['name']}_metal_predictions.pdb")
            save_predictions_to_pdb(result['cluster_centroids'], output_pdb)
            result['pdb_path'] = output_pdb
            
            # Save combined protein + zinc for PyMol visualization
            combined_pdb = os.path.join(output_dir, f"{result['name']}_combined.pdb")
            save_combined_pdb(pdb_path, result['cluster_centroids'], combined_pdb)
            result['combined_pdb_path'] = combined_pdb
            
            print(f"Saved predictions to: {output_pdb}")
            print(f"Saved combined PDB (for PyMol): {combined_pdb}")
        
        results.append(result)
    
    # Return single result if single input
    if len(results) == 1:
        return results[0]
    return results


def save_predictions_to_pdb(positions: np.ndarray, 
                            output_path: str,
                            element: str = "ZN") -> None:
    """Save predicted metal positions to a PDB file.
    
    Args:
        positions: Nx3 array of metal ion coordinates
        output_path: Path for output PDB file
        element: Element symbol (default: ZN for zinc)
    """
    with open(output_path, 'w') as f:
        f.write(f"REMARK   SuperMetal predicted metal binding sites\n")
        f.write(f"REMARK   Number of predictions: {len(positions)}\n")
        for i, pos in enumerate(positions):
            # PDB HETATM format
            f.write(f"HETATM{i+1:5d}  {element:2s}  {element:3s} A{i+1:4d}    "
                   f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                   f"  1.00  0.00          {element:>2s}\n")
        f.write("END\n")


def save_combined_pdb(protein_pdb_path: str,
                      positions: np.ndarray,
                      output_path: str,
                      element: str = "ZN") -> None:
    """Save combined PDB with protein structure and predicted metal positions.
    
    This creates a single PDB file that can be easily visualized in PyMol.
    The predicted metals are added as HETATM records at the end of the protein.
    
    Args:
        protein_pdb_path: Path to original protein PDB file
        positions: Nx3 array of predicted metal coordinates
        output_path: Path for output combined PDB file
        element: Element symbol (default: ZN for zinc)
    """
    with open(output_path, 'w') as f:
        f.write(f"REMARK   SuperMetal prediction results\n")
        f.write(f"REMARK   Number of predicted {element} sites: {len(positions)}\n")
        f.write(f"REMARK   Original protein: {os.path.basename(protein_pdb_path)}\n")
        f.write(f"REMARK\n")
        
        # Read and write original protein (skip END line)
        atom_count = 0
        with open(protein_pdb_path, 'r') as prot_f:
            for line in prot_f:
                if line.startswith('END'):
                    continue
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_count += 1
                f.write(line)
        
        # Add predicted metal positions
        f.write(f"REMARK   Predicted {element} binding sites below\n")
        for i, pos in enumerate(positions):
            atom_num = atom_count + i + 1
            # HETATM format for predicted zinc
            f.write(f"HETATM{atom_num:5d}  {element:2s}  {element:3s} X{i+1:4d}    "
                   f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                   f"  1.00 99.00          {element:>2s}\n")
        
        f.write("END\n")


def main():
    """Command-line interface for SuperMetal prediction."""
    parser = argparse.ArgumentParser(
        description="SuperMetal: Predict metal ion binding sites in proteins",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction (generates 100 positions, filters with confidence, clusters)
  python predict.py --protein protein.pdb
  
  # Evaluate against ground truth
  python predict.py --protein protein.pdb --ground-truth ligands.mol2
  
  # Without confidence filtering
  python predict.py --protein protein.pdb --no-confidence
  
  # Custom output directory
  python predict.py --protein protein.pdb --output predictions/
        """
    )
    
    parser.add_argument('--protein', '-p', type=str, required=True,
                       help='Path to input protein PDB file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for predictions')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Path to score model directory')
    parser.add_argument('--confidence-model-dir', type=str, default=None,
                       help='Path to confidence model directory')
    parser.add_argument('--num-metals', type=int, default=100,
                       help='Number of initial metal positions to generate (default: 100)')
    parser.add_argument('--inference-steps', type=int, default=20,
                       help='Number of diffusion steps (default: 20)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Disable confidence filtering')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                       help='Confidence threshold for filtering (default: 0.5)')
    parser.add_argument('--cluster-eps', type=float, default=5.0,
                       help='DBSCAN epsilon for clustering in Angstroms (default: 5.0)')
    parser.add_argument('--ground-truth', type=str, default=None,
                       help='Path to ground truth ligand mol2 file for evaluation')
    parser.add_argument('--huggingface', action='store_true',
                       help='Download model from HuggingFace Hub')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output PDB file')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    device = torch.device('cpu') if args.cpu else None
    
    results = predict(
        protein_path=args.protein,
        model_dir=args.model_dir,
        confidence_model_dir=args.confidence_model_dir,
        num_metal_predictions=args.num_metals,
        inference_steps=args.inference_steps,
        output_dir=args.output,
        save_pdb=not args.no_save,
        device=device,
        use_huggingface=args.huggingface,
        use_confidence=not args.no_confidence,
        confidence_threshold=args.confidence_threshold,
        cluster_eps=args.cluster_eps,
        ground_truth_ligand_path=args.ground_truth
    )
    
    print(f"\n{'='*60}")
    print(f"Prediction complete for: {results['name']}")
    print(f"{'='*60}")
    print(f"Raw positions generated:     {len(results['raw_positions'])}")
    print(f"After confidence filtering:  {len(results['filtered_positions'])}")
    print(f"Final cluster centroids:     {len(results['cluster_centroids'])}")
    
    # Print predicted positions
    if len(results['cluster_centroids']) > 0:
        print(f"\nPredicted metal binding sites (x, y, z):")
        for i, pos in enumerate(results['cluster_centroids'][:10]):
            print(f"  {i+1:2d}. ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")
        if len(results['cluster_centroids']) > 10:
            print(f"  ... and {len(results['cluster_centroids']) - 10} more")
    
    if results['pdb_path']:
        print(f"\nOutput files:")
        print(f"  Zinc positions only: {results['pdb_path']}")
        if results.get('combined_pdb_path'):
            print(f"  Combined (protein + zinc): {results['combined_pdb_path']}")
            print(f"\nVisualize in PyMol: pymol {results['combined_pdb_path']}")


if __name__ == "__main__":
    main()
