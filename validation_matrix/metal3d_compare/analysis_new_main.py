import os
import argparse
import csv
from Bio.PDB import PDBParser
from validation_matrix.utils.cluster_centroid import find_centroid
from validation_matrix.utils.find_zinc_pos import find_real_zinc_pos
from utils.nearest_point_dist import get_nearest_point_distances

def extract_zinc_positions_pdb(file_path):
    zinc_positions = []
    
    # Read the PDB file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Check if the line starts with "HETATM" and contains " ZN "
            if line.startswith("HETATM") and " ZN " in line:
                # Extract the coordinates (columns 31-38, 39-46, 47-54)
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                zinc_positions.append((x, y, z))
    
    return zinc_positions

def main(base_dir, rmsd_classification_cutoff, data_dir):
    # Initialize the PDB parser
    parser = PDBParser(QUIET=True)
    # Prepare the CSV file
    csv_file = "results.csv"
    csv_columns = ["name", "accuracy", "coverage"]

    # Iterate through the directory structure
    zn_positions_dict = {}
    n_zn_positions_metal3d, n_zn_positions_truth, n_qualified_real_idx = 0, 0, 0
    results = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        probe_file = next((file for file in os.listdir(subdir_path) if file.startswith("probe_") and file.endswith(".pdb")), None)
        name = subdir
        if not probe_file:
            print(f"No probe files found in {name}, accaracy is none, coverage is 0.")
            zn_positions_truth = find_real_zinc_pos(os.path.join(data_dir, f"{name}/{name}_ligands.mol2"))
            n_zn_positions_truth += len(zn_positions_truth)
            results.append({"name": name, "accuracy": None, "coverage": 0})
            continue
       
        file_path = os.path.join(subdir_path, probe_file)
        
        try:
            zn_positions_metal3d = extract_zinc_positions_pdb(file_path)
            zn_positions_truth = find_real_zinc_pos(os.path.join(data_dir, f"{name}/{name}_ligands.mol2"))
            nearest_dist_points, indices = get_nearest_point_distances(zn_positions_metal3d, zn_positions_truth)
            qualified_points, qualified_real_idx = [], []
            for i, (centroid_dist, idx) in enumerate(zip(nearest_dist_points, indices)):
                if centroid_dist <= rmsd_classification_cutoff:
                    qualified_points.append(zn_positions_metal3d[i])
                    qualified_real_idx.append(float(idx))
            accuracy = len(qualified_real_idx) / len(zn_positions_metal3d) * 100
            coverage = len(set(qualified_real_idx)) / len(zn_positions_truth) * 100
            print(f"accuracy: {accuracy:.2f}%, coverage: {coverage:.2f}%, name: {name}")
            results.append({"name": name, "accuracy": accuracy, "coverage": coverage})
        except Exception as e:
            zn_positions_metal3d = []
            qualified_real_idx = []
            zn_positions_truth = find_real_zinc_pos(os.path.join(data_dir, f"{name}/{name}_ligands.mol2"))
            print(f"An error occurred on {name}: {e}")
        n_zn_positions_metal3d += len(zn_positions_metal3d)
        n_zn_positions_truth += len(zn_positions_truth)
        n_qualified_real_idx += len(set(qualified_real_idx))
    
    total_accuracy = n_qualified_real_idx / n_zn_positions_metal3d * 100
    total_coverage = n_qualified_real_idx / n_zn_positions_truth * 100
    print(f"total accuracy: {total_accuracy:.2f}%, total coverage: {total_coverage:.2f}%")
    results.append({"name": "total", "accuracy": total_accuracy, "coverage": total_coverage})

    # Write results to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in results:
            writer.writerow(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDB files and calculate zinc position accuracy and coverage.")
    parser.add_argument("--base_dir", default="/home/xiaobo/project/diffusion/metal-site-prediction/Metal3D/metal_3d_test_0p1", type=str, help="Base directory containing the PDB files.")
    parser.add_argument("--rmsd_classification_cutoff", default=5, type=float, help="RMSD classification cutoff value.")
    parser.add_argument("--data_dir", default="/home/xiaobo/project/diffusion/diffdock-metal5-zincbind/data/zincbind_cleaned_processed", type=str, help="Directory containing the zincbind cleaned processed data.")
    
    args = parser.parse_args()
    main(args.base_dir, args.rmsd_classification_cutoff, args.data_dir)
