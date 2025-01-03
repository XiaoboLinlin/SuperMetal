import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils.diffusion_utils import modify_conformer, set_time
from utils.torsion import modify_conformer_torsion_angles
from scipy.spatial.transform import Rotation as R
from utils.visualise import save_zinc_to_pdb_file

def randomize_position(data_list, no_random, tr_sigma_max):
    
    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        
        complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        # complex_graph['ligand'].pos = torch.zeros_like(complex_graph['ligand'].pos)
        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            # tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(complex_graph['ligand'].pos.shape[0], 3))
            complex_graph['ligand'].pos += tr_update

def randomize_position_1(data_list, no_random, tr_sigma_max):
    
    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        
        complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        # complex_graph['ligand'].pos = torch.zeros_like(complex_graph['ligand'].pos)
        if not no_random:  # note for now the torsion angles are still randomised
            # tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(complex_graph['ligand'].pos.shape[0], 3))
            complex_graph['ligand'].pos += tr_update
            
def randomize_position_2(data_list, no_random, tr_sigma_max):
    
    ### truly randomize
    for complex_graph in data_list:
        # randomize position
        # molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        
        # complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        # complex_graph['ligand'].pos = torch.zeros_like(complex_graph['ligand'].pos)
        if not no_random:  # note for now the torsion angles are still randomised
            # tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            # tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(complex_graph['ligand'], 3))
            complex_graph['ligand'].pos = torch.normal(mean=0, std=tr_sigma_max, size=(complex_graph['ligand'].pos.shape[0], 3))

def randomize_position_multiple(data_list, no_random, tr_sigma_max, metal_num = 100):
    
    ### truly randomize
    for complex_graph in data_list:
        # randomize position
        # molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        
        # complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())

        # complex_graph['ligand'].pos = torch.zeros_like(complex_graph['ligand'].pos)
        if not no_random:  # note for now the torsion angles are still randomised
            ### 
            complex_graph['ligand'].pos = torch.normal(mean=0, std=tr_sigma_max, size=(metal_num, 3))
            complex_graph['ligand'].x = complex_graph['ligand'].x[0, :].repeat(metal_num, 1)
            complex_graph['ligand'].batch = torch.zeros(metal_num, dtype=torch.int)

def larger_original_graph(complex_graph, tr_sigma_max, metal_num = 100):
    ### make larger original graph in order to do rmsd
    
    
    pos = complex_graph['ligand'].pos
    # Calculate the base number of repetitions for each row
    base_repeats = metal_num // pos.size(0)  # Integer division

    # Calculate the number of extra rows needed to reach num_metal
    extra_rows = metal_num % pos.size(0)

    # Repeat each row the base number of times
    base_expanded_pos = pos.repeat_interleave(base_repeats, dim=0)

    # Repeat the first 'extra_rows' rows one more time to reach the total of num_metal
    additional_rows = pos[:extra_rows].repeat_interleave(1, dim=0)

    # Concatenate the base expanded tensor with the additional rows
    final_expanded_pos = torch.cat((base_expanded_pos, additional_rows), dim=0)
    complex_graph['ligand'].pos = final_expanded_pos
    complex_graph['ligand'].x = complex_graph['ligand'].x[0, :].repeat(metal_num, 1)
    complex_graph['ligand'].batch = torch.zeros(metal_num, dtype=torch.int)


def randomize_position_new(data_list, no_random, tr_sigma_max, num_metal = None):
    # in place modification of the list
    # if not no_torsion:
    #     # randomize torsion angles
    #     for complex_graph in data_list:
    #         torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['ligand'].edge_mask.sum())
    #         complex_graph['ligand'].pos = \
    #             modify_conformer_torsion_angles(complex_graph['ligand'].pos,
    #                                             complex_graph['ligand', 'ligand'].edge_index.T[
    #                                                 complex_graph['ligand'].edge_mask],
    #                                             complex_graph['ligand'].mask_rotate[0], torsion_updates)

    for complex_graph in data_list:
        # randomize position
        # molecule_center = torch.mean(complex_graph['ligand'].pos, dim=0, keepdim=True)
        
        # complex_graph['ligand'].pos = complex_graph['ligand'].pos - molecule_center
        # base_rmsd = np.sqrt(np.sum((complex_graph['ligand'].pos.cpu().numpy() - orig_complex_graph['ligand'].pos.numpy()) ** 2, axis=1).mean())
        if num_metal is not None:
            # Select the first row and add a dimension to keep it as a 2D tensor
            first_row = complex_graph['ligand'].x[0].unsqueeze(0)
            complex_graph['ligand'].x = first_row.repeat(num_metal, 1)
        else:
            num_metal = complex_graph['ligand'].pos.shape[0]
            
            
        complex_graph['ligand'].pos = torch.normal(mean=0, std=tr_sigma_max, size=(num_metal, 3))
        
        # complex_graph['ligand'].pos = torch.zeros(num_metal, 3)
        # if not no_random:  # note for now the torsion angles are still randomised
        #     # tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
        #     tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(num_metal, 3))
        #     complex_graph['ligand'].pos += tr_update


def sampling(data_list, model, inference_steps, tr_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr = tr_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
       

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            # if t_idx == 4:
            #     print("hello")
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma = t_to_sigma(t_tr)
            set_time(complex_graph_batch, t_tr, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, _, _ = model(complex_graph_batch)
            
            if torch.isnan(tr_score).any():
                print("stop")
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                # tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=1, size=(b, 3))
                tr_z = torch.zeros((len(tr_score), 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(len(tr_score), 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1])
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        # if t_idx == 4:
        #     print("here")
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence


def sampling_test1(data_list, model, inference_steps, tr_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualization_list=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr = tr_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
       

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            # if t_idx == 4:
            #     print("hello")
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma = t_to_sigma(t_tr)
            set_time(complex_graph_batch, t_tr, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, _, _ = model(complex_graph_batch)
                
            if torch.isnan(tr_score).any():
                print("stop")
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                # tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=1, size=(b, 3))
                tr_z = torch.zeros((len(tr_score), 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(len(tr_score), 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
            
            
            for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list()):
                num_metal = complex_graph['ligand'].pos.shape[0]
                new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i*num_metal:i*num_metal + num_metal])])
            
            # new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1])
            #              for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        # if t_idx == 4:
        #     print("here")
        data_list = new_data_list

        if visualization_list is not None:
            for idx, visualization in enumerate(visualization_list):
                visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
                                  part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence


def sampling_new(data_list, model, inference_steps, tr_schedule, device, t_to_sigma, model_args,
             no_random=False, ode=False, visualisation_path=None, confidence_model=None, confidence_data_list=None,
             confidence_model_args=None, batch_size=32, no_final_step_noise=False):
    N = len(data_list)

    for t_idx in range(inference_steps):
        t_tr = tr_schedule[t_idx]
        dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
       

        loader = DataLoader(data_list, batch_size=batch_size)
        new_data_list = []

        for complex_graph_batch in loader:
            # if t_idx == 4:
            #     print("hello")
            b = complex_graph_batch.num_graphs
            complex_graph_batch = complex_graph_batch.to(device)

            tr_sigma = t_to_sigma(t_tr)
            set_time(complex_graph_batch, t_tr, b, model_args.all_atoms, device)
            
            with torch.no_grad():
                tr_score, _, _ = model(complex_graph_batch)
            
            
            if torch.isnan(tr_score).any():
                print("stop")
            tr_g = tr_sigma * torch.sqrt(torch.tensor(2 * np.log(model_args.tr_sigma_max / model_args.tr_sigma_min)))

            if ode:
                tr_perturb = (0.5 * tr_g ** 2 * dt_tr * tr_score.cpu()).cpu()
                # rot_perturb = (0.5 * rot_score.cpu() * dt_rot * rot_g ** 2).cpu()
            else:
                # tr_z = torch.zeros((b, 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                #     else torch.normal(mean=0, std=1, size=(b, 3))
                tr_z = torch.zeros((len(tr_score), 3)) if no_random or (no_final_step_noise and t_idx == inference_steps - 1) \
                    else torch.normal(mean=0, std=1, size=(len(tr_score), 3))
                tr_perturb = (tr_g ** 2 * dt_tr * tr_score.cpu() + tr_g * np.sqrt(dt_tr) * tr_z).cpu()
            new_data_list.extend([modify_conformer(complex_graph, tr_perturb[i:i + 1])
                         for i, complex_graph in enumerate(complex_graph_batch.to('cpu').to_data_list())])
        # if t_idx == 4:
        #     print("here")
        data_list = new_data_list

        if visualisation_path is not None:
            for complex_graph in data_list:
                file_name = f"{visualisation_path}/zinc_{complex_graph['name']}_{t_idx}.pdb"
                zinc_coords = complex_graph['ligand'].pos + complex_graph.original_center
                save_zinc_to_pdb_file(zinc_coords, file_name)
                
            # for complex_graph in data_list:
            #     file_path_template = 'zinc_atoms_{complex_name}_{timestep}.pdb'
            #     # comment: 
            # # end for
            # for idx, visualization in enumerate(visualization_list):
            #     visualization.add((data_list[idx]['ligand'].pos + data_list[idx].original_center).detach().cpu(),
            #                       part=1, order=t_idx + 2)

    with torch.no_grad():
        if confidence_model is not None:
            loader = DataLoader(data_list, batch_size=batch_size)
            confidence_loader = iter(DataLoader(confidence_data_list, batch_size=batch_size))
            confidence = []
            for complex_graph_batch in loader:
                complex_graph_batch = complex_graph_batch.to(device)
                if confidence_data_list is not None:
                    confidence_complex_graph_batch = next(confidence_loader).to(device)
                    confidence_complex_graph_batch['ligand'].pos = complex_graph_batch['ligand'].pos
                    set_time(confidence_complex_graph_batch, 0, N, confidence_model_args.all_atoms, device)
                    confidence.append(confidence_model(confidence_complex_graph_batch))
                else:
                    confidence.append(confidence_model(complex_graph_batch))
            confidence = torch.cat(confidence, dim=0)
        else:
            confidence = None

    return data_list, confidence
