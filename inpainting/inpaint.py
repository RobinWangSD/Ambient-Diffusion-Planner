import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.train_utils import set_seed
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.dataset import DiffusionPlannerData

def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description='Diffusion Inpainting for Missing States in a Dataset')
    parser.add_argument('--name', type=str, help='experiment name', default="diffusion-planner-inpainting-from-data")
    parser.add_argument('--save_dir', type=str, help='save dir for results', default="./inpainting_results")
    
    # --- Model and Data Paths ---
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset for inpainting')
    parser.add_argument('--data_list', type=str, help='Data list file', default=None)
    parser.add_argument('--output_path', type=str, help='Path to save inpainted data', default="./inpainted_data")
    parser.add_argument('--normalization_file_path', default='normalization.json', type=str)

    # --- Inpainting Parameters ---
    parser.add_argument('--num_diffusion_steps', type=int, help='Number of diffusion steps for inpainting', default=50)
    parser.add_argument('--use_ema', default=True, type=boolean, help='Use EMA model from checkpoint if available')
    
    # --- Data Dimensions (must match training configuration) ---
    parser.add_argument('--future_len', type=int, help='Number of future time points', default=80)
    parser.add_argument('--time_len', type=int, help='Number of past time points', default=21)
    parser.add_argument('--agent_state_dim', type=int, help='Past state dimension for agents', default=11)
    parser.add_argument('--agent_num', type=int, help='Total number of agents', default=32)
    parser.add_argument('--static_objects_state_dim', type=int, help='State dimension for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='Number of static objects', default=5)
    parser.add_argument('--lane_len', type=int, help='Number of lane points', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='State dimension for lane points', default=12)
    parser.add_argument('--lane_num', type=int, help='Number of lanes', default=70)
    parser.add_argument('--route_len', type=int, help='Number of route lane points', default=20)
    parser.add_argument('--route_state_dim', type=int, help='State dimension for route lane points', default=12)
    parser.add_argument('--route_num', type=int, help='Number of route lanes', default=25)
    
    # --- Model Architecture (must match training configuration) ---
    parser.add_argument('--encoder_depth', type=int, help='Number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='Number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='Number of attention heads', default=6)
    parser.add_argument('--hidden_dim', type=int, help='Hidden dimension size', default=192)
    parser.add_argument('--diffusion_model_type', type=str, help='Type of diffusion model', default='x_start')
    parser.add_argument('--predicted_neighbor_num', type=int, help='Number of neighbor agents to predict', default=10)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='Encoder drop path rate', default=0.0)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='Decoder drop path rate', default=0.0)
    
    # --- Processing Parameters ---
    parser.add_argument('--batch_size', type=int, help='Batch size for inpainting', default=32)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', type=str, help='Device to use (e.g., "cuda", "cpu")', default='cuda')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility', default=3407)
    
    # --- Inference Options ---
    parser.add_argument('--num_samples', type=int, help='Number of inpainting samples to generate per input', default=1)
    parser.add_argument('--save_visualization', default=False, type=boolean, help='Save visualization plots of the results')
    
    args = parser.parse_args()
    
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    return args

class InpaintingDataset(Dataset):
    """
    Dataset wrapper for inpainting.
    This class identifies missing states from the dataset itself and generates a corresponding mask.
    A state is considered "missing" if its feature vector is all zeros.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get the tuple of data from the base dataset
        data_tuple = self.base_dataset[idx]
        
        # --- FIX: Convert NumPy arrays to PyTorch Tensors ---
        # The base dataset returns NumPy arrays, which need to be converted.
        ego_future_gt = torch.from_numpy(data_tuple[1]).float()
        neighbors_future_gt = torch.from_numpy(data_tuple[3]).float()
        
        # --- Generate Mask from Data (Now using Tensors) ---
        # A state is considered present if its feature vector is not all zeros.
        # `True` means the data is present (keep), `False` means it's missing (inpaint).
        ego_mask_present = torch.any(ego_future_gt != 0, dim=-1)
        neighbors_mask_present = torch.any(neighbors_future_gt != 0, dim=-1)

        # Combine ego and neighbor masks
        # Shape: (1 + num_neighbors, future_len)
        combined_mask = torch.cat([ego_mask_present.unsqueeze(0), neighbors_mask_present], dim=0)
        
        # Expand the mask to match the feature dimension (x, y, cos(h), sin(h))
        # Final shape: (1 + num_neighbors, future_len, 4)
        final_mask = combined_mask.unsqueeze(-1).expand(-1, -1, 4).bool()
        
        # Return the original data tuple (still as NumPy) with the derived mask (as Tensor)
        return data_tuple + (final_mask,)

# You may need to import the dpm_sampler directly if it's not already in the scope
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler

def inpaint_states(model, data_tuple_with_mask, args):
    """
    Performs diffusion-based inpainting by directly invoking the dpm_sampler
    with a custom corrector function to enforce known states.
    """
    model.eval()
    
    with torch.no_grad():
        # --- 1. Standard Data Preparation (same as before) ---
        *data_tuple, mask = data_tuple_with_mask
        
        ego_current_state, ego_future_gt, neighbor_agents_past, neighbors_future_gt, \
        lanes, lanes_speed_limit, lanes_has_speed_limit, route_lanes, \
        route_lanes_speed_limit, route_lanes_has_speed_limit, static_objects = data_tuple

        inputs = {
            'ego_current_state': ego_current_state, 'neighbor_agents_past': neighbor_agents_past,
            'lanes': lanes, 'lanes_speed_limit': lanes_speed_limit, 'lanes_has_speed_limit': lanes_has_speed_limit,
            'route_lanes': route_lanes, 'route_lanes_speed_limit': route_lanes_speed_limit, 
            'route_lanes_has_speed_limit': route_lanes_has_speed_limit, 'static_objects': static_objects
        }
        
        ego_future = torch.cat([
            ego_future_gt[..., :2],
            torch.cos(ego_future_gt[..., 2]).unsqueeze(-1),
            torch.sin(ego_future_gt[..., 2]).unsqueeze(-1)
        ], dim=-1)
        
        neighbors_valid_mask = torch.any(neighbors_future_gt != 0, dim=-1)
        neighbors_future = torch.cat([
            neighbors_future_gt[..., :2],
            torch.cos(neighbors_future_gt[..., 2]).unsqueeze(-1),
            torch.sin(neighbors_future_gt[..., 2]).unsqueeze(-1)
        ], dim=-1)
        neighbors_future[~neighbors_valid_mask] = 0.
        
        inputs = args.observation_normalizer(inputs)
        
        batch_size = ego_current_state.shape[0]
        device = ego_current_state.device
        Pn = args.predicted_neighbor_num
        
        ego_current = inputs["ego_current_state"][:, :4]
        neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
        neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

        gt_future = torch.cat([ego_future.unsqueeze(1), neighbors_future], dim=1)
        current_states = torch.cat([ego_current.unsqueeze(1), neighbors_current], dim=1)
        gt_future_normalized = args.state_normalizer(gt_future)
        
        # The mask for the future part of the trajectory
        future_mask = mask # Shape [B, 1+Pn, T, 4]
        
        sde = model.sde

        # --- 2. Run the Encoder to get context ---
        # The decoder's DiT model needs the context from the encoder
        encoder_outputs = model.encoder(inputs)
        ego_neighbor_encoding = encoder_outputs['encoding']

        # --- 3. Prepare Initial State `x_T` for Inpainting ---
        T = sde.T
        t_T = torch.full((batch_size,), T, device=device)
        
        # Noise the known parts of the data to time T
        mean_T, std_T = sde.marginal_prob(gt_future_normalized, t_T)
        std_T = std_T.view(-1, *([1] * (len(gt_future_normalized.shape) - 1)))
        known_part_at_T = mean_T + std_T * torch.randn_like(mean_T)
        
        # Start with pure noise for the missing parts
        noise_for_missing = torch.randn_like(gt_future_normalized)

        # Combine them using the mask
        x_T_future = torch.where(future_mask, known_part_at_T, noise_for_missing)
        
        # Combine with current state to match the DiT model's input shape
        # Input shape: [B, P, (1 + T) * 4]
        x_T_full = torch.cat([current_states.unsqueeze(2), x_T_future], dim=2).reshape(batch_size, -1, (args.future_len + 1) * 4)

        # --- 4. Define the Corrector Function for the DPM-Solver ---
        def inpainting_corrector(xt, t, step):
            # xt has shape [B, P, (1+T)*4]
            xt = xt.reshape(batch_size, -1, args.future_len + 1, 4)
            
            # We only correct the future part
            xt_future = xt[:, :, 1:, :]

            # Find the mean and std of the known data at the current time t
            mean_t, std_t = sde.marginal_prob(gt_future_normalized, t)
            std_t = std_t.view(-1, *([1] * (len(xt_future.shape) - 1)))

            # Sample from this distribution for the known parts
            corrected_known_part = mean_t + std_t * torch.randn_like(mean_t)

            # Use the mask to replace only the known parts of xt_future
            xt_future_corrected = torch.where(future_mask, corrected_known_part, xt_future)

            # Re-assemble the full trajectory and reshape
            xt_corrected = torch.cat([xt[:, :, :1, :], xt_future_corrected], dim=2)
            return xt_corrected.reshape(batch_size, -1, (args.future_len + 1) * 4)

        # --- 5. Call the DPM-Sampler Directly with the Corrector ---
        dit_model = model.decoder.decoder.dit
        
        inpainted_x0 = dpm_sampler(
            model=dit_model,
            x_T=x_T_full,
            diffusion_steps=args.num_diffusion_steps,
            other_model_params={
                "cross_c": ego_neighbor_encoding, 
                "route_lanes": inputs['route_lanes'],
                "neighbor_current_mask": neighbor_current_mask                            
            },
            dpm_solver_params={
                # This is the crucial part for inpainting!
                "correcting_xt_fn": inpainting_corrector,
            }
        )

        # --- 6. Post-process the Results (same as before) ---
        # Reshape to [B, P, T, 4] and extract the future part
        inpainted = inpainted_x0.reshape(batch_size, -1, args.future_len + 1, 4)[:, :, 1:, :]
        inpainted = args.state_normalizer.inverse(inpainted)
        
        ego_inpainted_raw = inpainted[:, 0, :, :]
        neighbors_inpainted_raw = inpainted[:, 1:, :, :]
        
        ego_heading = torch.atan2(ego_inpainted_raw[..., 3], ego_inpainted_raw[..., 2])
        ego_inpainted = torch.cat([ego_inpainted_raw[..., :2], ego_heading.unsqueeze(-1)], dim=-1)
        
        neighbors_heading = torch.atan2(neighbors_inpainted_raw[..., 3], neighbors_inpainted_raw[..., 2])
        neighbors_inpainted = torch.cat([neighbors_inpainted_raw[..., :2], neighbors_heading.unsqueeze(-1)], dim=-1)
        
        neighbor_validity_mask = torch.any(neighbors_future_gt != 0, dim=(-1, -2))
        neighbors_inpainted[~neighbor_validity_mask] = 0.
        
    return ego_inpainted, neighbors_inpainted

def visualize_scenario(
    ego_original, ego_inpainted, ego_mask,
    neighbors_original, neighbors_inpainted, neighbors_mask,
    lanes, route_lanes,
    save_path=None
):
    """
    Visualize the full scenario with improved clarity for original, missing,
    and inpainted trajectories.
    """
    
    # --- Convert all tensor inputs to numpy for plotting ---
    # The DataLoader provides tensors, but plotting functions need numpy arrays.
    # Ensure all relevant inputs are converted.
    if isinstance(ego_original, torch.Tensor): ego_original = ego_original.cpu().numpy()
    if isinstance(ego_inpainted, torch.Tensor): ego_inpainted = ego_inpainted.cpu().numpy()
    if isinstance(ego_mask, torch.Tensor): ego_mask = ego_mask.cpu().numpy()
    if isinstance(neighbors_original, torch.Tensor): neighbors_original = neighbors_original.cpu().numpy()
    if isinstance(neighbors_inpainted, torch.Tensor): neighbors_inpainted = neighbors_inpainted.cpu().numpy()
    if isinstance(neighbors_mask, torch.Tensor): neighbors_mask = neighbors_mask.cpu().numpy()
    if isinstance(lanes, torch.Tensor): lanes = lanes.cpu().numpy()
    if isinstance(route_lanes, torch.Tensor): route_lanes = route_lanes.cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(24, 12)) # Make figure wider for better readability
    fig.suptitle('Scenario Inpainting Results: Original vs. Inpainted', fontsize=18)

    # --- Colors and Markers ---
    COLOR_EGO_OBSERVED = 'blue'
    COLOR_EGO_MISSING_GT = 'lightgray' # Missing ground truth parts
    COLOR_EGO_INPAINTED = 'cyan'

    COLOR_NEIGHBOR_OBSERVED = 'red'
    COLOR_NEIGHBOR_MISSING_GT = 'darkgray' # Missing ground truth parts
    COLOR_NEIGHBOR_INPAINTED = 'orange'
    
    COLOR_LANES = 'gray'
    COLOR_ROUTE_LANES = 'green'

    MARKER_OBSERVED = 'o' # Circles for observed points
    MARKER_MISSING_GT = 'x' # X for ground truth missing points
    MARKER_INPAINTED = 'D' # Diamonds for inpainted points
    LINESTYLE_TRAJ = '-'
    LINESTYLE_MISSING = ':'


    # --- Iterate through both subplots to set up map elements ---
    for a in ax:
        # Plot Lanes
        for lane in lanes:
            if np.any(lane):
                a.plot(lane[:, 0], lane[:, 1], color=COLOR_LANES, linestyle='-', alpha=0.3, linewidth=0.5)
        # Highlight Route Lanes
        for r_lane in route_lanes:
            if np.any(r_lane):
                a.plot(r_lane[:, 0], r_lane[:, 1], color=COLOR_ROUTE_LANES, linestyle='--', alpha=0.6, linewidth=0.8)

    # --- Plot Original Scenario (Left Subplot) ---
    ax[0].set_title('Original Data (Observed & Missing Ground Truth)', fontsize=14)
    
    # Ego Original
    ego_obs_mask_bool = ego_mask[:, 0].astype(bool) # Convert to boolean
    ax[0].plot(ego_original[ego_obs_mask_bool, 0], ego_original[ego_obs_mask_bool, 1], 
               marker=MARKER_OBSERVED, linestyle=LINESTYLE_TRAJ, color=COLOR_EGO_OBSERVED, 
               label='Ego (Observed GT)', markersize=5, zorder=5)
    if np.any(~ego_obs_mask_bool):
        ax[0].plot(ego_original[~ego_obs_mask_bool, 0], ego_original[~ego_obs_mask_bool, 1], 
                   marker=MARKER_MISSING_GT, linestyle=LINESTYLE_MISSING, color=COLOR_EGO_MISSING_GT, 
                   label='Ego (Missing GT)', markersize=6, zorder=4)
    
    # Neighbors Original
    for i, neighbor in enumerate(neighbors_original):
        if np.any(neighbor): # Only plot valid neighbors
            neighbor_obs_mask_bool = neighbors_mask[i, :, 0].astype(bool)
            
            # Observed parts of neighbor trajectory
            ax[0].plot(neighbor[neighbor_obs_mask_bool, 0], neighbor[neighbor_obs_mask_bool, 1], 
                       marker=MARKER_OBSERVED, linestyle=LINESTYLE_TRAJ, color=COLOR_NEIGHBOR_OBSERVED, 
                       alpha=0.7, markersize=4, zorder=5)
            # Missing GT parts of neighbor trajectory
            if np.any(~neighbor_obs_mask_bool):
                 ax[0].plot(neighbor[~neighbor_obs_mask_bool, 0], neighbor[~neighbor_obs_mask_bool, 1], 
                            marker=MARKER_MISSING_GT, linestyle=LINESTYLE_MISSING, color=COLOR_NEIGHBOR_MISSING_GT, 
                            alpha=0.5, markersize=5, zorder=4)
    
    # Dummy plots for legend entries for neighbors (as they are in a loop)
    ax[0].plot([], [], marker=MARKER_OBSERVED, linestyle=LINESTYLE_TRAJ, color=COLOR_NEIGHBOR_OBSERVED, label='Neighbor (Observed GT)')
    ax[0].plot([], [], marker=MARKER_MISSING_GT, linestyle=LINESTYLE_MISSING, color=COLOR_NEIGHBOR_MISSING_GT, label='Neighbor (Missing GT)')


    # --- Plot Inpainted Scenario (Right Subplot) ---
    ax[1].set_title('Inpainted Results vs. Original Observed', fontsize=14)
    
    # Ego Inpainted
    ax[1].plot(ego_inpainted[:, 0], ego_inpainted[:, 1], 
               marker=MARKER_INPAINTED, linestyle=LINESTYLE_TRAJ, color=COLOR_EGO_INPAINTED, 
               label='Ego (Inpainted)', markersize=6, zorder=10, linewidth=2)
    # Overlay Ego Original Observed for comparison
    ax[1].plot(ego_original[ego_obs_mask_bool, 0], ego_original[ego_obs_mask_bool, 1], 
               marker=MARKER_OBSERVED, linestyle='None', color=COLOR_EGO_OBSERVED, 
               alpha=0.7, label='Ego (Observed GT)', markersize=5, zorder=11)

    # Neighbors Inpainted
    for i, neighbor_ip in enumerate(neighbors_inpainted):
         if np.any(neighbor_ip): # Only plot valid neighbors
            ax[1].plot(neighbor_ip[:, 0], neighbor_ip[:, 1], 
                       marker=MARKER_INPAINTED, linestyle=LINESTYLE_TRAJ, color=COLOR_NEIGHBOR_INPAINTED, 
                       alpha=0.8, markersize=5, zorder=9, linewidth=1.5)
            
            # Overlay Neighbors Original Observed for comparison
            neighbor_orig = neighbors_original[i]
            neighbor_obs_mask_bool = neighbors_mask[i, :, 0].astype(bool)
            ax[1].plot(neighbor_orig[neighbor_obs_mask_bool, 0], neighbor_orig[neighbor_obs_mask_bool, 1], 
                       marker=MARKER_OBSERVED, linestyle='None', color=COLOR_NEIGHBOR_OBSERVED, 
                       alpha=0.5, markersize=4, zorder=10)

    # Dummy plots for legend entries for neighbors (as they are in a loop)
    ax[1].plot([], [], marker=MARKER_INPAINTED, linestyle=LINESTYLE_TRAJ, color=COLOR_NEIGHBOR_INPAINTED, label='Neighbor (Inpainted)')
    ax[1].plot([], [], marker=MARKER_OBSERVED, linestyle='None', color=COLOR_NEIGHBOR_OBSERVED, label='Neighbor (Observed GT)')

    # --- Final Touches for Axes ---
    for a in ax:
        a.set_xlabel('X coordinate (m)', fontsize=12)
        a.set_ylabel('Y coordinate (m)', fontsize=12)
        a.legend(fontsize=10, loc='upper left')
        a.grid(True, linestyle='--', alpha=0.6)
        a.axis('equal') # Maintain aspect ratio
        
        # Optionally, set limits based on ego's observed range for tighter view
        # This can be adjusted or removed if a broader map view is preferred.
        min_x, max_x = ego_original[ego_obs_mask_bool, 0].min(), ego_original[ego_obs_mask_bool, 0].max()
        min_y, max_y = ego_original[ego_obs_mask_bool, 1].min(), ego_original[ego_obs_mask_bool, 1].max()
        padding = 10 # meters
        a.set_xlim(min_x - padding, max_x + padding)
        a.set_ylim(min_y - padding, max_y + padding)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight') # Higher DPI for better quality
    plt.close(fig)

def main(args):
    set_seed(args.seed)
    
    # --- Setup Directories ---
    os.makedirs(args.output_path, exist_ok=True)
    viz_path = os.path.join(args.output_path, 'visualizations')
    if args.save_visualization:
        os.makedirs(viz_path, exist_ok=True)
    
    # --- Load Model ---
    print(f"Loading model from {args.checkpoint_path}...")
    diffusion_planner = Diffusion_Planner(args)
    if args.checkpoint_path is not None:
        state_dict:Dict = torch.load(args.checkpoint_path, map_location=args.device)
        
        if 'ema_state_dict' in state_dict and args.use_ema:
            state_dict = state_dict['ema_state_dict']
        elif "model" in state_dict:
            state_dict = state_dict['model']

        # Handle DDP 'module.' prefix
        if list(state_dict.keys())[0].startswith("module."):
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        else:
            model_state_dict = state_dict
            
        diffusion_planner.load_state_dict(model_state_dict)
        diffusion_planner = diffusion_planner.to(args.device)
        diffusion_planner.eval()    
    else:
        print("Warning: No checkpoint path provided. Loading a model with random weights.")
    print("✅ Model loaded successfully.")
    
    # --- Prepare Dataset ---
    dataset = DiffusionPlannerData(
        args.data_path, args.data_list, args.agent_num, 
        args.predicted_neighbor_num, args.future_len
    )
    
    inpainting_dataset = InpaintingDataset(dataset)
    
    data_loader = DataLoader(
        inpainting_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True
    )
    
    print(f"📊 Dataset prepared with {len(dataset)} samples.")
    print(f"🚀 Starting inpainting with {args.num_diffusion_steps} diffusion steps...")
    
    # --- Main Inpainting Loop ---
    for batch_idx, batch_tuple_with_mask in enumerate(tqdm(data_loader, desc="Inpainting Batches")):
        # Move all tensors in the batch to the specified device
        # Handle NumPy arrays from the original data tuple
        batch_on_device = []
        for item in batch_tuple_with_mask:
            if isinstance(item, torch.Tensor):
                batch_on_device.append(item.to(args.device))
            else: # Assume it's a NumPy array that needs conversion for inpainting
                batch_on_device.append(torch.from_numpy(item).float().to(args.device))

        ego_samples = []
        neighbor_samples = []
        
        for _ in range(args.num_samples):
            ego_inpainted, neighbors_inpainted = inpaint_states(
                diffusion_planner, batch_on_device, args
            )
            ego_samples.append(ego_inpainted)
            neighbor_samples.append(neighbors_inpainted)
        
        ego_final = torch.stack(ego_samples, dim=1) if args.num_samples > 1 else ego_samples[0]
        neighbors_final = torch.stack(neighbor_samples, dim=1) if args.num_samples > 1 else neighbor_samples[0]
        
        # --- Save Results and Visualizations ---
        original_data_tuple = batch_tuple_with_mask[:-1] # This contains the original NumPy arrays
        mask = batch_tuple_with_mask[-1] # This is a Tensor

        # Save the numerical results
        output_file_path = os.path.join(args.output_path, f'batch_{batch_idx:04d}.npz')
        np.savez_compressed(
            output_file_path,
            ego_original=original_data_tuple[1],
            ego_inpainted=ego_final.cpu().numpy(),
            neighbors_original=original_data_tuple[3],
            neighbors_inpainted=neighbors_final.cpu().numpy(),
            mask=mask.cpu().numpy(),
            ego_current_state=original_data_tuple[0],
        )
        
        if args.save_visualization:
            for i in range(min(3, args.batch_size)): # Visualize up to 3 examples per batch
                sample_viz_path = os.path.join(viz_path, f'batch{batch_idx:04d}_sample{i:02d}.png')
                
                # Select the first sample if multiple were generated
                ego_to_viz = ego_final[i, 0] if args.num_samples > 1 else ego_final[i]
                neighbors_to_viz = neighbors_final[i, 0] if args.num_samples > 1 else neighbors_final[i]
                
                # *** CALL THE NEW VISUALIZATION FUNCTION ***
                visualize_scenario(
                    ego_original=original_data_tuple[1][i],
                    ego_inpainted=ego_to_viz.cpu().numpy(),
                    ego_mask=mask[i, 0].cpu().numpy(), # Ego is the 0-th agent
                    neighbors_original=original_data_tuple[3][i],
                    neighbors_inpainted=neighbors_to_viz.cpu().numpy(),
                    neighbors_mask=mask[i, 1:].cpu().numpy(), # Neighbors are the rest
                    lanes=original_data_tuple[4][i],
                    route_lanes=original_data_tuple[7][i],
                    save_path=sample_viz_path
                )

    print(f"\n✅ Inpainting complete. Results saved in: {args.output_path}")

if __name__ == '__main__':
    args = get_args()
    main(args)