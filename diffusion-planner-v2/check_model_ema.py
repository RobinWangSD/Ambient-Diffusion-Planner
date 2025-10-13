import torch
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Import your actual model
from diffusion_planner.model.factorized_diffusion_planner import Factorized_Diffusion_Planner
from diffusion_planner.utils.config import Config


def test_ema_model_loading(config: Config, ckpt_path: str, enable_ema: bool = True, device: str = "cpu"):
    """
    Test function to verify EMA model loading following the exact logic from FactorizedDiffusionPlanner
    
    Args:
        config: Config object for the model
        ckpt_path: Path to the checkpoint file
        enable_ema: Whether to load EMA weights or regular weights
        device: Device to load the model on ("cpu" or "cuda")
    """
    
    print("=" * 60)
    print(f"Testing EMA Model Loading")
    print(f"Checkpoint: {ckpt_path}")
    print(f"EMA Enabled: {enable_ema}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Validate device
    assert device in ["cpu", "cuda"], f"device {device} not supported"
    if device == "cuda":
        assert torch.cuda.is_available(), "cuda is not available"
    
    # Initialize the model
    planner = Factorized_Diffusion_Planner(config)
    
    if ckpt_path is not None:
        print("\n1. Loading checkpoint...")
        
        # Load Lightning checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Print checkpoint structure
        print(f"\n2. Checkpoint keys: {list(checkpoint.keys())}")
        
        # Extract model weights from Lightning state_dict
        lightning_state_dict = checkpoint['state_dict']
        print(f"\n3. Total keys in state_dict: {len(lightning_state_dict)}")
        
        # Sample some keys to understand the structure
        print("\n4. First 10 keys in state_dict:")
        for i, key in enumerate(list(lightning_state_dict.keys())[:]):
            print(f"   {i+1:2d}. {key}")
        
        if enable_ema:
            print("\n5. [EMA MODE] Extracting EMA weights...")
            
            # Look for EMA weights (stored with model_ema prefix in Lightning)
            model_state_dict = {}
            for key, value in lightning_state_dict.items():
                if key.startswith('model_ema.ema.'):
                    new_key = key.replace('model_ema.ema.', '')
                    model_state_dict[new_key] = value
            
            print(f"   - Found {len(model_state_dict)} EMA keys")
            
            # Print first few EMA keys for verification
            if model_state_dict:
                print("\n   First 5 EMA keys (after transformation):")
                for i, key in enumerate(list(model_state_dict.keys())[:5]):
                    print(f"     {i+1}. {key}")
            
            # Fallback to regular model weights if EMA not found
            if not model_state_dict:
                print("\n   [WARNING] No EMA weights found! Checking available prefixes...")
                
                # Debug: Check what prefixes are available
                prefixes = set()
                for key in lightning_state_dict.keys():
                    prefix = key.split('.')[0]
                    prefixes.add(prefix)
                print(f"   Available prefixes: {prefixes}")
                
                # This would trigger assertion in original code
                print("\n   [ERROR] This would trigger assertion failure in original code!")
                print("   assert False, print(lightning_state_dict.keys())")
                
                # For testing, let's try to load regular weights as fallback
                print("\n   Attempting fallback to regular model weights...")
                model_state_dict = {k.replace('model.', ''): v
                                  for k, v in lightning_state_dict.items()
                                  if k.startswith('model.')}
                print(f"   - Found {len(model_state_dict)} regular model keys")
        
        else:
            print("\n5. [REGULAR MODE] Extracting regular model weights...")
            
            # Use regular model weights
            model_state_dict = {k.replace('model.', ''): v
                              for k, v in lightning_state_dict.items()
                              if k.startswith('model.')}
            print(f"   - Found {len(model_state_dict)} regular model keys")
            
            if model_state_dict:
                print("\n   First 5 model keys (after transformation):")
                for i, key in enumerate(list(model_state_dict.keys())[:5]):
                    print(f"     {i+1}. {key}")
        
        # Load weights into model
        print("\n6. Loading weights into model...")
        try:
            planner.load_state_dict(model_state_dict, strict=False)
            print(f"   ✓ Successfully loaded checkpoint from: {ckpt_path}")
        except Exception as e:
            print(f"   ✗ Error loading weights: {e}")
            return None
        
    else:
        print("\nNo checkpoint provided, using random weights")
    
    # Set model to eval mode and move to device
    planner.eval()
    planner = planner.to(device)
    
    print("\n7. Model setup complete!")
    print(f"   - Model in eval mode: {not planner.training}")
    print(f"   - Model on device: {next(planner.parameters()).device}")
    
    return planner


def main():
    """
    Main function to test the EMA loading
    """
    # Example usage - replace with your actual values
    config = Config(
        '/data/out/users/luobwang/factorized_planner_train_log/training_log/pl_factorized_diffusion_planner_1000000_samples-64_observed-10_predicted-chunk_size_2-future_mask_false-use_chunking_false-if_factorized_false-decoder_depth_3/2025-08-20-00:52:26/args.json',
        None,
    )  # Initialize with your config
    ckpt_path = "/data/out/users/luobwang/factorized_planner_train_log/checkpoints/pl_factorized_diffusion_planner_1000000_samples-64_observed-10_predicted-chunk_size_2-future_mask_false-use_chunking_false-if_factorized_false-decoder_depth_3/periodic/last-v1.ckpt"  # Replace with actual path
    
    # Test: Load with EMA
    print("\n" + "="*80)
    print("TEST: Loading with EMA weights")
    print("="*80)
    model_with_ema = test_ema_model_loading(
        config=config,
        ckpt_path=ckpt_path,
        enable_ema=True,
        device="cpu"
    )


if __name__ == "__main__":
    main()
