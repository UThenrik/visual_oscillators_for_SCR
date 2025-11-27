import torch
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def save_model_and_results(
    save_dir,
    vae,
    latent_dynamics,
    config,
    loss_df,
    epoch=None,
    save_figures=True
):
    """Save model, training results, config, and figures to a directory."""
    save_path = Path(save_dir)
    
    if epoch is not None:
        epoch_folder = f"epoch_{epoch:03d}"
        save_path = save_path / epoch_folder
    
    save_path.mkdir(parents=True, exist_ok=True)
    suffix = ""
    
    M_inv, K, D = None, None, None
    W_raw, b_raw = None, None
    
    if hasattr(latent_dynamics, 'osc_net'):
        try:
            M_inv, K, D = latent_dynamics.osc_net.give_Minv_KD()
            print("✓ Extracted system matrices (M_inv, K, D) from latent_dynamics")
        except Exception as e:
            print(f"⚠ Could not extract system matrices: {e}")
        
        if hasattr(latent_dynamics.osc_net, 'W_raw') and hasattr(latent_dynamics.osc_net, 'b_raw'):
            W_raw = latent_dynamics.osc_net.W_raw
            b_raw = latent_dynamics.osc_net.b_raw
            print("✓ Extracted nonlinear forcing matrices (W, b) from latent_dynamics")
    
    model_checkpoint = {
        'vae_state_dict': vae.state_dict(),
        'latent_dynamics_state_dict': latent_dynamics.state_dict(),
        'epoch': epoch
    }
    torch.save(model_checkpoint, save_path / f"model_checkpoint{suffix}.pt")
    print(f"✓ Saved model checkpoint to {save_path / f'model_checkpoint{suffix}.pt'}")
    
    config_path = save_path / f"config{suffix}.json"
    with open(config_path, 'w') as f:
        config_serializable = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                config_serializable[key] = value
            else:
                config_serializable[key] = str(value)
        json.dump(config_serializable, f, indent=2)
    print(f"✓ Saved config to {config_path}")
    
    loss_df_path = save_path / f"loss_history{suffix}.csv"
    loss_df.to_csv(loss_df_path, index=False)
    print(f"✓ Saved loss history to {loss_df_path}")
    
    if save_figures:
        figures_dir = save_path / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        if not loss_df.empty:
            fig, axs = plt.subplots(3, 3, figsize=(16, 8))
            fig.suptitle("Training and Validation Losses", fontsize=16)
            
            axs[0, 0].plot(loss_df["epoch"], loss_df["t_total"], label="Train")
            axs[0, 0].plot(loss_df["epoch"], loss_df["v_total"], label="Val")
            axs[0, 0].set_title("Total Loss")
            axs[0, 0].set_xlabel("Epoch")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].set_yscale("log")
            axs[0, 0].legend()
            
            axs[0, 1].plot(loss_df["epoch"], loss_df["t_static_recon"], label="Train")
            axs[0, 1].plot(loss_df["epoch"], loss_df["v_static_recon"], label="Val")
            axs[0, 1].set_title("Static Recon Loss")
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("Loss")
            axs[0, 1].set_yscale("log")
            axs[0, 1].legend()
            
            axs[0, 2].plot(loss_df["epoch"], loss_df["t_kl"], label="Train")
            axs[0, 2].plot(loss_df["epoch"], loss_df["v_kl"], label="Val")
            axs[0, 2].set_title("KL Loss")
            axs[0, 2].set_xlabel("Epoch")
            axs[0, 2].set_ylabel("Loss")
            axs[0, 2].set_yscale("log")
            axs[0, 2].legend()
            
            axs[1, 0].plot(loss_df["epoch"], loss_df["t_dyn_recon"], label="Train")
            axs[1, 0].plot(loss_df["epoch"], loss_df["v_dyn_recon"], label="Val")
            axs[1, 0].set_title("Dynamic Recon Loss")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Loss")
            axs[1, 0].set_yscale("log")
            axs[1, 0].legend()
            
            axs[1, 1].plot(loss_df["epoch"], loss_df["t_dyn"], label="Train")
            axs[1, 1].plot(loss_df["epoch"], loss_df["v_dyn"], label="Val")
            axs[1, 1].set_title("Dynamic Loss")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].set_ylabel("Loss")
            axs[1, 1].set_yscale("log")
            axs[1, 1].legend()
            
            axs[1, 2].plot(loss_df["epoch"], loss_df["t_dyn_attn"], label="Train")
            axs[1, 2].plot(loss_df["epoch"], loss_df["v_dyn_attn"], label="Val")
            axs[1, 2].set_title("Attention Consistency Loss")
            axs[1, 2].set_xlabel("Epoch")
            axs[1, 2].set_ylabel("Loss")
            axs[1, 2].set_yscale("log")
            axs[1, 2].legend()
            
            axs[2, 0].plot(loss_df["epoch"], loss_df["t_attn_pos"], label="Train")
            axs[2, 0].plot(loss_df["epoch"], loss_df["v_attn_pos"], label="Val")
            axs[2, 0].set_title("Attention Position Loss")
            axs[2, 0].set_xlabel("Epoch")
            axs[2, 0].set_ylabel("Loss")
            axs[2, 0].set_yscale("log")
            axs[2, 0].legend()
            
            axs[2, 1].plot(loss_df["epoch"], loss_df["t_steady"], label="Train")
            axs[2, 1].plot(loss_df["epoch"], loss_df["v_steady"], label="Val")
            axs[2, 1].set_title("Steady State Loss")
            axs[2, 1].set_xlabel("Epoch")
            axs[2, 1].set_ylabel("Loss")
            axs[2, 1].set_yscale("log")
            axs[2, 1].legend()
            
            axs[2, 2].axis('off')
            
            plt.tight_layout()
            loss_plot_path = figures_dir / f"loss_plots{suffix}.pdf"
            plt.savefig(loss_plot_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved loss plots to {loss_plot_path}")
        
        if M_inv is not None and K is not None and D is not None:
            def plot_matrix(mat, title, ax=None, cmap="viridis"):
                if ax is None:
                    fig, ax = plt.subplots()
                im = ax.imshow(mat.cpu().detach().numpy(), cmap=cmap, aspect="auto")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            plot_matrix(M_inv, "Mass Matrix (M^-1)", ax=axs[0], cmap="Blues")
            plot_matrix(K, "Stiffness Matrix (K)", ax=axs[1], cmap="Reds")
            plot_matrix(D, "Damping Matrix (D)", ax=axs[2], cmap="Greens")
            plt.tight_layout()
            
            matrices_plot_path = figures_dir / f"system_matrices{suffix}.pdf"
            plt.savefig(matrices_plot_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved system matrices to {matrices_plot_path}")
        
        if W_raw is not None and b_raw is not None:
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            
            im0 = axs[0].imshow(W_raw.cpu().detach().numpy(), cmap="coolwarm", aspect="auto")
            axs[0].set_title("Nonlinear Forcing Weights (W)")
            plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
            
            im1 = axs[1].imshow(b_raw.cpu().detach().numpy().reshape(-1, 1), cmap="coolwarm", aspect="auto")
            axs[1].set_title("Nonlinear Forcing Biases (b)")
            plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            forcing_plot_path = figures_dir / f"nonlinear_forcing{suffix}.pdf"
            plt.savefig(forcing_plot_path, format='pdf', bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved nonlinear forcing matrices to {forcing_plot_path}")
    
    print(f"\n✓ All results saved to {save_path}")
    return save_path


def load_config(config_name, config_dir="configs", merge_with_base=True):
    """Load configuration from YAML file, optionally merging with base config."""
    config_path = Path(config_dir)
    
    if not config_name.endswith('.yaml'):
        config_name = config_name + '.yaml'
    
    if '/' in config_name or '\\' in config_name:
        config_file = Path(config_name)
    else:
        config_file = config_path / config_name
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_file}\n"
            f"Available configs in {config_path}:\n" + 
            "\n".join([f"  - {f.stem}" for f in config_path.glob("*.yaml")])
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    is_base = config_file.stem == 'base' or 'base' in str(config_file)
    if merge_with_base and not is_base:
        base_file = config_path / 'base.yaml'
        if base_file.exists():
            with open(base_file, 'r') as f:
                base_config = yaml.safe_load(f)
            merged_config = base_config.copy()
            merged_config.update(config)
            config = merged_config
            print(f"✓ Loaded config from {config_file} (merged with base.yaml)")
        else:
            print(f"⚠ base.yaml not found, skipping merge")
            print(f"✓ Loaded config from {config_file}")
    else:
        print(f"✓ Loaded config from {config_file}")
    
    return config

def merge_configs(base_config, override_config):
    """Merge two configuration dictionaries, with override_config taking precedence."""
    merged = base_config.copy()
    merged.update(override_config)
    return merged

