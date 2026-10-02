import torch
import numpy as np
import pickle

def prepare_static_data(dataset, configs_subset):
    """Prepare validation observations and actuation inputs from dataset."""
    
    # Get sample config from the correct subset
    config_sample = list(configs_subset.values())[0]
    
    # Load observations
    o = dataset['images']
    if o.max() > 1.0:
        o = o / 255.0
    
    # Ensure shape (N, C, H, W)
    if o.ndim == 3:
        o = o[:, None, :, :]
    elif o.shape[-1] == 3 and o.shape[1] != 3:
        o = np.transpose(o, (0, 3, 1, 2))
    
    tensor_o = torch.from_numpy(o).float()
    
    # Load actuator inputs
    actuation_dim = config_sample.get("actuation_dim", 2)
    actuator_input_list = []
    
    for i in range(1, actuation_dim + 1):
        actuator_input = torch.tensor(dataset[f'p{i}'], dtype=torch.float32)
        if actuator_input.ndim == 1:
            actuator_input = actuator_input.unsqueeze(-1)
        actuator_input_list.append(actuator_input)
    
    tensor_actuator_inputs = torch.cat(actuator_input_list, dim=-1)
    
    # Apply delayed actuation if specified
    num_delays = config_sample.get("num_actuation_delays", 1)
    
    if num_delays > 1:
        n_samples = tensor_actuator_inputs.shape[0]
        actuation_dim_base = actuation_dim
        
        delayed_actuator_inputs = torch.zeros(n_samples, actuation_dim_base * num_delays, dtype=torch.float32)
        
        for t in range(n_samples):
            delayed_states = []
            for delay in range(num_delays - 1, -1, -1):
                if t - delay < 0:
                    delayed_states.append(tensor_actuator_inputs[0])
                else:
                    delayed_states.append(tensor_actuator_inputs[t - delay])
            delayed_actuator_inputs[t] = torch.cat(delayed_states, dim=0)
        
        tensor_actuator_inputs = delayed_actuator_inputs
    
    return tensor_o, tensor_actuator_inputs

def prepare_dynamic_val_data(dataset, configs_subset):
    """Prepare validation observations and actuation inputs from dataset."""
    
    # Get sample config from the correct subset
    config_sample = list(configs_subset.values())[0]
    
    # Load observations
    o = dataset['images']
    if o.max() > 1.0:
        o = o / 255.0
    
    # Ensure shape (N, C, H, W)
    if o.ndim == 3:
        o = o[:, None, :, :]
    elif o.shape[-1] == 3 and o.shape[1] != 3:
        o = np.transpose(o, (0, 3, 1, 2))
    
    tensor_o = torch.from_numpy(o).float()
    
    # Load actuator inputs
    actuation_dim = config_sample.get("actuation_dim", 2)
    actuator_input_list = []
    
    for i in range(1, actuation_dim + 1):
        actuator_input = torch.tensor(dataset[f'p{i}'], dtype=torch.float32)
        if actuator_input.ndim == 1:
            actuator_input = actuator_input.unsqueeze(-1)
        actuator_input_list.append(actuator_input)
    
    tensor_actuator_inputs = torch.cat(actuator_input_list, dim=-1)
    
    # Apply delayed actuation if specified
    num_delays = config_sample.get("num_actuation_delays", 1)
    
    if num_delays > 1:
        n_samples = tensor_actuator_inputs.shape[0]
        actuation_dim_base = actuation_dim
        
        delayed_actuator_inputs = torch.zeros(n_samples, actuation_dim_base * num_delays, dtype=torch.float32)
        
        for t in range(n_samples):
            delayed_states = []
            for delay in range(num_delays - 1, -1, -1):
                if t - delay < 0:
                    delayed_states.append(tensor_actuator_inputs[0])
                else:
                    delayed_states.append(tensor_actuator_inputs[t - delay])
            delayed_actuator_inputs[t] = torch.cat(delayed_states, dim=0)
        
        tensor_actuator_inputs = delayed_actuator_inputs
    
    # Split train/val
    train_val_ratio = config_sample.get("train_val_ratio", 0.8)
    n_train = int(train_val_ratio * tensor_o.shape[0])
    
    val_o = tensor_o[n_train:]
    val_u = tensor_actuator_inputs[n_train:]
    
    return val_o, val_u

def load_simulator_pickles_and_compute_latents(pickle_paths, vaes, configs, device, dt=None):
    """
    Load simulator pickle files (list of dicts: obs_decoded, prev_decoded, next_decoded, ...)
    and compute z, z_dot for each model from observations only.
    Returns: dict[file_stem][model_name] with 'z', 'z_dot' (tensors n_states x latent_dim),
             and 'obs', 'prev_obs', 'next_obs' (tensors n_states x 1 x H x W).
    """
    from pathlib import Path
    out = {}
    for path in pickle_paths:
        path = Path(path)
        with open(path, "rb") as f:
            states = pickle.load(f)
        if not states:
            out[path.stem] = {}
            continue
        # Stack obs: (N, H, W) -> (N, 1, H, W) float32
        def _to_chw(img):
            img = np.asarray(img, dtype=np.float32)
            if img.ndim == 3:
                if img.shape[2] == 1:
                    img = img.squeeze(-1)
                else:
                    img = img.mean(-1)
            return img
        obs = np.stack([_to_chw(s["obs_decoded"]) for s in states])
        if obs.ndim == 3:
            obs = obs[:, None, :, :]
        prev_obs = np.stack([_to_chw(s.get("prev_decoded", s["obs_decoded"])) for s in states])
        if prev_obs.ndim == 3:
            prev_obs = prev_obs[:, None, :, :]
        next_obs = np.stack([_to_chw(s.get("next_decoded", s["obs_decoded"])) for s in states])
        if next_obs.ndim == 3:
            next_obs = next_obs[:, None, :, :]
        n = obs.shape[0]
        obs_t = torch.from_numpy(obs).float()
        prev_obs_t = torch.from_numpy(prev_obs).float()
        next_obs_t = torch.from_numpy(next_obs).float()
        out[path.stem] = {}
        for model_name, vae in vaes.items():
            cfg = configs[model_name]
            ld = cfg["latent_dim"]
            delta_t = dt if dt is not None else cfg.get("delta_t", 0.02)
            use_vae = cfg.get("use_vae", True)
            z_all = torch.zeros(n, ld, dtype=torch.float32)
            z_dot_all = torch.zeros(n, ld, dtype=torch.float32)
            o_t = torch.from_numpy(obs).float().to(device)
            with torch.no_grad():
                for i in range(n):
                    o_cur = o_t[i : i + 1]
                    if use_vae:
                        _, _, mu, _ = vae(o_cur)
                    else:
                        _, mu = vae(o_cur)
                    z_all[i] = mu.cpu().squeeze(0)
            def _to_batch(img):
                img = np.asarray(img, dtype=np.float32)
                if img.ndim == 3:
                    if img.shape[2] == 1:
                        img = img.squeeze(-1)
                    else:
                        img = img.mean(-1)
                return torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0)
            for i in range(n):
                s = states[i]
                o_cur = o_t[i : i + 1]
                prev = s.get("prev_decoded")
                nxt = s.get("next_decoded")
                if prev is not None and nxt is not None:
                    prev_b = _to_batch(prev)
                    nxt_b = _to_batch(nxt)
                    o_dot = (nxt_b - prev_b) / (2.0 * delta_t)
                    print(f"o_dot: {o_dot.shape}")
                    print(f"o_cur: {o_cur.shape}")
                    z_dot_all[i] = vae.latent_velocity_from_observation_velocity(o_cur, o_dot).cpu().squeeze(0)
            out[path.stem][model_name] = {
                "z": z_all,
                "z_dot": z_dot_all,
                "obs": obs_t,
                "prev_obs": prev_obs_t,
                "next_obs": next_obs_t,
            }
    return out


def inverse_sigmoid(x, eps=1e-7):
    """Logit function: inverse of sigmoid.
    
    Maps values from [0, 1] to (-inf, inf).
    Used to initialize learnable parameters in logit space when using sigmoid activation.
    """
    x = torch.clamp(x, eps, 1 - eps)  # Avoid log(0) and log(1)
    return torch.log(x / (1 - x))


def optimize_open_loop_control(
    dyn,
    z0,
    zd0,
    zT,
    zdT,
    u_init,
    T,
    dt,
    n_iter=200,
    lr=1e-2,
    scheduler_step_size=100,
    scheduler_gamma=0.5,
    fix_u0_to_uref=True,
    use_sigmoid=True,
    z_ref_traj=None,
    zd_ref_traj=None,
    w_Q=1.0,
    w_dQ=1.0,
    w_Qf=0.0,
    w_dQf=0.0,
    w_Ql=None,
    w_dQl=None,
    w_Qfl=None,
    w_dQfl=None,
    w_R=1e-3,
    device=None,
    delta_u_max=None,
    w_du_bound=1.0,
    u_max=0.9,
    z_stats=None,
    waypoint_window="next",
):
    """
    Model-agnostic gradient-based open-loop control

    Use cases:
    - Point-to-point (e.g. rest → static): pass (zT, zdT) only; loss = path + terminal + control.
    - Full trajectory: pass (z_ref_traj, zd_ref_traj) with length T; loss = track full path + terminal.
    - Waypoints: pass (z_ref_traj, zd_ref_traj) with length K < T (e.g. K=10). Waypoints are
      evenly spaced over T steps (first at step 0, last at step T-1). Trajectory loss (w_Q, w_dQ) per
      step: "next" = target is the next waypoint in each subsection; "symmetric" = target is the
      waypoint whose window contains the step (half before / half after that waypoint). w_Qf/w_dQf
      penalize exact error at the K waypoint steps.

    dyn: dynamics with .forward(z, zd, u, dt), .latent_dim, .actuation_dim
    z0, zd0: initial state/velocity [latent_dim]
    zT, zdT: terminal target [latent_dim]. Used for trajectory+terminal in point-to-point; for terminal
             only when z_ref_traj has length T (full trajectory). Ignored when z_ref_traj has length < T
             (sparse waypoints). Pass e.g. z_ref_traj[-1], zd_ref_traj[-1] when using ref traj for consistency.
    u_init: reference control [act_dim]
    T: horizon
    dt: time step
    z_ref_traj, zd_ref_traj: optional [L, latent_dim]. If L==T: track full path. If L < T: waypoints
      (L evenly spaced over T steps); trajectory loss per subsection, w_Qf at waypoint steps.
    w_Q, w_dQ: weights on state/velocity error along path (scalar or [latent_dim])
    w_Qf, w_dQf: weights on final state/velocity error (terminal cost); 0 = disabled
    w_Ql, w_dQl: optional. Same as w_Q, w_dQ but applied only at the final waypoint. If None, not added.
    w_Qfl, w_dQfl: optional. Same as w_Qf, w_dQf but applied only at the final waypoint. If None, not added.
    w_R: weight on control rate (consecutive differences of u)
    delta_u_max: optional max allowed |Δu| per step per channel (scalar or [act_dim]). If set, an
                 additional loss penalizes only the excess when |Δu| > delta_u_max (smooth, differentiable).
    w_du_bound: weight on the delta_u excess penalty (only used when delta_u_max is provided).
    z_stats: optional dict {"z_mean": tensor [latent_dim], "z_std": tensor [latent_dim]}.
             If provided, state and velocity errors are scaled by a single scalar (mean of z_std)
             so relative latent structure is preserved and loss weights are comparable across models.
             Pass e.g. z_stats[model_name] from precomputed stats (e.g. from static validation latents).
    waypoint_window: "next" (default) or "symmetric". When z_ref_traj has length K < T. "next" = each
             step's target is the next waypoint; "symmetric" = each step's target is the waypoint
             at the center of its window (half the interval before and after that waypoint).
    """
    # CRITICAL: Ensure dynamics model is in eval mode and parameters don't compute gradients
    # This prevents wasting memory/compute on dynamics parameter gradients during control optimization
    dyn.eval()  # Set to eval mode (disables dropout, batch norm uses running stats)
    
    # Store and disable gradient computation for all dynamics parameters
    param_grad_states = {}
    for name, param in dyn.named_parameters():
        param_grad_states[name] = param.requires_grad
        param.requires_grad_(False)
    
    try:
        dev = device or z0.device
        z0 = z0.to(dev).flatten()
        zd0 = zd0.to(dev).flatten()
        zT = zT.to(dev).flatten()
        zdT = zdT.to(dev).flatten()
        u_ref = u_init.to(dev).flatten()  # Ensure u_ref is 1D

        # Optional: single scalar scale from z_std so models are balanced, relative structure intact
        z_scale = None
        if z_stats is not None:
            z_scale = z_stats["z_std"].to(dev).flatten().mean().clamp(min=1e-6)

        act_dim = dyn.actuation_dim
        u_max_t = torch.as_tensor(u_max, device=dev, dtype=u_ref.dtype)
        # When use_sigmoid=True, u = sigmoid(raw)*u_max, so raw = inverse_sigmoid(u/u_max)
        u_ref_normalized = (u_ref / u_max_t.clamp(min=1e-9)).clamp(1e-7, 1 - 1e-7) if use_sigmoid else u_ref

        # Initialize learnable control parameters
        # When use_sigmoid=True, initialize in logit space so sigmoid(raw)*u_max ≈ u_ref
        if fix_u0_to_uref and T > 1:
            if use_sigmoid:
                u_learnable_raw = (
                    inverse_sigmoid(u_ref_normalized).unsqueeze(0).repeat(T - 1, 1) + 0.01 * torch.randn(T - 1, act_dim, device=dev)
                ).requires_grad_(True)
            else:
                u_learnable_raw = (
                    u_ref.unsqueeze(0).repeat(T - 1, 1) + 0.01 * torch.randn(T - 1, act_dim, device=dev)
                ).requires_grad_(True)
        else:
            if use_sigmoid:
                u_learnable_raw = (
                    inverse_sigmoid(u_ref_normalized).unsqueeze(0).repeat(T, 1) + 0.01 * torch.randn(T, act_dim, device=dev)
                ).requires_grad_(True)
            else:
                u_learnable_raw = (
                    u_ref.unsqueeze(0).repeat(T, 1) + 0.01 * torch.randn(T, act_dim, device=dev)
                ).requires_grad_(True)


        optimizer = torch.optim.Adam([u_learnable_raw], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma
        )

        use_traj_target = z_ref_traj is not None
        sparse_waypoints = False
        waypoint_steps = None  # [K] int: step index for each waypoint
        z_ref_expanded = None
        zd_ref_expanded = None
        if use_traj_target:
            z_ref_traj = z_ref_traj.to(dev)
            zd_ref_traj = (
                zd_ref_traj.to(dev)
                if zd_ref_traj is not None
                else torch.zeros_like(z_ref_traj, device=dev)
            )
            K = z_ref_traj.shape[0]
            if K < T:
                # Sparse waypoints: evenly space K waypoints over T steps (first at 0, last at T-1)
                sparse_waypoints = True
                waypoint_steps = [
                    int(round(k * (T - 1) / (K - 1))) if K > 1 else 0
                    for k in range(K)
                ]
                if K == 1:
                    waypoint_steps[0] = 0
                # For each step t, assign target waypoint
                if waypoint_window == "symmetric":
                    # Symmetric: waypoint k is target for steps in [mid_{k-1,k}, mid_{k,k+1})
                    mid = [
                        (waypoint_steps[k] + waypoint_steps[k + 1]) / 2.0
                        for k in range(K - 1)
                    ]
                    boundaries = [0.0] + mid + [float(T)]
                    next_target = torch.zeros(T, dtype=torch.long, device=dev)
                    for t in range(T):
                        # k such that boundaries[k] <= t < boundaries[k+1]
                        k = np.searchsorted(boundaries, t, side="right") - 1
                        next_target[t] = max(0, min(k, K - 1))
                else:
                    # "next": target = next waypoint in subsection
                    next_target = torch.zeros(T, dtype=torch.long, device=dev)
                    for k in range(K - 1):
                        next_target[waypoint_steps[k] : waypoint_steps[k + 1]] = k + 1
                    next_target[waypoint_steps[K - 1] : T] = K - 1
                z_ref_expanded = z_ref_traj[next_target]   # [T, latent_dim]
                zd_ref_expanded = zd_ref_traj[next_target]

        loss_history = []
        
        # Calculate progress milestones (every 10%)
        progress_milestones = [int(n_iter * p / 100) for p in range(10, 101, 10)]

        for iter_idx in range(n_iter):
            optimizer.zero_grad()

            if use_sigmoid:
                u_learnable = torch.sigmoid(u_learnable_raw) * u_max_t
            else:
                u_learnable = u_learnable_raw


            if fix_u0_to_uref and T > 1:
                u_seq = torch.cat([u_ref.unsqueeze(0), u_learnable], dim=0)
            else:
                u_seq = u_learnable

            z_curr = z0.unsqueeze(0)
            zd_curr = zd0.unsqueeze(0)
            z_traj_list = []
            zd_traj_list = []
            u_traj_list = []

            for t in range(T):
                u_t = u_seq[t : t + 1]
                u_traj_list.append(u_t)
                z_next, zd_next = dyn.forward(z_curr, zd_curr, u_t, dt)
                z_curr, zd_curr = z_next, zd_next
                z_traj_list.append(z_curr)
                zd_traj_list.append(zd_curr)

            z_traj = torch.cat(z_traj_list, dim=0)  # [T, latent_dim]
            zd_traj = torch.cat(zd_traj_list, dim=0)
            u_traj = torch.cat(u_traj_list, dim=0)  # [T, act_dim]

            # Optional: scale all state/velocity errors by single z_scale (preserves relative structure)
            # Trajectory loss: penalize error to target along all steps
            # Use .mean() to make loss independent of trajectory length T
            if sparse_waypoints:
                err_z = z_traj - z_ref_expanded
                err_zd = zd_traj - zd_ref_expanded
            elif use_traj_target:
                err_z = z_traj - z_ref_traj
                err_zd = zd_traj - zd_ref_traj
            else:
                err_z = z_traj - zT.unsqueeze(0)
                err_zd = zd_traj - zdT.unsqueeze(0)
            if z_scale is not None:
                err_z = err_z / z_scale
                err_zd = err_zd / z_scale
            traj_z = (w_Q * err_z.pow(2)).mean()
            traj_zd = (w_dQ * err_zd.pow(2)).mean()
            loss = traj_z + traj_zd

            # Terminal / waypoint-exact cost
            if sparse_waypoints and (w_Qf > 0 or w_dQf > 0):
                # Exact hit at the K waypoint steps
                waypoint_steps_t = torch.tensor(waypoint_steps, device=dev, dtype=torch.long)
                z_at_waypoints = z_traj[waypoint_steps_t]   # [K, latent_dim]
                zd_at_waypoints = zd_traj[waypoint_steps_t]
                err_zf = z_at_waypoints - z_ref_traj
                err_zdf = zd_at_waypoints - zd_ref_traj
                if z_scale is not None:
                    err_zf = err_zf / z_scale
                    err_zdf = err_zdf / z_scale
                if w_Qf > 0:
                    loss = loss + w_Qf * err_zf.pow(2).sum()
                if w_dQf > 0:
                    loss = loss + w_dQf * err_zdf.pow(2).sum()
            elif w_Qf > 0 or w_dQf > 0:
                z_final = z_traj[-1]
                zd_final = zd_traj[-1]
                err_zf = z_final - zT
                err_zdf = zd_final - zdT
                if z_scale is not None:
                    err_zf = err_zf / z_scale
                    err_zdf = err_zdf / z_scale
                if w_Qf > 0:
                    loss = loss + w_Qf * err_zf.pow(2).sum()
                if w_dQf > 0:
                    loss = loss + w_dQf * err_zdf.pow(2).sum()

            # Optional: extra loss weights only at the final waypoint (w_Ql, w_dQl, w_Qfl, w_dQfl)
            if w_Ql is not None or w_dQl is not None or w_Qfl is not None or w_dQfl is not None:
                if sparse_waypoints:
                    step_last = waypoint_steps[-1]
                    z_final_wp = z_traj[step_last]
                    zd_final_wp = zd_traj[step_last]
                    z_target_last = z_ref_traj[-1]
                    zd_target_last = zd_ref_traj[-1]
                else:
                    z_final_wp = z_traj[-1]
                    zd_final_wp = zd_traj[-1]
                    z_target_last = z_ref_traj[-1] if use_traj_target else zT
                    zd_target_last = zd_ref_traj[-1] if use_traj_target else zdT
                err_z_final_wp = z_final_wp - z_target_last
                err_zd_final_wp = zd_final_wp - zd_target_last
                if z_scale is not None:
                    err_z_final_wp = err_z_final_wp / z_scale
                    err_zd_final_wp = err_zd_final_wp / z_scale
                if w_Ql is not None and w_Ql > 0:
                    loss = loss + w_Ql * err_z_final_wp.pow(2).sum()
                if w_dQl is not None and w_dQl > 0:
                    loss = loss + w_dQl * err_zd_final_wp.pow(2).sum()
                if w_Qfl is not None and w_Qfl > 0:
                    loss = loss + w_Qfl * err_z_final_wp.pow(2).sum()
                if w_dQfl is not None and w_dQfl > 0:
                    loss = loss + w_dQfl * err_zd_final_wp.pow(2).sum()

            # Control rate penalty (R on consecutive differences of u)
            # Use .mean() to make loss independent of trajectory length T
            if w_R > 0 and T > 1:
                du = (u_traj[1:] - u_traj[:-1]).pow(2).mean()
                loss = loss + w_R * du

            # Delta-u excess penalty: penalize only when |Δu| exceeds delta_u_max (per step, per channel).
            # No penalty below threshold; smooth relu-based penalty on the excess. Requires T > 1.
            if w_du_bound > 0 and delta_u_max is not None and T > 1:
                du = u_traj[1:] - u_traj[:-1]  # [T-1, act_dim]
                delta_u_max_t = torch.as_tensor(delta_u_max, device=du.device, dtype=du.dtype)
                if delta_u_max_t.ndim == 0:
                    delta_u_max_t = delta_u_max_t.reshape(1, 1).expand_as(du)
                elif delta_u_max_t.ndim == 1:
                    delta_u_max_t = delta_u_max_t.reshape(1, -1).expand_as(du)
                excess = torch.relu(du.abs() - delta_u_max_t)  # only positive when |Δu| > delta_u_max
                loss = loss + w_du_bound * excess.pow(2).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_history.append(loss.item())
            
            # Print progress at 10% milestones
            if (iter_idx + 1) in progress_milestones:
                progress_pct = ((progress_milestones.index(iter_idx + 1) + 1) * 10)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"  Iteration {iter_idx + 1}/{n_iter} ({progress_pct}%): Loss = {loss.item():.6f}, LR = {current_lr:.2e}")

        if fix_u0_to_uref and T > 1:
            u_seq_opt = torch.cat([u_ref.unsqueeze(0), u_learnable.detach()], dim=0)
        else:
            u_seq_opt = u_learnable.detach()

        # Final rollout for full trajectory (detached, for plotting)
        z_curr = z0.unsqueeze(0)
        zd_curr = zd0.unsqueeze(0)
        z_traj_final = [z0.unsqueeze(0)]  # Include initial state
        zd_traj_final = [zd0.unsqueeze(0)]
        
        for t in range(T):
            u_t = u_seq_opt[t : t + 1]
            z_next, zd_next = dyn.forward(z_curr, zd_curr, u_t, dt)
            z_curr, zd_curr = z_next, zd_next
            z_traj_final.append(z_curr)
            zd_traj_final.append(zd_curr)
        
        z_traj_full = torch.cat(z_traj_final, dim=0).detach()  # [T+1, latent_dim]
        zd_traj_full = torch.cat(zd_traj_final, dim=0).detach()  # [T+1, latent_dim]

        return {
            "u_seq": u_seq_opt,
            "z_traj": z_traj_full,  # [T+1, latent_dim]
            "zd_traj": zd_traj_full,  # [T+1, latent_dim]
            "loss_history": loss_history,
            "zT_pred": z_traj_full[-1],
            "zdT_pred": zd_traj_full[-1],
        }
    
    finally:
        # Restore original requires_grad state for all parameters
        for name, param in dyn.named_parameters():
            if name in param_grad_states:
                param.requires_grad_(param_grad_states[name])


def _result_to_numpy(result):
    """Convert a result dict from optimize_open_loop_control to numpy arrays (for saving)."""
    return {
        "z_traj": result["z_traj"].cpu().numpy(),
        "zd_traj": result["zd_traj"].cpu().numpy(),
        "u_seq": result["u_seq"].cpu().numpy(),
        "loss_history": np.array(result["loss_history"]),
        "zT_pred": result["zT_pred"].cpu().numpy(),
        "zdT_pred": result["zdT_pred"].cpu().numpy(),
    }




def save_control_output(
    model_name,
    target_file,
    result,
    simulator_pickle_paths=None,
    save_dir="results/control_outputs",
    name_suffix=None,
):
    """
    Save a single control optimization result to an npz file.

    model_name: e.g. '2 Seg Koopman + Attention'
    target_file: key/stem for the target (e.g. 'Koopman_Static_Extrapolated_1', 'Koopman_Dynamic_Normal_1')
    result: dict from optimize_open_loop_control (single run)
    simulator_pickle_paths: if given, used to resolve target_file to a file stem; otherwise target_file is used as stem.
    save_dir: directory for output npz files.
    name_suffix: optional string appended before .npz (e.g. "_01") to save multiple files for the same target
        (e.g. one per segment in multi-step). If None or "", filename is {safe_model}_{stem}.npz.
    """
    from pathlib import Path
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    stem = target_file
    safe_model = model_name.replace(" ", "_").replace("+", "plus")
    suffix = (name_suffix if name_suffix is not None else "") or ""
    fname = f"{safe_model}_{stem}{suffix}.npz"

    out = _result_to_numpy(result)
    np.savez_compressed(save_path / fname, **out)
    return str(save_path / fname)


# Plot rollout trajectory
import matplotlib.pyplot as plt

def plot_rollout(result, z0=None, zd0=None, zT=None, zdT=None, u_init=None, u_final=None, z_ref_traj=None, zd_ref_traj=None):
    """Plot optimized control trajectory in notebook style.

    result: dict from optimize_open_loop_control with keys 'z_traj', 'zd_traj', 'u_seq'
    z0, zd0: optional initial states for markers (if None, uses first element of trajectories)
    zT, zdT: optional target states for markers and dashed lines
    u_init, u_final: optional initial/final controls for markers
    z_ref_traj, zd_ref_traj: optional reference trajectory. If length T+1 or T, plotted as full path.
      If length K < T (sparse waypoints), waypoints are plotted at evenly spaced step indices.
    """
    z_traj = result["z_traj"].cpu()  # [T+1, latent_dim]
    zd_traj = result["zd_traj"].cpu()  # [T+1, latent_dim]
    u_seq = result["u_seq"].cpu()  # [T, act_dim]
    
    T = u_seq.shape[0]
    steps = np.arange(T + 1)
    steps_u = np.arange(T)
    dt = 0.02  # 50 Hz, user specified

    # Resolve ref trajectory step indices and values for plotting (z and zd)
    def ref_steps_and_vals(ref, name="z"):
        if ref is None:
            return None, None
        ref = ref.cpu() if torch.is_tensor(ref) else torch.tensor(ref)
        L = ref.shape[0]
        if L >= T:
            step_idx = np.arange(min(L, T + 1))
            return step_idx, ref[: len(step_idx)]
        # Sparse waypoints: same spacing as in optimize_open_loop_control
        K = L
        waypoint_steps = [
            int(round(k * (T - 1) / (K - 1))) if K > 1 else 0
            for k in range(K)
        ]
        if K == 1:
            waypoint_steps[0] = 0
        return np.array(waypoint_steps), ref

    # Use provided initial states or extract from trajectory
    if z0 is None:
        z0_plot = z_traj[0]
    else:
        z0_plot = z0.cpu() if torch.is_tensor(z0) else torch.tensor(z0)

    if zd0 is None:
        zd0_plot = zd_traj[0]
    else:
        zd0_plot = zd0.cpu() if torch.is_tensor(zd0) else torch.tensor(zd0)

    # Extract final states from trajectory
    z_final_plot = z_traj[-1]
    zd_final_plot = zd_traj[-1]

    # Use provided targets or final trajectory values
    if zT is not None:
        zT_plot = zT.cpu() if torch.is_tensor(zT) else torch.tensor(zT)
    else:
        zT_plot = z_final_plot

    if zdT is not None:
        zdT_plot = zdT.cpu() if torch.is_tensor(zdT) else torch.tensor(zdT)
    else:
        zdT_plot = zd_final_plot

    # Use provided controls or extract from sequence
    # Track if explicitly provided for plotting decisions
    u_init_provided = u_init is not None
    u_final_provided = u_final is not None

    if u_init is None:
        u_init_plot = u_seq[0]
    else:
        u_init_plot = u_init.cpu() if torch.is_tensor(u_init) else torch.tensor(u_init)

    if u_final is None:
        u_final_plot = u_seq[-1]
    else:
        u_final_plot = u_final.cpu() if torch.is_tensor(u_final) else torch.tensor(u_final)

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    colors_z = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # Latent z: predicted (solid), true init/final (markers), final line (dashed)
    for d in range(z_traj.shape[1]):
        c = colors_z[d % len(colors_z)]
        axes[0].plot(steps, z_traj[:, d].numpy(), color=c, alpha=0.8)
        # Initial state as open circle marker
        axes[0].plot(0, z0_plot[d].item(), marker='o', color=c, markerfacecolor='none', 
                    markersize=8, label=f"z_real init {d}" if d == 0 else None)
        # Final state as X marker
        axes[0].plot(T, zT_plot[d].item(), marker='X', color=c, markersize=8, 
                    label=f"z_real final {d}" if d == 0 else None)
        # Horizontal dashed line at final value
        axes[0].axhline(y=zT_plot[d].item(), color=c, linestyle=':', linewidth=1, alpha=0.7)

    z_ref_steps, z_ref_vals = ref_steps_and_vals(z_ref_traj, "z")
    if z_ref_steps is not None and z_ref_vals is not None:
        z_ref_np = z_ref_vals.numpy()
        # Like z_dot: dashed line always; scatter markers only at waypoints when ref is sparse (not full-length)
        z_ref_is_sparse = len(z_ref_steps) < T + 1
        for d in range(z_ref_np.shape[1]):
            c = colors_z[d % len(colors_z)]
            axes[0].plot(z_ref_steps, z_ref_np[:, d], '--', color=c, alpha=0.4, linewidth=1)
            if z_ref_is_sparse:
                axes[0].scatter(z_ref_steps, z_ref_np[:, d], marker='s', s=28, color=c, alpha=0.9, zorder=5, label="waypoints" if d == 0 else None)

    axes[0].set_title("Latent z: predicted (solid), true init/final (markers), final line (dashed)")
    axes[0].set_ylabel("value")
    axes[0].legend()

    # Latent z_dot: predicted (solid), true init (circle), target final=0 (X, dashed line)
    for d in range(zd_traj.shape[1]):
        c = colors_z[d % len(colors_z)]
        axes[1].plot(steps, zd_traj[:, d].numpy(), color=c, alpha=0.8)
        # Initial dataset z_dot as open circle
        axes[1].plot(0, zd0_plot[d].item(), marker='o', color=c, markerfacecolor='none', 
                    markersize=8, label=f"z_dot_real init {d}" if d == 0 else None)
        # Final/target z_dot as X marker
        axes[1].plot(T, zdT_plot[d].item(), marker='X', color=c, markersize=8, 
                    label=f"z_dot_real final (always 0) {d}" if d == 0 else None)
        # Horizontal dashed line at target (usually 0)
        axes[1].axhline(y=zdT_plot[d].item(), color=c, linestyle=':', linewidth=1, alpha=0.7)

    zd_ref_steps, zd_ref_vals = ref_steps_and_vals(zd_ref_traj, "zd")
    if zd_ref_steps is not None and zd_ref_vals is not None:
        zd_ref_np = zd_ref_vals.numpy()
        zd_ref_is_sparse = len(zd_ref_steps) < T + 1
        for d in range(zd_ref_np.shape[1]):
            c = colors_z[d % len(colors_z)]
            axes[1].plot(zd_ref_steps, zd_ref_np[:, d], '--', color=c, alpha=0.4, linewidth=1)
            if zd_ref_is_sparse:
                axes[1].scatter(zd_ref_steps, zd_ref_np[:, d], marker='s', s=28, color=c, alpha=0.9, zorder=5, label="waypoints" if d == 0 else None)

    axes[1].set_title("Latent z_dot: predicted (solid), true init (circle), target final=0 (X, dashed line)")
    axes[1].set_ylabel("value")
    axes[1].legend()

    # ---- Shade regions where *any* u_derivative exceeds 2 bar/s in axis[2] ----
    # u is [T, act_dim], steps_u is [T]
    u_np = u_seq.numpy()
    du_dt = np.diff(u_np, axis=0) / dt  # [T-1, act_dim], units: bar/s
    du_dt_exceeds = np.any(np.abs(du_dt) > 2, axis=1)  # [T-1], is True wherever any act_dim exceeds thresh

    # Find contiguous segments where du/dt exceeds limit
    from itertools import groupby
    from operator import itemgetter

    exceeding_idxs = np.where(du_dt_exceeds)[0]  # these are base indices into steps_u (interval [i, i+1])
    # Group contiguous indices for shading
    shaded_regions = []
    for k, g in groupby(enumerate(exceeding_idxs), lambda ix: ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        # shade region from group[0] to group[-1]+1 in time (since diff is between [i] and [i+1])
        shaded_regions.append((group[0], group[-1]+1))

    # Control u: predicted (solid), optionally real init/final (markers) if provided
    colors_u = colors_z
    for i in range(u_seq.shape[1]):
        c = colors_u[i % len(colors_u)]
        axes[2].plot(steps_u, u_seq[:, i].numpy(), color=c, alpha=0.8)
        # Initial control marker (only if explicitly provided)
        if u_init_provided:
            axes[2].plot(0, u_init_plot[i].item(), marker='o', color=c, markerfacecolor='none', 
                        markersize=8, label=f"u_real init {i}" if i == 0 else None)
        # Final control marker and dashed line (only if explicitly provided)
        if u_final_provided:
            axes[2].plot(T - 1, u_final_plot[i].item(), marker='X', color=c, markersize=8, 
                        label=f"u_real final {i}" if i == 0 else None)
            axes[2].axhline(y=u_final_plot[i].item(), color=c, linestyle=':', linewidth=1, alpha=0.7)

    # Shade all time regions where du/dt exceeds 2 bar/s (across any control dim) in red
    y_min, y_max = axes[2].get_ylim()
    for start_idx, end_idx in shaded_regions:
        axes[2].axvspan(start_idx, end_idx, color='red', alpha=0.2, zorder=0)
    axes[2].set_ylim(y_min, y_max)

    title_suffix = ", real init/final (markers)" if (u_init_provided and u_final_provided) else ""
    axes[2].set_title(f"Control u: predicted (solid){title_suffix}")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("u")
    axes[2].legend()

    plt.tight_layout()
    return fig

# Usage example:
# plot_rollout(result_static, zT=zT, zdT=zdT, dt=dt)