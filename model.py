"""
JEPA World Model from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_env_state
def init_env_state(room_size: int = 8, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)

    return torch.randint(0, room_size, (2,)).float()

# Step 2 - apply_action
def apply_action(state: torch.Tensor, action: int, room_size: int = 8) -> torch.Tensor:
    # TODO: Apply a discrete action to the agent state with wall clamping.
    deltas = {
        0: (0.0, -1.0),
        1: (0.0, 1.0),
        2: (-1.0, 0.0),
        3: (1.0, 0.0),
    }
    
    dx, dy = deltas[int(action)]
    delta = torch.tensor([dx, dy], dtype = state.dtype, device = state.device)

    next_state = state + delta
    return torch.clamp(next_state, 0, room_size - 1)

# Step 3 - render_observation
def render_observation(state: torch.Tensor, room_size: int = 8) -> torch.Tensor:
    # TODO: Render a (1, room_size, room_size) float32 obs with agent pixel 1.0...
    obs = torch.zeros(
        (1, room_size, room_size), dtype=torch.float32, device=state.device
    )

    x = int(state[0].item())
    y = int(state[1].item())

    obs[0, y, x] = 1.0
    return obs

# Step 4 - env_reset
def env_reset(room_size: int = 8, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Reset the environment by sampling a new state and its observation.
    state = init_env_state(room_size=room_size, seed=seed)
    observation = render_observation(state=state, room_size=room_size)
    return state, observation

# Step 5 - env_step
def env_step(state: torch.Tensor, action: int, room_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    # TODO: Advance the env by one action; return (next_state, next_observation).
    next_state = apply_action(state, action, room_size)

    next_obs = render_observation(next_state, room_size)
    return next_state, next_obs

# Step 6 - collect_random_transitions
import torch


def collect_random_transitions(num_transitions: int, room_size: int = 8, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    
    state, obs = env_reset(room_size, seed)
    
    obs_sample = obs.unsqueeze(0) if obs.ndim == 2 else obs

    observations = torch.empty((num_transitions, *obs_sample.shape), dtype=obs.dtype, device=obs.device)
    next_observations = torch.empty((num_transitions, *obs_sample.shape), dtype=obs.dtype, device=obs.device)
    actions = torch.empty((num_transitions,), dtype=torch.long, device=obs.device)
    states = torch.empty((num_transitions, *state.shape), dtype=state.dtype, device=state.device)
    next_states = torch.empty((num_transitions, *state.shape), dtype=state.dtype, device=state.device)

    for i in range(num_transitions):    
        action = torch.randint(0, 4, ())
        next_state, next_obs = env_step(state, action.item(), room_size)
        
        observations[i] = obs.unsqueeze(0) if obs.ndim == 2 else obs
        actions[i] = action
        next_observations[i] = next_obs.unsqueeze(0) if next_obs.ndim == 2 else next_obs
        states[i] = state
        next_states[i] = next_state

        state, obs = next_state, next_obs
    
    return {
        'observations': observations,
        'actions': actions,
        'next_observations': next_observations,
        'states': states,
        'next_states': next_states,
    }

# Step 7 - build_transition_dataset
def build_transition_dataset(num_transitions: int = 512, room_size: int = 8, seed: int = 0) -> dict:
    # TODO: Build a JEPA training-ready transition dataset by collecting random transitions...
    return collect_random_transitions(num_transitions, room_size=room_size, seed=seed)

# Step 8 - init_encoder_params
def init_encoder_params(obs_channels: int = 1, room_size: int = 8, latent_dim: int = 32, seed: int = 0) -> dict:
    # TODO: Initialize parameters of a small CNN encoder that maps pixel observations to latent embeddings.
   
    torch.manual_seed(seed)

    h1 = (room_size + 2 * 1 - 3) + 1
    h2 = (h1 + 2 * 1 - 3) // 2 + 1
    fc_in = 32 * h2 * h2
    
    def create_weight(shape):
        return (torch.randn(shape) * 0.1).requires_grad_(True)

    def create_bias(shape):
        return torch.zeros(shape, requires_grad=True)

    params = {
        'conv1_w': create_weight((16, obs_channels, 3, 3)),
        'conv1_b': create_bias((16,)),
        'conv2_w': create_weight((32, 16, 3, 3)),
        'conv2_b': create_bias((32,)),
        'fc_w': create_weight((latent_dim, fc_in)),
        'fc_b': create_bias((latent_dim,)),
    }

    return params

# Step 9 - encoder_forward
def encoder_forward(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Run a small two-layer CNN that maps pixel observations to latent embeddings.
    x = F.conv2d(
        obs,
        encoder_params['conv1_w'],
        encoder_params['conv1_b'],
        stride=1,
        padding=1,
    )
    x = F.relu(x)

    x = F.conv2d(
        x,
        encoder_params['conv2_w'],
        encoder_params['conv2_b'],
        stride=2,
        padding=1,
    )
    x = F.relu(x)

    x = x.flatten(start_dim=1)

    out = F.linear(
        x,
        encoder_params['fc_w'],
        encoder_params['fc_b'],
    )

    return out

# Step 10 - init_target_encoder
def init_target_encoder(encoder_params: dict) -> dict:
    # TODO: Create the EMA target encoder by deep-copying the online encoder params.
    target_params = {
        key: value.detach().clone()
        for key, value in encoder_params.items()
    }
    
    return target_params

# Step 11 - ema_update
def ema_update(target_params: dict, encoder_params: dict, tau: float = 0.99) -> dict:
    # TODO: Refresh target encoder params via EMA of the online encoder.
    
    updated_params = {}

    with torch.no_grad():
        for key in target_params:
            target_val = target_params[key]
            online_val = encoder_params[key]
            updated_params[key] = tau * target_val + (1.0 - tau) * online_val

    return updated_params

# Step 12 - encode_batch
def encode_batch(obs: torch.Tensor, encoder_params: dict) -> torch.Tensor:
    # TODO: Batch-encode observations into latent embeddings using encoder params.
    return encoder_forward(obs, encoder_params)

# Step 13 - init_predictor_params
def init_predictor_params(latent_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64, seed: int = 0) -> dict:
    torch.manual_seed(seed)

    def create_weight(shape):
        return (torch.randn(shape) * 0.02).requires_grad_(True)

    def create_bias(shape):
        return torch.zeros(shape, requires_grad=True)

    params = {
        'action_embed_w': create_weight((action_dim, latent_dim)),
        'fc1_w': create_weight((hidden_dim, 2 * latent_dim)),  # 2 * latent_dim due to concatenation [z; a_embed]
        'fc1_b': create_bias((hidden_dim,)),
        'fc2_w': create_weight((latent_dim, hidden_dim)),
        'fc2_b': create_bias((latent_dim,)),
    }

    return params

# Step 14 - embed_action
def embed_action(actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Embed discrete actions into continuous vectors via a learned action embedding table.
    embed_weight = predictor_params['action_embed_w']
    
    return embed_weight[actions]

# Step 15 - predictor_forward
import torch.nn.functional as F
def predictor_forward(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Implement predictor_forward, the forward pass of the action-conditioned dynamics predictor.
    action_emb = embed_action(actions, predictor_params)

    x = torch.cat([embeddings, action_emb], dim=-1)

    h = F.linear(x, predictor_params['fc1_w'], predictor_params['fc1_b'])
    h = F.relu(h)

    pred_next_embeddings = F.linear(
        h, predictor_params['fc2_w'], predictor_params['fc2_b']
    )

    return pred_next_embeddings

# Step 16 - predict_next_embedding
def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Predict the next latent embedding given current embeddings and actions...
    return predictor_forward(embeddings, actions, predictor_params)

# Step 17 - prediction_loss
def prediction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # TODO: Compute the JEPA prediction loss as mean squared error...
    return F.mse_loss(predicted, target)

# Step 18 - variance_loss
def variance_loss(embeddings: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # TODO: Compute VICReg variance hinge loss encouraging each dim std >= gamma.
    var = embeddings.var(dim=0, unbiased=True)

    std = torch.sqrt(var + eps)
    loss = torch.mean(torch.relu(gamma - std))

    return loss

# Step 19 - covariance_loss
def covariance_loss(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Implement `covariance_loss` to compute the VICReg covariance regularization from embeddings.
    b, d = embeddings.shape

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)

    cov = (centered.T @ centered) / (b - 1)
    
    cov_sq_sum = cov.pow(2).sum()
    diag_sq_sum = cov.diagonal().pow(2).sum()
    off_diag_sum = cov_sq_sum - diag_sq_sum

    return off_diag_sum / d

# Step 20 - vicreg_regularizer
def vicreg_regularizer(embeddings: torch.Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, gamma: float = 1.0) -> torch.Tensor:
    l_var = variance_loss(embeddings, gamma=gamma)

    l_cov = covariance_loss(embeddings)
    return var_weight * l_var + cov_weight * l_cov

# Step 21 - jepa_loss
def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    # TODO: Compose the full JEPA objective from prediction and VICReg terms...
    return pred_weight * prediction_loss(predicted, target) + var_weight * vicreg_regularizer(online_embeddings)

# Step 22 - collapse_metric
def collapse_metric(embeddings: torch.Tensor) -> torch.Tensor:
    # TODO: Measure collapse as the mean of per-dimension batch stds
    per_dim_std = embeddings.std(dim=0)

    return per_dim_std.mean()

# Step 23 - jepa_training_step
def jepa_training_step(batch: dict, encoder_params: dict, target_params: dict, predictor_params: dict, lr: float = 1e-3, tau: float = 0.99) -> tuple[dict, dict, dict, float, float]:
    # TODO: Perform one full JEPA gradient update on a batch of transitions
    obs = batch['observations']
    actions = batch['actions']
    next_obs = batch['next_observations']

    online_encoder = {
        k: v.detach().requires_grad_(True) for k, v in encoder_params.items()
    }
    online_predictor = {
        k: v.detach().requires_grad_(True) for k, v in predictor_params.items()
    }

    online_z = encode_batch(obs, online_encoder)

    with torch.no_grad():
        target_z = encode_batch(next_obs, target_params)

    predicted_z = predict_next_embedding(online_z, actions, online_predictor)

    loss = jepa_loss(predicted_z, target_z, online_z)

    loss.backward()

    def update_params_sgd(params: dict) -> dict:
        new_params = {}
        for k, p in params.items():
            if p.grad is not None:
                p_updated = (p - lr * p.grad).detach().requires_grad_(True)
            else:
                p_updated = p.detach().requires_grad_(True)
            new_params[k] = p_updated
        return new_params

    updated_encoder_params = update_params_sgd(online_encoder)
    updated_predictor_params = update_params_sgd(online_predictor)

    updated_target_params = ema_update(
        target_params, updated_encoder_params, tau=tau
    )

    loss_val = float(loss.item())
    collapse_val = float(collapse_metric(online_z.detach()).item())

    return (
        updated_encoder_params,
        updated_target_params,
        updated_predictor_params,
        loss_val,
        collapse_val,
    )

# Step 24 - train_jepa
def train_jepa(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, num_steps: int = 50, batch_size: int = 32, lr: float = 1e-3, tau: float = 0.99, seed: int = 0) -> tuple[dict, dict, dict, list]:
    # TODO: Train JEPA for num_steps with collapse monitoring, return params + history
    torch.manual_seed(seed)
  
    n_samples = dataset["observations"].shape[0]
    history = []

    for _ in range(num_steps):
        indices = torch.randint(0, n_samples, (batch_size,))

        batch = {
            "observations": dataset["observations"][indices],
            "actions": dataset["actions"][indices],
            "next_observations": dataset["next_observations"][indices],
        }

        (
            encoder_params,
            target_params,
            predictor_params,
            loss_value,
            collapse_value,
        ) = jepa_training_step(
            batch,
            encoder_params,
            target_params,
            predictor_params,
            lr=lr,
            tau=tau,
        )

        history.append({
            "loss": loss_value,
            "collapse": collapse_value,
        })

    return encoder_params, target_params, predictor_params, history

# Step 25 - rollout_latent_dynamics
def rollout_latent_dynamics(initial_embedding: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    # TODO: Roll out multi-step latent dynamics via the action-conditioned predictor...
    unbatched = initial_embedding.dim() == 1

    if unbatched:
        current = initial_embedding.unsqueeze(0)
    else:
        current = initial_embedding

    b = current.shape[0]

    if actions.dim() == 1:
        actions_batched = actions.unsqueeze(0).expand(b, -1)
    else:
        actions_batched = actions

    t_steps = actions_batched.shape[1]

    trajectory = [current]

    for t in range(t_steps):
        a_t = actions_batched[:, t]
        current = predict_next_embedding(current, a_t, predictor_params)
        trajectory.append(current)

    stacked = torch.stack(trajectory, dim=0)

    if unbatched:
        return stacked.squeeze(1)

    return stacked

# Step 26 - multi_step_prediction_error
def multi_step_prediction_error(dataset: dict, encoder_params: dict, target_params: dict, predictor_params: dict, horizon: int = 5, num_samples: int = 32) -> float:
    # TODO: Evaluate multi-step latent prediction accuracy via mean MSE over samples
    obs_all = dataset['observations']
    act_all = dataset['actions']
    next_obs_all = dataset['next_observations']


    n_transitions = obs_all.shape[0]
    num_valid = min(num_samples, n_transitions - horizon)
    if num_valid <= 0:
        return 0.0

    with torch.no_grad():
        start_obs = obs_all[:num_valid]

        action_seqs = torch.stack(
            [act_all[i : i + horizon] for i in range(num_valid)],
            dim=0,
        )

        future_obs = torch.stack(
            [next_obs_all[i : i + horizon] for i in range(num_valid)],
            dim=1,
        )

        z_0 = encode_batch(start_obs, encoder_params)

        pred_traj = rollout_latent_dynamics(z_0, action_seqs, predictor_params)
        pred_future_latents = pred_traj[1:]

        h_steps, b_size, c, h, w = future_obs.shape
        future_obs_flat = future_obs.reshape(h_steps * b_size, c, h, w)
        target_latents_flat = encode_batch(future_obs_flat, target_params)
        target_future_latents = target_latents_flat.reshape(
            h_steps, b_size, -1
        )

        mse = torch.mean((pred_future_latents - target_future_latents) ** 2)

    return mse.item()

# Step 27 - init_linear_probe
def init_linear_probe(latent_dim: int = 32, state_dim: int = 2, seed: int = 0) -> dict:
    # TODO: Initialize a linear probe that maps latent embeddings to true agent state (x, y).
    w = torch.randn(state_dim, latent_dim) * 0.01
    b = torch.zeros(state_dim)

    return {
        'w': w,
        'b': b,
    }

# Step 28 - train_linear_probe
def train_linear_probe(embeddings: torch.Tensor, states: torch.Tensor, probe_params: dict, num_steps: int = 100, lr: float = 1e-2) -> dict:
    # TODO: Train the linear probe via MSE regression from frozen embeddings to true agent states.
    z = embeddings.detach()
    s = states.detach()

    w = probe_params["w"].detach().clone().requires_grad_(True)
    b = probe_params["b"].detach().clone().requires_grad_(True)

    for _ in range(num_steps):
        pred = z @ w.T + b
        loss = torch.mean((pred - s) ** 2)

        loss.backward()

        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad

            # Reset gradients for the next iteration
            w.grad = None
            b.grad = None

    return {
        "w": w.detach().clone(),
        "b": b.detach().clone(),
    }

# Step 29 - probe_state_recovery
def probe_state_recovery(dataset: dict, encoder_params: dict, probe_params: dict | None = None, num_probe_steps: int = 100) -> dict:
    # TODO: Encode observations, train linear probe, report state recovery metrics
    observations = dataset['observations']
    states = dataset['states'].float()

    embeddings = encode_batch(observations, encoder_params)
    latent_dim = embeddings.shape[-1]
    state_dim = states.shape[-1]

    if probe_params is None:
        probe_params = init_linear_probe(
            latent_dim=latent_dim, state_dim=state_dim, seed=0
        )

    trained_probe = train_linear_probe(
        embeddings, states, probe_params, num_steps=num_probe_steps
    )

    w = trained_probe['w']
    b = trained_probe['b']
    pred = embeddings @ w + b

    mse = float(torch.mean((pred - states) ** 2).item())
    mean_abs_error = float(torch.mean(torch.abs(pred - states)).item())

    return {
        'mse': mse,
        'mean_abs_error': mean_abs_error,
        'probe_params': trained_probe,
    }

# Step 30 - encode_goal (not yet solved)
# TODO: implement

# Step 31 - latent_cost (not yet solved)
# TODO: implement

# Step 32 - sample_action_sequences (not yet solved)
# TODO: implement

# Step 33 - score_action_sequences (not yet solved)
# TODO: implement

# Step 34 - select_best_plan (not yet solved)
# TODO: implement

# Step 35 - mpc_step (not yet solved)
# TODO: implement

# Step 36 - run_mpc_episode (not yet solved)
# TODO: implement

# Step 37 - evaluate_planner (not yet solved)
# TODO: implement

# Step 38 - jepa_world_model_experiment (not yet solved)
# TODO: implement

