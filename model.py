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

# Step 19 - covariance_loss (not yet solved)
# TODO: implement

# Step 20 - vicreg_regularizer (not yet solved)
# TODO: implement

# Step 21 - jepa_loss (not yet solved)
# TODO: implement

# Step 22 - collapse_metric (not yet solved)
# TODO: implement

# Step 23 - jepa_training_step (not yet solved)
# TODO: implement

# Step 24 - train_jepa (not yet solved)
# TODO: implement

# Step 25 - rollout_latent_dynamics (not yet solved)
# TODO: implement

# Step 26 - multi_step_prediction_error (not yet solved)
# TODO: implement

# Step 27 - init_linear_probe (not yet solved)
# TODO: implement

# Step 28 - train_linear_probe (not yet solved)
# TODO: implement

# Step 29 - probe_state_recovery (not yet solved)
# TODO: implement

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

