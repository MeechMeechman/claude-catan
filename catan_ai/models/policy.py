"""
Complete policy network for Catan AI.

Combines the cross-dimensional encoder with a flat action head (with masking)
and a value function head. Uses the structured encoder for board understanding
but outputs flat action logits matching catanatron's action space for simplicity
and correctness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

from catan_ai.models.encoder import CrossDimEncoder


class CatanPolicy(nn.Module):
    """Actor-critic policy for Catan.

    The policy network uses:
    - CrossDimEncoder for structured board understanding
    - Flat action logits with action masking for the policy (actor)
    - MLP value head for the critic

    This pragmatic design gets the benefit of structured observation encoding
    while keeping the action space handling simple and correct.
    """

    def __init__(
        self,
        hex_input_dim: int,
        vertex_input_dim: int,
        edge_input_dim: int,
        player_input_dim: int,
        action_space_size: int,
        num_tiles: int = 19,
        num_nodes: int = 54,
        num_edges: int = 72,
        hidden_dim: int = 128,
        encoder_layers: int = 4,
        encoder_output_dim: int = 256,
    ):
        super().__init__()

        self.action_space_size = action_space_size
        self.num_tiles = num_tiles
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        # Encoder
        self.encoder = CrossDimEncoder(
            hex_input_dim=hex_input_dim,
            vertex_input_dim=vertex_input_dim,
            edge_input_dim=edge_input_dim,
            player_input_dim=player_input_dim,
            hidden_dim=hidden_dim,
            num_xdim_layers=encoder_layers,
            output_dim=encoder_output_dim,
        )

        # Actor: flat logits over action space with masking
        self.actor = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_space_size),
        )

        # Critic: state value
        self.critic = nn.Sequential(
            nn.Linear(encoder_output_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for policy and value output layers
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def _encode(
        self,
        obs: Dict[str, torch.Tensor],
        vertex_to_hex: torch.Tensor,
        hex_to_vertex: torch.Tensor,
        vertex_adj: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(
            obs["hex_features"],
            obs["vertex_features"],
            obs["edge_features"],
            obs["player_features"],
            vertex_to_hex,
            hex_to_vertex,
            vertex_adj,
        )

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        vertex_to_hex: torch.Tensor,
        hex_to_vertex: torch.Tensor,
        vertex_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass returning action logits and value.

        Args:
            obs: dict with hex_features, vertex_features, edge_features,
                 player_features, action_mask
            vertex_to_hex, hex_to_vertex, vertex_adj: static adjacency tensors
        Returns:
            logits: (B, action_space_size) masked logits
            value: (B, 1)
        """
        state_emb = self._encode(obs, vertex_to_hex, hex_to_vertex, vertex_adj)

        # Actor
        logits = self.actor(state_emb)
        action_mask = obs["action_mask"].bool()
        logits = logits.masked_fill(~action_mask, float("-inf"))

        # Critic
        value = self.critic(state_emb)

        return logits, value

    def get_action_and_value(
        self,
        obs: Dict[str, torch.Tensor],
        vertex_to_hex: torch.Tensor,
        hex_to_vertex: torch.Tensor,
        vertex_adj: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log_prob, entropy, and value for PPO.

        Args:
            obs: structured observation dict
            action: if provided, compute log_prob of this action instead of sampling
            deterministic: if True, take argmax instead of sampling
        Returns:
            action: (B,) int64
            log_prob: (B,)
            entropy: (B,)
            value: (B,)
        """
        logits, value = self.forward(obs, vertex_to_hex, hex_to_vertex, vertex_adj)
        value = value.squeeze(-1)

        # Handle case where all actions are masked (shouldn't happen normally)
        all_masked = logits.isinf() & (logits < 0)
        if all_masked.all(dim=-1).any():
            # Fallback: uniform over all actions
            safe_logits = torch.zeros_like(logits)
        else:
            safe_logits = logits

        probs = F.softmax(safe_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        if action is None:
            if deterministic:
                action = safe_logits.argmax(dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(
        self,
        obs: Dict[str, torch.Tensor],
        vertex_to_hex: torch.Tensor,
        hex_to_vertex: torch.Tensor,
        vertex_adj: torch.Tensor,
    ) -> torch.Tensor:
        state_emb = self._encode(obs, vertex_to_hex, hex_to_vertex, vertex_adj)
        return self.critic(state_emb).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PolicyPlayer:
    """Wraps CatanPolicy as a catanatron Player for self-play and evaluation."""

    def __init__(self, color, policy: CatanPolicy, env_ref, device="cpu", deterministic=False):
        from catanatron.models.player import Player
        self._player = Player(color)
        self.color = color
        self.policy = policy
        self.env_ref = env_ref  # reference to CatanTrainEnv for obs extraction
        self.device = device
        self.deterministic = deterministic
        self.is_bot = True

    def decide(self, game, playable_actions):
        """Interface for catanatron's Player.decide()."""
        # Temporarily set the game on env_ref to extract observations
        old_game = self.env_ref.game
        self.env_ref.game = game
        obs = self.env_ref._get_obs()
        self.env_ref.game = old_game

        # Convert to tensors
        obs_t = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.device) for k, v in obs.items()}
        # action_mask needs to be bool
        obs_t["action_mask"] = obs_t["action_mask"].bool()

        adj = self._get_adj_tensors()

        with torch.no_grad():
            action, _, _, _ = self.policy.get_action_and_value(
                obs_t, *adj, deterministic=self.deterministic
            )

        action_idx = action.item()
        from catanatron.gym.envs.action_space import from_action_space
        catan_action = from_action_space(
            action_idx, self.color, self.env_ref.player_colors, self.env_ref.map_type
        )

        # Fallback if action is invalid
        if catan_action not in playable_actions:
            import random
            return random.choice(playable_actions)

        return catan_action

    def _get_adj_tensors(self):
        d = self.device
        return (
            torch.tensor(self.env_ref.vertex_to_hex_adj, dtype=torch.long).to(d),
            torch.tensor(self.env_ref.hex_to_vertex_adj, dtype=torch.long).to(d),
            torch.tensor(self.env_ref.vertex_adj, dtype=torch.long).to(d),
        )

    def reset_state(self):
        self._player.reset_state()
