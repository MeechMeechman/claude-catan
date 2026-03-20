"""
Hierarchical autoregressive action heads for Catan.

Following Charlesworth (2021) and AlphaStar, the action space is
decomposed into sequential sub-decisions:
    action_type -> location/parameter

Each sub-head is conditioned on the shared state embedding plus
autoregressive context from prior sub-decisions. Action masking
applies independently at each head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ActionTypeHead(nn.Module):
    """Selects the high-level action type.

    Action types (mapping to catanatron ActionType):
        0: ROLL
        1: END_TURN
        2: BUILD_SETTLEMENT (needs vertex)
        3: BUILD_CITY (needs vertex)
        4: BUILD_ROAD (needs edge)
        5: BUY_DEVELOPMENT_CARD
        6: PLAY_KNIGHT_CARD
        7: PLAY_YEAR_OF_PLENTY (needs resource selection)
        8: PLAY_MONOPOLY (needs resource selection)
        9: PLAY_ROAD_BUILDING
        10: MARITIME_TRADE (needs trade params)
        11: MOVE_ROBBER (needs hex + player)
        12: DISCARD
    """
    NUM_TYPES = 13

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_TYPES),
        )
        self.embed = nn.Embedding(self.NUM_TYPES, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, state_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            state_emb: (B, input_dim)
            mask: (B, NUM_TYPES) bool, True = valid
        Returns:
            logits: (B, NUM_TYPES)
            action_type_emb: (B, hidden_dim) embedding of sampled/argmax type
        """
        logits = self.net(state_emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits

    def get_embedding(self, action_type_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(action_type_idx)


class VertexHead(nn.Module):
    """Selects a vertex (node) on the board. Used for settlement/city placement."""

    def __init__(self, state_dim: int, vertex_spatial_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Attention-based selection: query from state, keys from vertex embeddings
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(vertex_spatial_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        state_emb: torch.Tensor,
        vertex_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_emb: (B, state_dim)
            vertex_embs: (B, num_nodes, vertex_spatial_dim)
            mask: (B, num_nodes) bool
        Returns:
            logits: (B, num_nodes)
        """
        q = self.query(state_emb).unsqueeze(1)  # (B, 1, H)
        k = self.key(vertex_embs)  # (B, N, H)
        logits = (q * k).sum(dim=-1) / self.scale  # (B, N) -- dot product attention
        # Squeeze out the query dim if needed
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class EdgeHead(nn.Module):
    """Selects an edge for road placement."""

    def __init__(self, state_dim: int, edge_spatial_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(edge_spatial_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        state_emb: torch.Tensor,
        edge_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query(state_emb).unsqueeze(1)
        k = self.key(edge_embs)
        logits = (q * k).sum(dim=-1) / self.scale
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class HexHead(nn.Module):
    """Selects a hex tile (for robber placement)."""

    def __init__(self, state_dim: int, hex_spatial_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.query = nn.Linear(state_dim, hidden_dim)
        self.key = nn.Linear(hex_spatial_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        state_emb: torch.Tensor,
        hex_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.query(state_emb).unsqueeze(1)
        k = self.key(hex_embs)
        logits = (q * k).sum(dim=-1) / self.scale
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class ResourceHead(nn.Module):
    """Selects a resource type (for monopoly, year of plenty, trades)."""
    NUM_RESOURCES = 5  # WOOD, BRICK, SHEEP, WHEAT, ORE

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_RESOURCES),
        )

    def forward(self, state_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logits = self.net(state_emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class PlayerHead(nn.Module):
    """Selects a target player (for robber steal, trade partner)."""

    def __init__(self, input_dim: int, max_players: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_players),
        )

    def forward(self, state_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        logits = self.net(state_emb)
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class HierarchicalActionHead(nn.Module):
    """Combined hierarchical action head that maps flat catanatron action indices
    to the hierarchical decomposition and back.

    For training, we compute log_prob of a flat action by:
    1. Determine which action type it corresponds to
    2. Compute log_prob of that action type
    3. Compute log_prob of the location/parameter sub-action
    4. Sum the log_probs

    For inference, we:
    1. Sample action type
    2. Sample location/parameter conditioned on type
    3. Map back to flat action index
    """

    def __init__(
        self,
        state_dim: int,
        hex_spatial_dim: int,
        vertex_spatial_dim: int,
        edge_spatial_dim: int,
        num_tiles: int = 19,
        num_nodes: int = 54,
        num_edges: int = 72,
        max_players: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_tiles = num_tiles
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.action_type_head = ActionTypeHead(state_dim, hidden_dim)
        self.vertex_head = VertexHead(state_dim + hidden_dim, vertex_spatial_dim, hidden_dim)
        self.edge_head = EdgeHead(state_dim + hidden_dim, edge_spatial_dim, hidden_dim)
        self.hex_head = HexHead(state_dim + hidden_dim, hex_spatial_dim, hidden_dim)
        self.resource_head = ResourceHead(state_dim + hidden_dim, hidden_dim // 2)
        self.resource_head2 = ResourceHead(state_dim + hidden_dim, hidden_dim // 2)
        self.player_head = PlayerHead(state_dim + hidden_dim, max_players, hidden_dim // 2)

    def forward_flat(
        self,
        state_emb: torch.Tensor,
        hex_embs: torch.Tensor,
        vertex_embs: torch.Tensor,
        edge_embs: torch.Tensor,
        action_mask: torch.Tensor,
        action_space_size: int,
    ) -> torch.Tensor:
        """Compute logits over the full flat action space.

        This is the simplest approach: we compute all sub-head logits and
        combine them into a single flat logit vector matching catanatron's
        action space. The action_mask is applied at the end.

        For the initial version, we use a single MLP over the state embedding
        to produce flat logits, with the structured heads used for the value
        function's auxiliary losses during training.

        Args:
            state_emb: (B, state_dim)
            action_mask: (B, action_space_size) bool
            action_space_size: int
        Returns:
            logits: (B, action_space_size)
        """
        # For the flat action space, we use a direct projection
        # This is pragmatic: the hierarchical decomposition helps during
        # training via auxiliary losses, but the primary policy outputs
        # flat logits with masking for correctness.
        return None  # Use forward_policy instead

    def forward_policy(
        self,
        state_emb: torch.Tensor,
        hex_embs: torch.Tensor,
        vertex_embs: torch.Tensor,
        edge_embs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute all sub-head logits for use in policy computation.

        Returns dict of logits for each head. The PPO code handles
        mapping these to flat actions via the action mapping table.
        """
        at_logits = self.action_type_head(state_emb)

        # For conditioned heads, we concatenate state with a zero
        # autoregressive embedding (at inference, this gets the actual
        # sampled action type embedding)
        dummy_ar = torch.zeros(state_emb.shape[0], self.action_type_head.hidden_dim,
                               device=state_emb.device)
        conditioned = torch.cat([state_emb, dummy_ar], dim=-1)

        return {
            "action_type": at_logits,
            "vertex": self.vertex_head(conditioned, vertex_embs),
            "edge": self.edge_head(conditioned, edge_embs),
            "hex": self.hex_head(conditioned, hex_embs),
            "resource1": self.resource_head(conditioned),
            "resource2": self.resource_head2(conditioned),
            "player": self.player_head(conditioned),
        }
