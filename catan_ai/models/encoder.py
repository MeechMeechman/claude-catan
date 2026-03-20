"""
Cross-dimensional neural network encoder for Catan board state.

Based on Gendre & Kaneko (2020) "Deep Reinforcement Learning with
Cross-Dimensional Neural Networks." Uses inflation/deflation operations
to connect hex, vertex, and edge feature spaces through residual layers.

The three board element types (hexes, vertices, edges) have different
feature semantics and neighborhood structures. Standard CNNs/MLPs
treat all dimensions identically and lose this structure. The Xdim
architecture explicitly separates and connects them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InflateHexToVertex(nn.Module):
    """Inflate hex features to vertex space by gathering from adjacent hexes.

    Each vertex is adjacent to up to 3 hexes. We gather those hex features,
    concatenate, and project to vertex feature dimension.
    """

    def __init__(self, hex_dim: int, vertex_dim: int, hex_to_vertex_adj: np.ndarray):
        super().__init__()
        # hex_to_vertex_adj: (num_tiles, 6) -> for each tile, 6 vertex indices
        # We need the inverse: for each vertex, which hexes are adjacent
        # vertex_to_hex_adj is passed separately
        self.project = nn.Linear(hex_dim * 3, vertex_dim)

    def forward(self, hex_feats: torch.Tensor, vertex_to_hex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hex_feats: (batch, num_tiles, hex_dim)
            vertex_to_hex: (num_nodes, 3) int64, -1 padded
        Returns:
            (batch, num_nodes, vertex_dim)
        """
        B, T, D = hex_feats.shape
        N = vertex_to_hex.shape[0]

        mask = (vertex_to_hex >= 0).float()  # (N, 3)
        safe_idx = vertex_to_hex.clamp(min=0)  # (N, 3)

        # Gather: index into hex_feats along the tile dimension
        # safe_idx: (N, 3) -> expand to (B, N, 3, D) for gather on dim=1
        idx = safe_idx.unsqueeze(0).unsqueeze(-1).expand(B, N, 3, D)  # (B, N, 3, D)
        src = hex_feats.unsqueeze(2).expand(B, T, 3, D)  # need (B, T_or_more, 3, D) -- wrong approach

        # Simpler: flatten the gather. hex_feats is (B, T, D).
        # For each vertex's 3 adjacent hexes, pick from hex_feats.
        # Use index_select-style: hex_feats[:, safe_idx, :] gives (B, N, 3, D)
        gathered = hex_feats[:, safe_idx.view(-1), :].view(B, N, 3, D)
        gathered = gathered * mask.unsqueeze(0).unsqueeze(-1)  # zero padding

        gathered = gathered.reshape(B, N, 3 * D)
        return self.project(gathered)


class DeflateVertexToHex(nn.Module):
    """Deflate vertex features back to hex space by aggregating from adjacent vertices.

    Each hex has exactly 6 adjacent vertices. We gather, mean-pool, and project.
    """

    def __init__(self, vertex_dim: int, hex_dim: int):
        super().__init__()
        self.project = nn.Linear(vertex_dim, hex_dim)

    def forward(self, vertex_feats: torch.Tensor, hex_to_vertex: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertex_feats: (batch, num_nodes, vertex_dim)
            hex_to_vertex: (num_tiles, 6) int64, -1 padded
        Returns:
            (batch, num_tiles, hex_dim)
        """
        B, N, D = vertex_feats.shape
        T = hex_to_vertex.shape[0]

        mask = (hex_to_vertex >= 0).float()  # (T, 6)
        safe_idx = hex_to_vertex.clamp(min=0)  # (T, 6)

        # vertex_feats[:, safe_idx.view(-1), :] -> (B, T*6, D) -> (B, T, 6, D)
        gathered = vertex_feats[:, safe_idx.view(-1), :].view(B, T, 6, D)
        gathered = gathered * mask.unsqueeze(0).unsqueeze(-1)
        count = mask.unsqueeze(0).unsqueeze(-1).sum(dim=2).clamp(min=1)
        pooled = gathered.sum(dim=2) / count
        return self.project(pooled)


class VertexMessagePass(nn.Module):
    """Message passing between adjacent vertices (along edges)."""

    def __init__(self, dim: int):
        super().__init__()
        self.message_fn = nn.Linear(dim, dim)
        self.update_fn = nn.Linear(2 * dim, dim)

    def forward(self, vertex_feats: torch.Tensor, vertex_adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertex_feats: (batch, num_nodes, dim)
            vertex_adj: (num_nodes, 3) int64, -1 padded
        Returns:
            (batch, num_nodes, dim)
        """
        B, N, D = vertex_feats.shape

        mask = (vertex_adj >= 0).float()  # (N, 3)
        safe_idx = vertex_adj.clamp(min=0)  # (N, 3)

        # vertex_feats[:, safe_idx.view(-1), :] -> (B, N*3, D) -> (B, N, 3, D)
        neighbors = vertex_feats[:, safe_idx.view(-1), :].view(B, N, 3, D)
        neighbors = neighbors * mask.unsqueeze(0).unsqueeze(-1)
        messages = self.message_fn(neighbors)
        count = mask.unsqueeze(0).unsqueeze(-1).sum(dim=2).clamp(min=1)
        agg = messages.sum(dim=2) / count

        combined = torch.cat([vertex_feats, agg], dim=-1)
        return self.update_fn(combined)


class XdimResBlock(nn.Module):
    """One cross-dimensional residual block.

    Flow: hex -> inflate -> vertex message pass -> deflate -> hex (residual)
    Also updates vertex features via message passing with residual.
    """

    def __init__(self, hex_dim: int, vertex_dim: int):
        super().__init__()
        self.inflate = InflateHexToVertex(hex_dim, vertex_dim, None)
        self.vertex_mp = VertexMessagePass(vertex_dim)
        self.deflate = DeflateVertexToHex(vertex_dim, hex_dim)

        self.hex_norm = nn.LayerNorm(hex_dim)
        self.vertex_norm = nn.LayerNorm(vertex_dim)

        self.hex_ff = nn.Sequential(
            nn.Linear(hex_dim, hex_dim * 2),
            nn.GELU(),
            nn.Linear(hex_dim * 2, hex_dim),
        )
        self.vertex_ff = nn.Sequential(
            nn.Linear(vertex_dim, vertex_dim * 2),
            nn.GELU(),
            nn.Linear(vertex_dim * 2, vertex_dim),
        )

    def forward(self, hex_feats, vertex_feats, vertex_to_hex, hex_to_vertex, vertex_adj):
        # Inflate hex info to vertices
        inflated = self.inflate(hex_feats, vertex_to_hex)
        vertex_feats = vertex_feats + inflated

        # Vertex message passing
        mp_out = self.vertex_mp(vertex_feats, vertex_adj)
        vertex_feats = self.vertex_norm(vertex_feats + mp_out)
        vertex_feats = vertex_feats + self.vertex_ff(vertex_feats)

        # Deflate vertex info back to hexes
        deflated = self.deflate(vertex_feats, hex_to_vertex)
        hex_feats = self.hex_norm(hex_feats + deflated)
        hex_feats = hex_feats + self.hex_ff(hex_feats)

        return hex_feats, vertex_feats


class CrossDimEncoder(nn.Module):
    """Full cross-dimensional encoder for Catan board state.

    Takes structured observations and produces a fixed-size embedding
    combining board topology with player state information.

    Architecture:
        1. Project raw features to hidden dims
        2. N cross-dimensional residual blocks
        3. Pool hex and vertex features
        4. Concatenate with edge features and player features
        5. Final MLP to produce state embedding
    """

    def __init__(
        self,
        hex_input_dim: int,
        vertex_input_dim: int,
        edge_input_dim: int,
        player_input_dim: int,
        hidden_dim: int = 128,
        num_xdim_layers: int = 4,
        output_dim: int = 256,
    ):
        super().__init__()

        self.hex_hidden = hidden_dim
        self.vertex_hidden = hidden_dim
        self.output_dim = output_dim

        # Input projections
        self.hex_proj = nn.Linear(hex_input_dim, hidden_dim)
        self.vertex_proj = nn.Linear(vertex_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim // 2)

        # Cross-dimensional blocks
        self.xdim_blocks = nn.ModuleList([
            XdimResBlock(hidden_dim, hidden_dim)
            for _ in range(num_xdim_layers)
        ])

        # Edge processing (simple MLP since edges connect vertices)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
        )

        # Player feature processing
        self.player_mlp = nn.Sequential(
            nn.Linear(player_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Final combination
        # hex_pool + vertex_pool + edge_pool + player = 4 * hidden_dim
        pool_dim = hidden_dim + hidden_dim + hidden_dim // 2 + hidden_dim
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        hex_features: torch.Tensor,
        vertex_features: torch.Tensor,
        edge_features: torch.Tensor,
        player_features: torch.Tensor,
        vertex_to_hex: torch.Tensor,
        hex_to_vertex: torch.Tensor,
        vertex_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hex_features: (B, num_tiles, hex_input_dim)
            vertex_features: (B, num_nodes, vertex_input_dim)
            edge_features: (B, num_edges, edge_input_dim)
            player_features: (B, player_input_dim)
            vertex_to_hex: (num_nodes, 3) int64 (static)
            hex_to_vertex: (num_tiles, 6) int64 (static)
            vertex_adj: (num_nodes, 3) int64 (static)
        Returns:
            state_embedding: (B, output_dim)
        """
        # Project inputs
        h = self.hex_proj(hex_features)
        v = self.vertex_proj(vertex_features)
        e = self.edge_proj(edge_features)

        # Cross-dimensional message passing
        for block in self.xdim_blocks:
            h, v = block(h, v, vertex_to_hex, hex_to_vertex, vertex_adj)

        # Edge processing
        e = self.edge_mlp(e)

        # Player feature processing
        p = self.player_mlp(player_features)

        # Pool spatial features
        h_pool = h.mean(dim=1)  # (B, hidden)
        v_pool = v.mean(dim=1)  # (B, hidden)
        e_pool = e.mean(dim=1)  # (B, hidden//2)

        # Combine everything
        combined = torch.cat([h_pool, v_pool, e_pool, p], dim=-1)
        return self.output_mlp(combined)

    def forward_with_spatial(
        self,
        hex_features, vertex_features, edge_features, player_features,
        vertex_to_hex, hex_to_vertex, vertex_adj,
    ):
        """Like forward() but also returns per-vertex and per-hex embeddings
        for the action heads that need spatial selections."""
        h = self.hex_proj(hex_features)
        v = self.vertex_proj(vertex_features)
        e = self.edge_proj(edge_features)

        for block in self.xdim_blocks:
            h, v = block(h, v, vertex_to_hex, hex_to_vertex, vertex_adj)

        e = self.edge_mlp(e)
        p = self.player_mlp(player_features)

        h_pool = h.mean(dim=1)
        v_pool = v.mean(dim=1)
        e_pool = e.mean(dim=1)

        combined = torch.cat([h_pool, v_pool, e_pool, p], dim=-1)
        state_emb = self.output_mlp(combined)

        return state_emb, h, v, e
