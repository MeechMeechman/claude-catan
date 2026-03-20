"""
Structured Catan environment wrapper around Catanatron.

Extracts separate hex/vertex/edge/player feature tensors for the
cross-dimensional encoder, while using Catanatron's game engine underneath.
"""

import sys
import os
import random
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Add catanatron to path
_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "catanatron_repo", "catanatron")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from catanatron.game import Game, TURNS_LIMIT
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.models.map import (
    NUM_NODES,
    NUM_EDGES,
    NUM_TILES,
    build_map,
    number_probability,
    CatanMap,
)
from catanatron.models.enums import (
    RESOURCES,
    DEVELOPMENT_CARDS,
    SETTLEMENT,
    CITY,
    ROAD,
    ActionType,
    VICTORY_POINT,
)
from catanatron.models.board import get_edges, STATIC_GRAPH
from catanatron.state_functions import (
    get_player_buildings,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.models.decks import freqdeck_count
from catanatron.gym.envs.action_space import (
    get_action_array,
    to_action_space,
    from_action_space,
)
from catanatron.features import iter_players

# Constants for the BASE map
HEX_FEAT_DIM = 8  # resource(6) + proba(1) + robber(1)
VERTEX_FEAT_DIM_PER_PLAYER = 3  # settlement(1) + city(1) + harbor_mask(1)
EDGE_FEAT_DIM_PER_PLAYER = 1  # road(1)
HARBOR_TYPES = 6  # 5 resources + 3:1


def _build_edge_index(catan_map: CatanMap) -> np.ndarray:
    """Build sorted edge list from the map for consistent indexing."""
    edges = sorted(get_edges(catan_map.land_nodes))
    return edges


def _build_adjacency(catan_map: CatanMap):
    """Build hex-vertex, hex-edge, vertex-edge adjacency structures."""
    hex_to_vertices = {}  # tile_id -> list of node_ids
    vertex_to_hexes = {}  # node_id -> list of tile_ids

    for tile_id, tile in catan_map.tiles_by_id.items():
        nodes = list(tile.nodes.values())
        hex_to_vertices[tile_id] = nodes
        for n in nodes:
            vertex_to_hexes.setdefault(n, []).append(tile_id)

    return hex_to_vertices, vertex_to_hexes


class CatanTrainEnv(gym.Env):
    """Training environment that outputs structured observations for the
    cross-dimensional network encoder.

    Observations are a dict:
        hex_features:    (NUM_TILES, hex_feat_dim)
        vertex_features: (NUM_NODES, vertex_feat_dim)
        edge_features:   (NUM_EDGES, edge_feat_dim)
        player_features: (player_feat_dim,)
        action_mask:     (action_space_size,) bool

    Adjacency matrices (static, stored once):
        hex_to_vertex:   (NUM_TILES, 6) int indices
        vertex_to_hex:   (NUM_NODES, 3) int indices (-1 for padding)
        vertex_adj:      (NUM_NODES, 3) int indices (-1 for padding)
    """

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        config = config or {}

        self.num_players = config.get("num_players", 2)
        self.map_type = config.get("map_type", "BASE")
        self.vps_to_win = config.get("vps_to_win", 10)
        self.reward_type = config.get("reward_type", "sparse")  # sparse | shaped

        # Enemy player: support class or string name
        enemy_spec = config.get("enemy_class", RandomPlayer)
        if isinstance(enemy_spec, str):
            enemy_spec = self._resolve_enemy_class(enemy_spec)
        self.enemy_player_class = enemy_spec

        # Potential-based reward shaping weights
        self.shaping_weights = config.get("shaping_weights", {
            "vp": 0.3,
            "production": 0.1,
            "road_length": 0.05,
        })

        # Build color assignments
        all_colors = [Color.BLUE, Color.RED, Color.WHITE, Color.ORANGE]
        self.p0_color = Color.BLUE
        enemy_colors = all_colors[1:self.num_players]
        self.enemies = [self.enemy_player_class(c) for c in enemy_colors]
        self.p0 = Player(self.p0_color)
        self.players = [self.p0] + self.enemies
        self.player_colors = tuple(p.color for p in self.players)

        # Action space
        self.action_array = get_action_array(self.player_colors, self.map_type)
        self.action_space_size = len(self.action_array)
        self.action_space = spaces.Discrete(self.action_space_size)

        # Build static adjacency from a reference map
        ref_map = build_map(self.map_type)
        self._edge_list = _build_edge_index(ref_map)
        self._edge_to_idx = {tuple(sorted(e)): i for i, e in enumerate(self._edge_list)}
        self._hex_to_verts, self._vert_to_hexes = _build_adjacency(ref_map)
        self.num_tiles = len(ref_map.tiles_by_id)
        self.num_nodes = len(ref_map.land_nodes)
        self.num_edges = len(self._edge_list)

        # Build adjacency tensors (static)
        self.hex_to_vertex_adj = np.full((self.num_tiles, 6), -1, dtype=np.int32)
        for tid, nodes in self._hex_to_verts.items():
            for j, n in enumerate(nodes):
                self.hex_to_vertex_adj[tid, j] = n

        self.vertex_to_hex_adj = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        for nid, tids in self._vert_to_hexes.items():
            if nid < self.num_nodes:
                for j, tid in enumerate(tids[:3]):
                    self.vertex_to_hex_adj[nid, j] = tid

        # Vertex-vertex adjacency from STATIC_GRAPH (only land nodes)
        land_node_set = ref_map.land_nodes
        self.vertex_adj = np.full((self.num_nodes, 3), -1, dtype=np.int32)
        for n in range(self.num_nodes):
            if n in STATIC_GRAPH:
                # Filter to only land node neighbors
                neighbors = sorted(nb for nb in STATIC_GRAPH.neighbors(n) if nb in land_node_set)
                for j, nb in enumerate(neighbors[:3]):
                    self.vertex_adj[n, j] = nb

        # Feature dimensions
        self.hex_feat_dim = HEX_FEAT_DIM
        self.vertex_feat_dim = 2 * self.num_players + HARBOR_TYPES
        self.edge_feat_dim = self.num_players
        self.player_feat_dim = self._calc_player_feat_dim()

        # Observation space (as dict of boxes)
        self.observation_space = spaces.Dict({
            "hex_features": spaces.Box(-1, 10, (self.num_tiles, self.hex_feat_dim), dtype=np.float32),
            "vertex_features": spaces.Box(-1, 10, (self.num_nodes, self.vertex_feat_dim), dtype=np.float32),
            "edge_features": spaces.Box(-1, 10, (self.num_edges, self.edge_feat_dim), dtype=np.float32),
            "player_features": spaces.Box(-1, 100, (self.player_feat_dim,), dtype=np.float32),
            "action_mask": spaces.MultiBinary(self.action_space_size),
        })

        self.game: Optional[Game] = None
        self._prev_potential = 0.0

    def _calc_player_feat_dim(self) -> int:
        # Per current player: 5 resources + 5 dev cards + 3 pieces left +
        #   1 army + 1 road_flag + 1 army_flag + 1 longest_road_len + 1 vps +
        #   1 has_rolled + 1 has_played_dev = 20
        # Per opponent: 1 resource_count + 1 dev_count + 3 pieces +
        #   1 army + 1 road_flag + 1 army_flag + 1 longest_road + 1 vps +
        #   4 dev_played = 14
        # Global: 5 bank resources + 1 dev pile + 2 phase flags = 8
        return 20 + 14 * (self.num_players - 1) + 8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        catan_map = build_map(self.map_type)
        for p in self.players:
            p.reset_state()
        self.game = Game(
            players=self.players,
            seed=seed,
            catan_map=catan_map,
            vps_to_win=self.vps_to_win,
        )
        self._advance_until_p0()
        self._prev_potential = self._compute_potential()

        obs = self._get_obs()
        info = {"valid_actions": self._get_valid_action_indices()}
        return obs, info

    def step(self, action: int):
        catan_action = from_action_space(
            action, self.p0_color, self.player_colors, self.map_type
        )

        # Check validity
        if catan_action not in self.game.playable_actions:
            obs = self._get_obs()
            return obs, -1.0, False, False, {"valid_actions": self._get_valid_action_indices()}

        self.game.execute(catan_action)
        self._advance_until_p0()

        obs = self._get_obs()
        info = {"valid_actions": self._get_valid_action_indices()}

        winning_color = self.game.winning_color()
        terminated = winning_color is not None
        truncated = self.game.state.num_turns >= TURNS_LIMIT

        # Compute reward
        if terminated:
            reward = 1.0 if winning_color == self.p0_color else -1.0
        elif truncated:
            reward = 0.0
        else:
            reward = 0.0

        # Add potential-based shaping if configured
        if self.reward_type == "shaped" and not terminated:
            new_potential = self._compute_potential()
            shaping = 0.999 * new_potential - self._prev_potential
            reward += shaping
            self._prev_potential = new_potential

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        valid = set(self._get_valid_action_indices())
        return np.array([i in valid for i in range(self.action_space_size)], dtype=bool)

    # ---- Observation extraction ----

    def _get_obs(self) -> Dict[str, np.ndarray]:
        game = self.game
        state = game.state
        board = state.board
        catan_map = board.map

        # 1) Hex features: resource one-hot(6) + proba(1) + robber(1)
        hex_feats = np.zeros((self.num_tiles, self.hex_feat_dim), dtype=np.float32)
        resource_idx = {"WOOD": 0, "BRICK": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
        for tid, tile in catan_map.tiles_by_id.items():
            if tile.resource is not None:
                hex_feats[tid, resource_idx[tile.resource]] = 1.0
                hex_feats[tid, 6] = number_probability(tile.number)
            else:
                hex_feats[tid, 5] = 1.0  # desert
            if catan_map.tiles[board.robber_coordinate] == tile:
                hex_feats[tid, 7] = 1.0

        # 2) Vertex features: per-player settlement(1) + city(1) + harbor(6)
        vertex_feats = np.zeros((self.num_nodes, self.vertex_feat_dim), dtype=np.float32)
        for pi, (i, color) in enumerate(iter_players(state.colors, self.p0_color)):
            for nid in get_player_buildings(state, color, SETTLEMENT):
                if nid < self.num_nodes:
                    vertex_feats[nid, 2 * pi] = 1.0
            for nid in get_player_buildings(state, color, CITY):
                if nid < self.num_nodes:
                    vertex_feats[nid, 2 * pi + 1] = 1.0

        # Harbor features (shared, not player-specific)
        harbor_offset = 2 * self.num_players
        for resource, node_ids in catan_map.port_nodes.items():
            if resource is None:
                idx = 5  # 3:1 port
            else:
                idx = resource_idx.get(resource, 5)
            for nid in node_ids:
                if nid < self.num_nodes:
                    vertex_feats[nid, harbor_offset + idx] = 1.0

        # 3) Edge features: per-player road ownership
        edge_feats = np.zeros((self.num_edges, self.edge_feat_dim), dtype=np.float32)
        for pi, (i, color) in enumerate(iter_players(state.colors, self.p0_color)):
            for edge in get_player_buildings(state, color, ROAD):
                edge_key = tuple(sorted(edge))
                eidx = self._edge_to_idx.get(edge_key)
                if eidx is not None:
                    edge_feats[eidx, pi] = 1.0

        # 4) Player features (flat vector)
        player_feats = self._extract_player_features()

        # 5) Action mask
        mask = self.action_masks()

        return {
            "hex_features": hex_feats,
            "vertex_features": vertex_feats,
            "edge_features": edge_feats,
            "player_features": player_feats,
            "action_mask": mask,
        }

    def _extract_player_features(self) -> np.ndarray:
        state = self.game.state
        feats = []

        for pi, (i, color) in enumerate(iter_players(state.colors, self.p0_color)):
            key = player_key(state, color)
            ps = state.player_state

            if pi == 0:  # current player - full info
                for r in RESOURCES:
                    feats.append(float(ps[key + f"_{r}_IN_HAND"]))
                for d in DEVELOPMENT_CARDS:
                    feats.append(float(ps[key + f"_{d}_IN_HAND"]))
                feats.append(float(ps[key + "_SETTLEMENTS_AVAILABLE"]))
                feats.append(float(ps[key + "_CITIES_AVAILABLE"]))
                feats.append(float(ps[key + "_ROADS_AVAILABLE"]))
                feats.append(float(ps[key + "_LONGEST_ROAD_LENGTH"]))
                feats.append(float(ps[key + "_ACTUAL_VICTORY_POINTS"]))
                feats.append(float(ps[key + "_HAS_ARMY"]))
                feats.append(float(ps[key + "_HAS_ROAD"]))
                feats.append(float(ps[key + "_HAS_ROLLED"]))
                feats.append(float(ps[key + "_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"]))
                # Army size from played knights
                feats.append(float(ps.get(key + "_PLAYED_KNIGHT", 0)))
            else:  # opponent - hidden info
                feats.append(float(player_num_resource_cards(state, color)))
                feats.append(float(player_num_dev_cards(state, color)))
                feats.append(float(ps[key + "_SETTLEMENTS_AVAILABLE"]))
                feats.append(float(ps[key + "_CITIES_AVAILABLE"]))
                feats.append(float(ps[key + "_ROADS_AVAILABLE"]))
                feats.append(float(ps[key + "_LONGEST_ROAD_LENGTH"]))
                feats.append(float(ps[key + "_VICTORY_POINTS"]))
                feats.append(float(ps[key + "_HAS_ARMY"]))
                feats.append(float(ps[key + "_HAS_ROAD"]))
                # Dev cards played
                for d in DEVELOPMENT_CARDS:
                    if d == VICTORY_POINT:
                        continue
                    feats.append(float(ps.get(key + f"_PLAYED_{d}", 0)))
                feats.append(float(ps.get(key + "_PLAYED_KNIGHT", 0)))

        # Global features
        for r in RESOURCES:
            feats.append(float(freqdeck_count(state.resource_freqdeck, r)))
        feats.append(float(len(state.development_listdeck)))
        possibilities = set(a.action_type for a in self.game.playable_actions)
        feats.append(float(ActionType.MOVE_ROBBER in possibilities))
        feats.append(float(ActionType.DISCARD in possibilities))

        return np.array(feats, dtype=np.float32)

    # ---- Potential-based reward shaping ----

    def _compute_potential(self) -> float:
        if self.game is None:
            return 0.0
        state = self.game.state
        key = player_key(state, self.p0_color)
        ps = state.player_state

        vp = float(ps[key + "_ACTUAL_VICTORY_POINTS"])

        # Production rate: sum of probabilities for all tiles adjacent to settlements/cities
        production = 0.0
        catan_map = state.board.map
        for nid in get_player_buildings(state, self.p0_color, SETTLEMENT):
            for tile in catan_map.adjacent_tiles.get(nid, []):
                if tile.resource is not None:
                    production += number_probability(tile.number)
        for nid in get_player_buildings(state, self.p0_color, CITY):
            for tile in catan_map.adjacent_tiles.get(nid, []):
                if tile.resource is not None:
                    production += 2 * number_probability(tile.number)

        road_len = float(ps[key + "_LONGEST_ROAD_LENGTH"])

        w = self.shaping_weights
        potential = w["vp"] * vp + w["production"] * production + w["road_length"] * road_len

        # Normalize to [0, 1] range approximately
        # Max VP ~10, max production ~3.0, max road ~15
        potential = potential / (w["vp"] * 10 + w["production"] * 3.0 + w["road_length"] * 15)
        return potential

    # ---- Helpers ----

    def _get_valid_action_indices(self) -> List[int]:
        return sorted(
            to_action_space(a, self.player_colors, self.map_type)
            for a in self.game.playable_actions
        )

    def _advance_until_p0(self):
        while (
            self.game.winning_color() is None
            and self.game.state.num_turns < TURNS_LIMIT
            and self.game.state.current_color() != self.p0_color
        ):
            self.game.play_tick()

    def set_enemies(self, enemies: List[Player]):
        """Replace enemy players (for self-play with trained agents)."""
        self.enemies = enemies
        self.players = [self.p0] + self.enemies
        self.player_colors = tuple(p.color for p in self.players)

    @staticmethod
    def _resolve_enemy_class(name: str):
        """Resolve enemy class from string name (for multiprocessing pickling)."""
        import sys
        _REPO = os.path.join(os.path.dirname(__file__), "..", "catanatron_repo", "catanatron")
        if _REPO not in sys.path:
            sys.path.insert(0, _REPO)

        if name == "value_fn" or name == "value":
            from catanatron.players.value import ValueFunctionPlayer
            return ValueFunctionPlayer
        elif name == "alphabeta":
            from catanatron.players.minimax import AlphaBetaPlayer
            return AlphaBetaPlayer
        elif name == "random":
            return RandomPlayer
        else:
            raise ValueError(f"Unknown enemy type: {name}")
