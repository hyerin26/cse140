import typing

import pacai.agents.greedy
import pacai.core.action
import pacai.core.agent
import pacai.core.features
import pacai.core.gamestate
import pacai.search.distance
from pacai.core.board import Position
from pacai.pacman.board import MARKER_CAPSULE

import pacai.capture.gamestate

from pacai.core.agentinfo import AgentInfo

def _team_modifier_from_index(agent_index: int) -> int:
    return -1 if (agent_index % 2 == 0) else 1

def _side_of_pos(state, pos: Position) -> int:
    return -1 if (pos.col < (state.board.width / 2)) else 1

def _enemy_capsules(state, agent_index: int) -> list[Position]:
    my_side = _team_modifier_from_index(agent_index)
    capsules = []
    for p in state.board.get_marker_positions(MARKER_CAPSULE):
        if _side_of_pos(state, p) != my_side:
            capsules.append(p)
    return capsules

# Distance threshold to consider ghosts as a serious threat
GHOST_DANGER_RANGE: float = 5.0
# When ghost is extremely close (immediate escape needed)
GHOST_CRITICAL_RANGE: float = 2.0

# -------------------------
# Team factory
# -------------------------

def create_team() -> list[AgentInfo]:
    agent1_info = AgentInfo(name=f"{__name__}.OffensiveAgent")
    agent2_info = AgentInfo(name=f"{__name__}.DefensiveAgent")
    return [agent1_info, agent2_info]


# -------------------------
# Defensive Agent (Enhanced)
# -------------------------

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    Defensive agent: Guards home territory and chases invaders.
    When no invaders present, patrols near the border.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()

        # [Enhanced] Weight adjustments
        self.weights['on_home_side'] = 200.0          # Strongly prefer staying on home side
        self.weights['stopped'] = -150.0               # Don't stop (strengthened from baseline)
        self.weights['reverse'] = -5.0                 # Prevent back-and-forth (strengthened from baseline)
        self.weights['num_invaders'] = -1000.0         # Minimize number of invaders
        self.weights['distance_to_invader'] = -20.0    # [Enhanced] Chase invaders faster (-10 → -20)
        self.weights['distance_to_patrol'] = -3.0      # [New] Move to patrol position when no invaders
        self.weights['scared_ghost_penalty'] = -500.0  # [New] Penalty when scared (after being hit by capsule)

        if override_weights is None:
            override_weights = {}
        for (k, v) in override_weights.items():
            self.weights[k] = v

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)


def _extract_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        return features

    # Should stay on home side
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # Don't stop
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prevent reversing
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    # Handle invaders
    invader_positions = state.get_invader_positions(agent_index=agent.agent_index)
    features['num_invaders'] = len(invader_positions)

    if len(invader_positions) > 0:
        invader_distances = [
            agent._distances.get_distance(current_position, invader_position)
            for invader_position in invader_positions.values()
        ]
        valid_distances = [d for d in invader_distances if d is not None]
        if valid_distances:
            features['distance_to_invader'] = min(valid_distances)
    else:
        # [New] Patrol near border center when no invaders
        board_width = state.board.width
        board_height = state.board.height
        my_side = _team_modifier_from_index(agent.agent_index)

        # Set patrol target just inside border
        if my_side == -1:  # Red team (left)
            patrol_col = board_width // 2 - 2
        else:              # Blue team (right)
            patrol_col = board_width // 2 + 1

        patrol_row = board_height // 2
        patrol_pos = Position(row=patrol_row, col=patrol_col)

        d = agent._distances.get_distance(current_position, patrol_pos)
        if d is not None:
            features['distance_to_patrol'] = d

    # [New] Penalty when scared (should flee after being hit by capsule)
    if state.is_scared(agent.agent_index):
        features['scared_ghost_penalty'] = 1.0

    return features


# -------------------------
# Offensive Agent (Enhanced)
# -------------------------

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    Offensive agent: Eats opponent food while avoiding ghosts and strategically using capsules.
    Returns home safely after eating a certain amount of food.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        self._food_eaten: int = 0  # Track currently carried food

        # [Enhanced] Complete weight adjustment
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -2.0           # [Enhanced] More aggressive food seeking (-1 → -2)
        self.weights['distance_to_enemy_capsule'] = -3.0  # Slightly higher capsule priority
        self.weights['stopped'] = -200.0                  # [New] Strong penalty for stopping
        self.weights['reverse'] = -10.0                   # [New] Penalty for reversing
        self.weights['distance_to_ghost'] = 20.0          # [Enhanced] Strong ghost avoidance (was minimal)
        self.weights['ghost_critical'] = -1000.0          # [New] Immediate escape when ghost is adjacent
        self.weights['capsule_when_chased'] = -50.0       # [New] Prioritize capsule when chased
        self.weights['avoid_center'] = -5.0               # [New] Penalty for center area (opponent's common path)

        if override_weights is None:
            override_weights = {}
        for (k, v) in override_weights.items():
            self.weights[k] = v

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)


def _extract_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prevent reversing
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        return features

    # ── Ghost distance calculation (first) ─────────────────────────────
    ghost_positions = state.get_nonscared_opponent_positions(agent_index=agent.agent_index)
    ghost_distances = []
    if len(ghost_positions) > 0:
        for gp in ghost_positions.values():
            d = agent._distances.get_distance(current_position, gp)
            if d is not None:
                ghost_distances.append(d)

    min_ghost_dist = min(ghost_distances) if ghost_distances else 9999
    has_nonscared_ghosts = len(ghost_positions) > 0

    # [Enhanced] Immediate escape when ghost is very close (critical situation)
    if min_ghost_dist <= GHOST_CRITICAL_RANGE:
        features['ghost_critical'] = 1.0
    else:
        features['ghost_critical'] = 0.0

    # [Enhanced] Ghost distance reflection: stronger avoidance when closer
    if state.is_pacman(agent.agent_index) and min_ghost_dist < GHOST_DANGER_RANGE:
        # Inverse of distance: closer means higher value (avoidance)
        features['distance_to_ghost'] = 1.0 / (min_ghost_dist + 0.1)
    else:
        features['distance_to_ghost'] = 0.0

    # ── Detect being chased ───────────────────────────────────────
    being_chased = state.is_pacman(agent.agent_index) and min_ghost_dist < GHOST_DANGER_RANGE

    # ── Capsule ─────────────────────────────────────────────────
    enemy_caps = _enemy_capsules(state, agent.agent_index)
    if len(enemy_caps) > 0:
        cap_distances = [agent._distances.get_distance(current_position, cp) for cp in enemy_caps]
        cap_distances = [d for d in cap_distances if d is not None]
        if cap_distances:
            min_cap_dist = min(cap_distances)
            # [New] Prioritize capsule even more when being chased
            if being_chased:
                features['capsule_when_chased'] = min_cap_dist
            features['distance_to_enemy_capsule'] = min_cap_dist
    else:
        features['distance_to_enemy_capsule'] = 0.0

    # ── Food ─────────────────────────────────────────────────
    food_positions = state.get_food(agent_index=agent.agent_index)

    if len(food_positions) > 0:
        food_distances = [
            agent._distances.get_distance(current_position, food_position)
            for food_position in food_positions
        ]
        valid_food_dist = [d for d in food_distances if d is not None]
        if valid_food_dist:
            features['distance_to_food'] = min(valid_food_dist)
    else:
        features['distance_to_food'] = -100000
    
    # ── [New] Center avoidance ───────────────────────────────────────
    board_height = state.board.height
    center_row = board_height / 2.0
    
    # Higher value when closer to center
    # e.g., if map height is 16, center_row = 8, row=8 → 0, row=0 or 16 → 8
    distance_from_center = abs(current_position.row - center_row)
    
    # Normalize center distance (0~1 range)
    # Half of map height is maximum distance
    max_vertical_distance = board_height / 2.0
    normalized_center_distance = distance_from_center / max_vertical_distance
    
    # Penalty when close to center (1 - normalized_center_distance)
    # Center: 1.0, Top/Bottom edges: 0.0
    # Avoid center only when on opponent side and non-scared ghosts exist
    if not state.is_ghost(agent_index=agent.agent_index) and has_nonscared_ghosts:
        features['avoid_center'] = 1.0 - normalized_center_distance
    else:
        features['avoid_center'] = 0.0

    return features
