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
    # pacai capture 규칙: 짝수 인덱스 팀 vs 홀수 인덱스 팀
    return -1 if (agent_index % 2 == 0) else 1

def _side_of_pos(state, pos: Position) -> int:
    # 왼쪽 절반이면 -1, 오른쪽 절반이면 +1
    return -1 if (pos.col < (state.board.width / 2)) else 1

def _enemy_capsules(state, agent_index: int) -> list[Position]:
    my_side = _team_modifier_from_index(agent_index)
    capsules = []

    for p in state.board.get_marker_positions(MARKER_CAPSULE):
        # 내 진영 캡슐은 제외하고, 상대 진영 캡슐만!
        if _side_of_pos(state, p) != my_side:
            capsules.append(p)

    return capsules

# ghost가 너무 멀리 있으면 무시하는 거리
GHOST_IGNORE_RANGE: float = 2.5

# -------------------------
# Team factory
# -------------------------

def create_team() -> list[AgentInfo]:
    """
    Baseline과 동일: OffensiveAgent + DefensiveAgent 한 팀.
    """
    agent1_info = AgentInfo(name=f"{__name__}.OffensiveAgent")
    agent2_info = AgentInfo(name=f"{__name__}.DefensiveAgent")
    return [agent1_info, agent2_info]


# -------------------------
# Defensive Agent (Baseline)
# -------------------------

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes defending its own territory.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_baseline_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()

        # Set base weights (baseline).
        self.weights['on_home_side'] = 80.0
        self.weights['stopped'] = -100.0
        self.weights['reverse'] = -2.0
        self.weights['num_invaders'] = -1000.0
        self.weights['distance_to_invader'] = -10.0

        if override_weights is None:
            override_weights = {}
        for (k, v) in override_weights.items():
            self.weights[k] = v

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)


def _extract_baseline_defensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(DefensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        # We are dead and waiting to respawn.
        return features

    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # Remember that the state we get is already a successor, so we have to look two actions back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    # We don't like any invaders on our side.
    invader_positions = state.get_invader_positions(agent_index=agent.agent_index)
    features['num_invaders'] = len(invader_positions)

    # Hunt down the closest invader!
    if len(invader_positions) > 0:
        invader_distances = [
            agent._distances.get_distance(current_position, invader_position)
            for invader_position in invader_positions.values()
        ]
        features['distance_to_invader'] = min(d for d in invader_distances if d is not None)

    return features


# -------------------------
# Offensive Agent (Baseline)
# -------------------------

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    A capture agent that prioritizes getting food while avoiding ghosts.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_baseline_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()

        # Set base weights
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -1.0
        self.weights['distance_to_enemy_capsule'] = -2.0
        self.weights['distance_to_ghost_squared'] = 0.001

        # Baseline에서는 weight를 따로 안 두지만,
        # feature로는 distance_to_ghost / distance_to_ghost_squared를 만들어둡니다.
        # 필요하면 override_weights로 가중치를 추가해 튜닝할 수 있어요.

        if override_weights is None:
            override_weights = {}
        for (k, v) in override_weights.items():
            self.weights[k] = v

    def game_start(self, initial_state: pacai.core.gamestate.GameState) -> None:
        self._distances.compute(initial_state.board)


def _extract_baseline_offensive_features(
        state: pacai.core.gamestate.GameState,
        action: pacai.core.action.Action,
        agent: pacai.core.agent.Agent | None = None,
        **kwargs: typing.Any) -> pacai.core.features.FeatureDict:
    agent = typing.cast(OffensiveAgent, agent)
    state = typing.cast(pacai.capture.gamestate.GameState, state)

    features: pacai.core.features.FeatureDict = pacai.core.features.FeatureDict()
    features['score'] = state.get_normalized_score(agent.agent_index)

    # Note the side of the board we are on.
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # Prefer moving over stopping.
    features['stopped'] = int(action == pacai.core.action.STOP)

    # Prefer not turning around.
    # Remember that the state we get is already a successor, so we have to look two actions back.
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        # We are dead and waiting to respawn.
        return features

    enemy_caps = _enemy_capsules(state, agent.agent_index)

    if len(enemy_caps) > 0:
        cap_distances = [agent._distances.get_distance(current_position, cp) for cp in enemy_caps]
        cap_distances = [d for d in cap_distances if d is not None]
        if len(cap_distances) > 0:
            features['distance_to_enemy_capsule'] = min(cap_distances)
    else:
        features['distance_to_enemy_capsule'] = 0.0

    # Closest food.
    food_positions = state.get_food(agent_index=agent.agent_index)
    if len(food_positions) > 0:
        food_distances = [
            agent._distances.get_distance(current_position, food_position)
            for food_position in food_positions
        ]
        features['distance_to_food'] = min(d for d in food_distances if d is not None)
    else:
        # There is no food left, give a large score.
        features['distance_to_food'] = -100000

    # Closest non-scared opponent ghost.
    # [추가] 내가 적진에 있을 때(=Pac-Man)만 고스트 위험을 반영.
    if state.is_pacman(agent.agent_index):
        ghost_positions = state.get_nonscared_opponent_positions(agent_index=agent.agent_index)
        if len(ghost_positions) > 0:
            ghost_distances = [
                agent._distances.get_distance(current_position, ghost_position)
                for ghost_position in ghost_positions.values()
            ]
            dghost = min(d for d in ghost_distances if d is not None)

            # 베이스라인 방식: 너무 멀면 무시
            if dghost > GHOST_IGNORE_RANGE:
                dghost = 1000

            features['distance_to_ghost'] = float(dghost)
            features['distance_to_ghost_squared'] = float(dghost * dghost)
        else:
            features['distance_to_ghost'] = 0.0
            features['distance_to_ghost_squared'] = 0.0
    else:
        # 우리 진영(고스트)일 땐 상대 고스트 회피 필요 없음
        features['distance_to_ghost'] = 0.0
        features['distance_to_ghost_squared'] = 0.0

    return features
