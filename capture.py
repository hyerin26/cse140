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

# 유령을 본격적으로 위험으로 판단하는 거리 임계값
GHOST_DANGER_RANGE: float = 5.0
# 유령이 매우 가까울 때 (즉시 도망)
GHOST_CRITICAL_RANGE: float = 2.0

# -------------------------
# Team factory
# -------------------------

def create_team() -> list[AgentInfo]:
    agent1_info = AgentInfo(name=f"{__name__}.OffensiveAgent")
    agent2_info = AgentInfo(name=f"{__name__}.DefensiveAgent")
    return [agent1_info, agent2_info]


# -------------------------
# Defensive Agent (개선)
# -------------------------

class DefensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    수비 에이전트: 내 진영을 지키며 침입자를 쫓는다.
    침입자가 없을 때는 경계선 근처를 순찰한다.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_defensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()

        # [개선] 가중치 재조정
        self.weights['on_home_side'] = 200.0          # 내 진영에 머무르기 강력히 선호
        self.weights['stopped'] = -150.0               # 멈추지 않기 (기존보다 강화)
        self.weights['reverse'] = -5.0                 # 왔다갔다 방지 (기존보다 강화)
        self.weights['num_invaders'] = -1000.0         # 침입자 수 최소화
        self.weights['distance_to_invader'] = -20.0    # [개선] 침입자 더 빠르게 추격 (-10 → -20)
        self.weights['distance_to_patrol'] = -3.0      # [신규] 침입자 없을 때 순찰 위치로 이동
        self.weights['scared_ghost_penalty'] = -500.0  # [신규] 내가 scared 상태일 때 패널티

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

    # 내 진영에 있어야 함
    features['on_home_side'] = int(state.is_ghost(agent_index=agent.agent_index))

    # 멈추지 않기
    features['stopped'] = int(action == pacai.core.action.STOP)

    # 왔다갔다 방지
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    # 침입자 처리
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
        # [신규] 침입자가 없으면 경계선 중앙 근처를 순찰
        board_width = state.board.width
        board_height = state.board.height
        my_side = _team_modifier_from_index(agent.agent_index)

        # 내 진영 경계선 바로 안쪽 열을 순찰 목표로 설정
        if my_side == -1:  # 레드팀 (왼쪽)
            patrol_col = board_width // 2 - 2
        else:              # 블루팀 (오른쪽)
            patrol_col = board_width // 2 + 1

        patrol_row = board_height // 2
        patrol_pos = Position(row=patrol_row, col=patrol_col)

        d = agent._distances.get_distance(current_position, patrol_pos)
        if d is not None:
            features['distance_to_patrol'] = d

    # [신규] 내가 scared 상태면 패널티 (캡슐 먹혀서 scared 되면 도망가야 함)
    if state.is_scared(agent.agent_index):
        features['scared_ghost_penalty'] = 1.0

    return features


# -------------------------
# Offensive Agent (개선)
# -------------------------

class OffensiveAgent(pacai.agents.greedy.GreedyFeatureAgent):
    """
    공격 에이전트: 상대 음식을 먹되, 유령을 피하고 캡슐을 전략적으로 활용한다.
    음식을 일정량 먹으면 안전하게 집으로 복귀한다.
    """

    def __init__(self,
            override_weights: dict[str, float] | None = None,
            **kwargs: typing.Any) -> None:
        kwargs['feature_extractor_func'] = _extract_offensive_features
        super().__init__(**kwargs)

        self._distances: pacai.search.distance.DistancePreComputer = pacai.search.distance.DistancePreComputer()
        self._food_eaten: int = 0  # 현재 들고 있는 음식 수 추적

        # [개선] 가중치 전면 재조정
        self.weights['score'] = 100.0
        self.weights['distance_to_food'] = -2.0           # [개선] 음식 더 적극적으로 추구 (-1 → -2)
        self.weights['distance_to_enemy_capsule'] = -3.0  # 캡슐 우선순위 약간 상향
        self.weights['stopped'] = -200.0                  # [신규] 멈추면 강력한 패널티
        self.weights['reverse'] = -10.0                   # [신규] 왔다갔다 패널티
        self.weights['distance_to_ghost'] = 20.0          # [개선] 유령 회피 강력히 (기존 거의 없었음)
        self.weights['ghost_critical'] = -1000.0          # [신규] 유령 바로 옆이면 즉시 도망
        self.weights['capsule_when_chased'] = -50.0       # [신규] 쫓길 때 캡슐 우선

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

    # 왔다갔다 방지
    agent_actions = state.get_agent_actions(agent.agent_index)
    if len(agent_actions) > 1:
        features['reverse'] = int(action == state.get_reverse_action(agent_actions[-2]))

    current_position = state.get_agent_position(agent.agent_index)
    if current_position is None:
        return features

    # ── 유령 거리 계산 (가장 먼저) ─────────────────────────────
    ghost_positions = state.get_nonscared_opponent_positions(agent_index=agent.agent_index)
    ghost_distances = []
    if len(ghost_positions) > 0:
        for gp in ghost_positions.values():
            d = agent._distances.get_distance(current_position, gp)
            if d is not None:
                ghost_distances.append(d)

    min_ghost_dist = min(ghost_distances) if ghost_distances else 9999

    # [개선] 유령이 매우 가까우면 즉시 도망 (critical 상황)
    if min_ghost_dist <= GHOST_CRITICAL_RANGE:
        features['ghost_critical'] = 1.0
    else:
        features['ghost_critical'] = 0.0

    # [개선] 유령 거리 반영: 가까울수록 강하게 회피
    if state.is_pacman(agent.agent_index) and min_ghost_dist < GHOST_DANGER_RANGE:
        # 거리의 역수로 표현: 가까울수록 큰 값 (회피)
        features['distance_to_ghost'] = 1.0 / (min_ghost_dist + 0.1)
    else:
        features['distance_to_ghost'] = 0.0

    # ── 쫓기는 상황 감지 ───────────────────────────────────────
    being_chased = state.is_pacman(agent.agent_index) and min_ghost_dist < GHOST_DANGER_RANGE

    # ── 캡슐 ─────────────────────────────────────────────────
    enemy_caps = _enemy_capsules(state, agent.agent_index)
    if len(enemy_caps) > 0:
        cap_distances = [agent._distances.get_distance(current_position, cp) for cp in enemy_caps]
        cap_distances = [d for d in cap_distances if d is not None]
        if cap_distances:
            min_cap_dist = min(cap_distances)
            # [신규] 쫓기는 상황에서는 캡슐을 더욱 우선시
            if being_chased:
                features['capsule_when_chased'] = min_cap_dist
            features['distance_to_enemy_capsule'] = min_cap_dist
    else:
        features['distance_to_enemy_capsule'] = 0.0

    # ── 음식 ─────────────────────────────────────────────────
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

    return features