"""

======================
Complete standalone simulation — grid world mode with full trust framework integrated.
Based on: "Securing LLM Multi-Agent Systems via Dynamic Trust Management" (Arjun Vooturi)

HOW TO RUN:
    python sim_test_with_trust.py

No other files needed. Everything is in this one file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import heapq
import math
import random
import re
import json


# =========================================================
# TYPES
# =========================================================

Cell = Tuple[int, int]
RobotId = str


@dataclass
class Robot:
    robot_id: RobotId
    team: str
    position: Cell
    hp: int = 100
    alive: bool = True
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    distance_covered: float = 0.0
    last_position: Optional[Cell] = None


@dataclass
class ParsedCommand:
    raw_text: str
    action: str
    scope: str
    target: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LocalObservation:
    robot_id: RobotId
    team: str
    position: Cell
    nearby_teammates: List[Tuple[RobotId, Cell]]
    nearby_enemies: List[Tuple[RobotId, Cell]]
    nearby_obstacles: List[Cell]
    visible_regions: List[str]
    hp: int


@dataclass
class GlobalState:
    timestep: int
    blue_positions: Dict[RobotId, Cell]
    red_positions: Dict[RobotId, Cell]
    alive_blue: int
    alive_red: int
    blue_base: Cell
    red_base: Cell
    contested_regions: List[str]
    threat_map: Dict[RobotId, float]


@dataclass
class HumanPlan:
    parsed_commands: List[ParsedCommand] = field(default_factory=list)


@dataclass
class Metrics:
    kills: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))
    deaths: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))
    damage_dealt: Dict[RobotId, float] = field(default_factory=lambda: defaultdict(float))
    damage_taken: Dict[RobotId, float] = field(default_factory=lambda: defaultdict(float))
    objective_captures: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))
    command_compliance: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))
    command_violations: Dict[RobotId, int] = field(default_factory=lambda: defaultdict(int))
    command_prev_dist: Dict[RobotId, Optional[float]] = field(default_factory=dict)
    command_last_tgt: Dict[RobotId, Optional[Tuple[int, int]]] = field(default_factory=dict)
    winning_rate_history: List[Tuple[int, float]] = field(default_factory=list)
    situation_history: List[Tuple[int, str]] = field(default_factory=list)


@dataclass
class PlottingData:
    timestep: int
    winning_rate: float
    situation_tag: str
    strategy_suggested: str
    total_compliance: int
    total_violations: int
    score: int
    blue_alive: int
    red_alive: int
    contested_regions: List[str]
    threat_level: float

    def to_dict(self):
        return asdict(self)


@dataclass
class PositioningConfig:
    min_separation: int = 1


# =========================================================
# ENVIRONMENT
# =========================================================

GRID_W = 10
GRID_H = 10

GRID = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

BLUE_BASE = (0, 0)
RED_BASE  = (9, 9)

REGIONS = {
    "R1":     [(1, 1), (1, 2), (2, 1), (2, 2)],
    "R2":     [(7, 7), (7, 8), (8, 7), (8, 8)],
    "CENTER": [(4, 4), (4, 5), (5, 4), (5, 5)],
}


@dataclass
class EnvironmentState:
    blue_robots: Dict[RobotId, Robot]
    red_robots:  Dict[RobotId, Robot]
    timestep:  int  = 0
    max_steps: int  = 20
    done:      bool = False


# =========================================================
# TRUST FRAMEWORK  (PPT: Dynamic Trust Management)
# =========================================================

TRUST_INITIAL         = 0.7
TRUST_CORRECT_DELTA   = +0.1
TRUST_INCORRECT_DELTA = -0.2
TRUST_THRESHOLD       = 0.5
TRUST_MIN             = 0.0
TRUST_MAX             = 1.0


@dataclass
class AgentTrustRecord:
    agent_id: str
    team: str
    score: float = TRUST_INITIAL
    rounds_active: int = 0
    correct_outputs: int = 0
    incorrect_outputs: int = 0
    verifier_intercepts: int = 0
    converged_at_round: Optional[int] = None
    score_history: List[Tuple[int, float]] = field(default_factory=list)

    def update(self, correct: bool, timestep: int) -> float:
        self.rounds_active += 1
        if correct:
            self.correct_outputs += 1
            self.score = min(TRUST_MAX, self.score + TRUST_CORRECT_DELTA)
        else:
            self.incorrect_outputs += 1
            self.score = max(TRUST_MIN, self.score + TRUST_INCORRECT_DELTA)
        self.score_history.append((timestep, round(self.score, 3)))
        if self.score < TRUST_THRESHOLD and self.converged_at_round is None:
            self.converged_at_round = timestep
        return self.score

    @property
    def is_low_trust(self) -> bool:
        return self.score < TRUST_THRESHOLD


@dataclass
class VerifierDecision:
    timestep: int
    agent_id: str
    trust_score: float
    action_proposed: str
    accepted: bool
    reason: str


@dataclass
class TrustMetrics:
    total_decisions: int = 0
    wrong_accepted: int = 0
    correct_accepted: int = 0
    wrong_blocked: int = 0
    verifier_activations: int = 0
    task_attempts: int = 0
    task_successes: int = 0
    convergence_rounds: List[int] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def error_propagation_rate(self) -> float:
        if self.total_decisions == 0:
            return 0.0
        return round(self.wrong_accepted / self.total_decisions, 4)

    @property
    def mean_trust_convergence(self) -> Optional[float]:
        if not self.convergence_rounds:
            return None
        return round(sum(self.convergence_rounds) / len(self.convergence_rounds), 2)

    @property
    def incorrect_decisions_accepted(self) -> int:
        return self.wrong_accepted

    @property
    def task_success_rate(self) -> float:
        if self.task_attempts == 0:
            return 1.0
        return round(self.task_successes / self.task_attempts, 4)

    def snapshot(self, timestep: int, trust_records: Dict[str, AgentTrustRecord]):
        self.history.append({
            "t": timestep,
            "error_propagation_rate": self.error_propagation_rate,
            "wrong_accepted": self.wrong_accepted,
            "correct_accepted": self.correct_accepted,
            "wrong_blocked": self.wrong_blocked,
            "task_success_rate": self.task_success_rate,
            "trust_scores": {aid: rec.score for aid, rec in trust_records.items()},
            "low_trust_agents": [aid for aid, rec in trust_records.items() if rec.is_low_trust],
        })

    def report(self) -> str:
        lines = [
            "",
            "===== TRUST FRAMEWORK METRICS (PPT Section 4) =====",
            f"  Metric 1 - Error Propagation Rate  : {self.error_propagation_rate:.1%}",
            f"  Metric 2 - Trust Convergence       : {self.mean_trust_convergence} rounds"
                + (" (never flagged)" if self.mean_trust_convergence is None else ""),
            f"  Metric 3 - Wrong Decisions Accepted: {self.incorrect_decisions_accepted}",
            f"  Metric 4 - Task Success Rate       : {self.task_success_rate:.1%}",
            f"  Verifier activations               : {self.verifier_activations}",
            "====================================================",
        ]
        return "\n".join(lines)


class TrustRegistry:
    def __init__(self):
        self._records: Dict[str, AgentTrustRecord] = {}
        self.metrics = TrustMetrics()

    def register(self, agent_id: str, team: str):
        if agent_id not in self._records:
            self._records[agent_id] = AgentTrustRecord(agent_id=agent_id, team=team)

    def score(self, agent_id: str) -> float:
        return self._records[agent_id].score if agent_id in self._records else TRUST_INITIAL

    def is_low_trust(self, agent_id: str) -> bool:
        return self.score(agent_id) < TRUST_THRESHOLD

    def all_scores(self) -> Dict[str, float]:
        return {aid: rec.score for aid, rec in self._records.items()}

    def low_trust_agents(self) -> List[str]:
        return [aid for aid, rec in self._records.items() if rec.is_low_trust]

    def update_trust(self, agent_id: str, correct: bool, timestep: int) -> float:
        if agent_id not in self._records:
            return TRUST_INITIAL
        new_score = self._records[agent_id].update(correct, timestep)
        rec = self._records[agent_id]
        if rec.converged_at_round == timestep:
            self.metrics.convergence_rounds.append(timestep)
        return new_score

    def snapshot(self, timestep: int):
        self.metrics.snapshot(timestep, self._records)

    def print_scores(self):
        print("[TrustRegistry] Agent scores:")
        for aid, rec in sorted(self._records.items()):
            flag = " <<< LOW-TRUST" if rec.is_low_trust else ""
            print(f"  {aid} ({rec.team}): {rec.score:.2f}{flag}")


class VerifierAgent:
    def __init__(self, registry: TrustRegistry):
        self.registry = registry
        self.decisions: List[VerifierDecision] = []

    def verify(self, agent_id: str, proposed_action: str,
               context: Dict[str, Any], timestep: int) -> Tuple[bool, str]:
        trust = self.registry.score(agent_id)
        self.registry.metrics.total_decisions += 1

        if trust >= TRUST_THRESHOLD:
            self._record(timestep, agent_id, trust, proposed_action, True, "trusted")
            self.registry.metrics.correct_accepted += 1
            self.registry.metrics.task_successes += 1
            self.registry.metrics.task_attempts += 1
            self.registry.update_trust(agent_id, correct=True, timestep=timestep)
            return True, "trusted"

        # Low-trust path — Verifier evaluates
        self.registry.metrics.verifier_activations += 1
        self.registry.metrics.task_attempts += 1
        accepted, reason = self._evaluate(proposed_action, context)

        if accepted:
            self.registry.metrics.correct_accepted += 1
            self.registry.metrics.task_successes += 1
            self.registry.update_trust(agent_id, correct=True, timestep=timestep)
        else:
            self.registry.metrics.wrong_blocked += 1
            self.registry.update_trust(agent_id, correct=False, timestep=timestep)

        self._record(timestep, agent_id, trust, proposed_action, accepted, reason)
        return accepted, reason

    def _evaluate(self, proposed_action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        strategic_goal = context.get("strategic_goal", "")
        has_flag       = context.get("has_flag", False)
        own_base       = context.get("own_base")
        target_pos     = context.get("target_position")
        current_pos    = context.get("current_position")
        action_lower   = proposed_action.lower()

        if has_flag and own_base and target_pos and current_pos:
            dist_now = _trust_dist(current_pos, own_base)
            dist_tgt = _trust_dist(target_pos, own_base)
            if dist_tgt > dist_now + 2:
                return False, "flag-carrier moving away from base"

        if strategic_goal == "defend" and "attack" in action_lower:
            return False, "defend-agent issuing attack (goal mismatch)"

        if strategic_goal == "attack" and own_base and target_pos:
            if own_base == target_pos:
                return False, "attack-agent retreating to own base"

        if strategic_goal == "hold" and "spread" in action_lower:
            return False, "hold-agent spreading (abandons zone)"

        return True, "verified-ok"

    def verify_blue_targets(self, blue_targets: Dict[str, Any],
                            env_context: Dict[str, Any],
                            timestep: int) -> Tuple[Dict[str, Any], List[str]]:
        filtered = {}
        blocked  = []
        for rid, target in blue_targets.items():
            ctx = {**env_context, "agent_id": rid, "target_position": target}
            accepted, reason = self.verify(rid, f"move_to_{target}", ctx, timestep)
            if accepted:
                filtered[rid] = target
            else:
                print(f"  [Verifier] BLOCKED {rid}: {reason} (trust={self.registry.score(rid):.2f})")
                blocked.append(rid)
                fallback = env_context.get("current_positions", {}).get(rid, target)
                filtered[rid] = fallback
        return filtered, blocked

    def report_outcome(self, agent_id: str, was_correct: bool, timestep: int):
        if was_correct:
            self.registry.metrics.task_successes += 1
        else:
            self.registry.metrics.wrong_accepted += 1
        self.registry.update_trust(agent_id, correct=was_correct, timestep=timestep)

    def _record(self, timestep, agent_id, trust, action, accepted, reason):
        self.decisions.append(VerifierDecision(
            timestep=timestep, agent_id=agent_id,
            trust_score=round(trust, 3), action_proposed=action,
            accepted=accepted, reason=reason,
        ))


def _trust_dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def init_trust_for_grid(env: EnvironmentState) -> Tuple[TrustRegistry, VerifierAgent]:
    registry = TrustRegistry()
    for rid, robot in env.blue_robots.items():
        registry.register(rid, robot.team)
    verifier = VerifierAgent(registry)
    print("\n[TrustFramework] Initialized. Starting scores:")
    registry.print_scores()
    return registry, verifier


def trust_filter_blue_targets(registry: TrustRegistry, verifier: VerifierAgent,
                               blue_targets: Dict[str, Any], env: EnvironmentState,
                               timestep: int) -> Dict[str, Any]:
    current_positions = {rid: r.position for rid, r in env.blue_robots.items() if r.alive}
    env_context = {
        "own_base": BLUE_BASE,
        "enemy_base": RED_BASE,
        "current_positions": current_positions,
        "team": "BLUE",
    }
    filtered, _ = verifier.verify_blue_targets(blue_targets, env_context, timestep)
    registry.snapshot(timestep)
    return filtered


def update_trust_after_combat(registry: TrustRegistry, verifier: VerifierAgent,
                               env: EnvironmentState, timestep: int):
    for rid, robot in env.blue_robots.items():
        if rid not in registry.all_scores():
            continue
        if not robot.alive:
            verifier.report_outcome(rid, was_correct=False, timestep=timestep)
        elif robot.kills > 0:
            verifier.report_outcome(rid, was_correct=True, timestep=timestep)


# =========================================================
# ADVERSARIAL AGENT SIMULATION  (PPT Section 4 — Executor B)
# =========================================================
# The PPT experiment: one agent (B2) is configured to produce
# incorrect outputs at a fixed probability, simulating prompt
# injection or misalignment.  The trust framework should
# detect and flag it within ~4 rounds automatically.

ADVERSARIAL_AGENT    = "B2"          # Which blue robot is adversarial
ADVERSARIAL_PROB     = 0.6           # 60% chance of bad output each round
                                     # (matches PPT "fixed error probability")

def simulate_adversarial_outputs(registry: TrustRegistry,
                                  verifier: VerifierAgent,
                                  env: EnvironmentState,
                                  timestep: int,
                                  adversarial_agent: str = ADVERSARIAL_AGENT,
                                  error_prob: float = ADVERSARIAL_PROB):
    """
    Simulates the PPT Section 4 experiment:
      - Adversarial agent (B2) randomly injects bad outputs
      - Trust score updated accordingly
      - Verifier intercepts once score < 0.5
    Called every step BEFORE trust_filter_blue_targets.
    """
    print(f"\n  [Adversarial Sim] Checking {adversarial_agent} (error_prob={error_prob:.0%}):")

    for rid in list(env.blue_robots.keys()):
        robot = env.blue_robots[rid]
        if not robot.alive:
            continue

        if rid == adversarial_agent:
            # Adversarial agent: randomly produces wrong output
            is_bad = random.random() < error_prob
            if is_bad:
                registry.update_trust(rid, correct=False, timestep=timestep)
                registry.metrics.wrong_accepted += 1
                print(f"    {rid} [ADVERSARIAL]: injected BAD output  "
                      f"-> score now {registry.score(rid):.2f}"
                      + (" <<< FLAGGED" if registry.is_low_trust(rid) else ""))
            else:
                registry.update_trust(rid, correct=True, timestep=timestep)
                print(f"    {rid} [ADVERSARIAL]: produced good output  "
                      f"-> score now {registry.score(rid):.2f}")
        else:
            # Normal agents always produce correct output
            registry.update_trust(rid, correct=True, timestep=timestep)
            print(f"    {rid} [trusted]:     correct output          "
                  f"-> score now {registry.score(rid):.2f}")


def print_trust_status(registry: TrustRegistry, timestep: int):
    print(f"\n[Trust t={timestep}]", end=" ")
    scores = registry.all_scores()
    for aid, s in sorted(scores.items()):
        flag = " !" if registry.is_low_trust(aid) else ""
        print(f"{aid}={s:.2f}{flag}", end="  ")
    print()
    low = registry.low_trust_agents()
    if low:
        print(f"  >>> Low-trust agents (below {TRUST_THRESHOLD}): {low}")


def print_trust_report(registry: TrustRegistry):
    print(registry.metrics.report())
    print("\n[TrustFramework] Per-agent final summary:")
    for aid, rec in sorted(registry._records.items()):
        conv = f"flagged @ t={rec.converged_at_round}" if rec.converged_at_round else "never flagged"
        print(f"  {aid}: score={rec.score:.2f} | "
              f"correct={rec.correct_outputs} wrong={rec.incorrect_outputs} | {conv}")


def export_trust_data(registry: TrustRegistry, filename: str = "trust_data.json"):
    data = {
        "metrics": {
            "error_propagation_rate": registry.metrics.error_propagation_rate,
            "mean_trust_convergence_rounds": registry.metrics.mean_trust_convergence,
            "incorrect_decisions_accepted": registry.metrics.incorrect_decisions_accepted,
            "task_success_rate": registry.metrics.task_success_rate,
            "verifier_activations": registry.metrics.verifier_activations,
            "total_decisions": registry.metrics.total_decisions,
        },
        "agent_records": {
            aid: {
                "team": rec.team,
                "final_score": rec.score,
                "correct_outputs": rec.correct_outputs,
                "incorrect_outputs": rec.incorrect_outputs,
                "converged_at_round": rec.converged_at_round,
                "score_history": rec.score_history,
            }
            for aid, rec in registry._records.items()
        },
        "history": registry.metrics.history,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[TrustFramework] Trust data saved to {filename}")


# =========================================================
# HELPERS
# =========================================================

def in_bounds(cell: Cell) -> bool:
    x, y = cell
    return 0 <= x < GRID_H and 0 <= y < GRID_W


def is_free(cell: Cell) -> bool:
    x, y = cell
    return in_bounds(cell) and GRID[x][y] == 0


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Cell, b: Cell) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_neighbors(cell: Cell) -> List[Cell]:
    x, y = cell
    candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [c for c in candidates if is_free(c)]


def nearest_cell(start: Cell, cells: List[Cell]) -> Cell:
    return min(cells, key=lambda c: manhattan(start, c))


def visible_obstacles(position: Cell, radius: int = 2) -> List[Cell]:
    px, py = position
    out = []
    for x in range(max(0, px-radius), min(GRID_H, px+radius+1)):
        for y in range(max(0, py-radius), min(GRID_W, py+radius+1)):
            if GRID[x][y] == 1:
                out.append((x, y))
    return out


def visible_regions(position: Cell, radius: int = 2) -> List[str]:
    out = []
    for region_name, cells in REGIONS.items():
        for c in cells:
            if manhattan(position, c) <= radius:
                out.append(region_name)
                break
    return out


# =========================================================
# BLOCK 1 — ENVIRONMENT INITIALIZATION
# =========================================================

def block1_environment_initialization() -> EnvironmentState:
    blue_robots = {
        "B1": Robot("B1", "BLUE", (0, 0)),
        "B2": Robot("B2", "BLUE", (0, 1)),
        "B3": Robot("B3", "BLUE", (1, 0)),
    }
    red_robots = {
        "R1": Robot("R1", "RED", (9, 9)),
        "R2": Robot("R2", "RED", (9, 8)),
        "R3": Robot("R3", "RED", (8, 9)),
    }
    return EnvironmentState(
        blue_robots=blue_robots,
        red_robots=red_robots,
        timestep=0,
        max_steps=20,
        done=False,
    )


# =========================================================
# BLOCK 2 — LOCAL OBSERVATION
# =========================================================

def block2_local_observation(env: EnvironmentState) -> Dict[RobotId, LocalObservation]:
    obs: Dict[RobotId, LocalObservation] = {}
    all_robots = {**env.blue_robots, **env.red_robots}
    for rid, robot in all_robots.items():
        if not robot.alive:
            continue
        teammates = []
        enemies   = []
        for oid, other in all_robots.items():
            if oid == rid or not other.alive:
                continue
            if manhattan(robot.position, other.position) <= 3:
                if other.team == robot.team:
                    teammates.append((oid, other.position))
                else:
                    enemies.append((oid, other.position))
        obs[rid] = LocalObservation(
            robot_id=rid, team=robot.team, position=robot.position,
            nearby_teammates=teammates, nearby_enemies=enemies,
            nearby_obstacles=visible_obstacles(robot.position),
            visible_regions=visible_regions(robot.position),
            hp=robot.hp,
        )
    return obs


# =========================================================
# BLOCK 3 — GLOBAL STATE ENCODING
# =========================================================

def block3_global_state_encoding(env: EnvironmentState,
                                  local_obs: Dict[RobotId, LocalObservation]) -> GlobalState:
    blue_positions = {rid: r.position for rid, r in env.blue_robots.items() if r.alive}
    red_positions  = {rid: r.position for rid, r in env.red_robots.items()  if r.alive}

    contested = []
    for region_name, cells in REGIONS.items():
        blue_in = any(pos in cells for pos in blue_positions.values())
        red_in  = any(pos in cells for pos in red_positions.values())
        if blue_in and red_in:
            contested.append(region_name)

    threat_map = {}
    for rid, robot in env.red_robots.items():
        if robot.alive:
            threat_map[rid] = float(robot.kills + 1) / float(robot.deaths + 1)

    return GlobalState(
        timestep=env.timestep,
        blue_positions=blue_positions,
        red_positions=red_positions,
        alive_blue=sum(1 for r in env.blue_robots.values() if r.alive),
        alive_red=sum(1 for r in env.red_robots.values()  if r.alive),
        blue_base=BLUE_BASE,
        red_base=RED_BASE,
        contested_regions=contested,
        threat_map=threat_map,
    )


# =========================================================
# BLOCK 4 — SYSTEM SUMMARY + STRATEGY
# =========================================================

def block4_system_summary(global_state: GlobalState) -> str:
    summary = [
        f"Timestep {global_state.timestep}",
        f"Blue alive: {global_state.alive_blue}",
        f"Red alive: {global_state.alive_red}",
    ]
    if global_state.contested_regions:
        summary.append(f"Contested: {', '.join(global_state.contested_regions)}")
    else:
        summary.append("No contested regions")
    if global_state.threat_map:
        highest = max(global_state.threat_map, key=global_state.threat_map.get)
        summary.append(f"Highest red threat: {highest}")
    return " | ".join(summary)


def calculate_winning_rate(global_state: GlobalState) -> float:
    blue  = global_state.alive_blue
    red   = global_state.alive_red
    total = blue + red
    if total == 0:
        return 0.5
    base = blue / total
    adj  = 0.03 * len(global_state.contested_regions)
    if global_state.threat_map:
        avg_threat = sum(global_state.threat_map.values()) / len(global_state.threat_map)
        if avg_threat > 1.5:
            adj -= 0.05 * (avg_threat - 1.0)
    return max(0.05, min(0.95, base + adj))


def determine_situation_tag(global_state: GlobalState, max_steps: int = 20) -> str:
    t    = global_state.timestep
    blue = global_state.alive_blue
    red  = global_state.alive_red
    tags = []
    if t < 3:
        tags.append("early_game")
    elif t > max_steps - 3:
        tags.append("end_game")
    if blue > red + 1:
        tags.append("dominating")
    elif red > blue + 1:
        tags.append("outnumbered")
    elif blue == red and blue > 0:
        tags.append("balanced")
    if global_state.contested_regions:
        tags.append("contested")
    if blue <= 1:
        tags.append("critical_blue")
    if red  <= 1:
        tags.append("critical_red")
    return "_".join(tags) if tags else "standard"


def generate_llm_summary(env: EnvironmentState, global_state: GlobalState) -> str:
    win_rate = calculate_winning_rate(global_state)
    parts = [
        f"Turn {global_state.timestep}",
        f"Forces: Blue({global_state.alive_blue}) vs Red({global_state.alive_red})",
        f"BLUE Win Rate: {win_rate:.0%}",
    ]
    if global_state.contested_regions:
        parts.append(f"HotZones: {','.join(global_state.contested_regions)}")
    if global_state.threat_map:
        top = max(global_state.threat_map.items(), key=lambda x: x[1])
        parts.append(f"PriorityTarget: {top[0]}(threat:{top[1]:.1f})")

    if global_state.alive_blue > global_state.alive_red + 1:
        situation = "Numerical superiority - push recommended"
    elif global_state.alive_red > global_state.alive_blue + 1:
        situation = "Outnumbered - consolidate defensive positions"
    elif env.timestep < 3:
        situation = "Opening phase - establish map control"
    elif env.timestep > env.max_steps - 3:
        situation = "Endgame - secure all objectives"
    else:
        situation = "Tactical stalemate - probe for flanking"

    return " | ".join(parts) + f"\nSITUATION: {situation}"


def generate_top3_strategies(global_state: GlobalState) -> str:
    blue      = global_state.alive_blue
    red       = global_state.alive_red
    contested = global_state.contested_regions

    lines = ["Top 3 strategies:"]

    if blue > red:
        lines.append("  1. [MEDIUM] Numerical Press — exploit advantage, force engagement")
    elif red > blue:
        lines.append("  1. [LOW]    Guerrilla Protocol — avoid fair fights, use terrain")
    else:
        lines.append("  1. [MEDIUM] Probing Engagement — test enemy, force positional errors")

    if contested:
        lines.append(f" 2. [HIGH]   Secure {contested[0]} — overwatch then push")
    else:
        lines.append("  2. [MEDIUM] Create Pressure — threaten enemy flank")

    lines.append("  3. [LOW]    Bait Protocol — fake weakness, draw enemy out of position")
    return "\n".join(lines)


# =========================================================
# BLOCK 5 — HUMAN COMMAND INTERFACE
# =========================================================

def parse_blue_command(text: str) -> Optional[ParsedCommand]:
    s = text.strip().lower()

    m = re.match(r"^all\s+(.*)$", s)
    if m:
        scope, target, rest = "all", "all", m.group(1)
    else:
        m = re.match(r"^robot\s+([a-zA-Z0-9_]+)\s+(.*)$", s)
        if not m:
            return None
        scope  = "robot"
        target = m.group(1).upper()
        rest   = m.group(2)

    m = re.match(r"^move\s+to\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "move", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})

    m = re.match(r"^hold\s+region\s+([a-zA-Z0-9_]+)$", rest)
    if m:
        return ParsedCommand(text, "hold", scope, target, {"region": m.group(1).upper()})

    m = re.match(r"^defend\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "defend", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})

    m = re.match(r"^attack\s+([a-zA-Z0-9_]+)$", rest)
    if m:
        t = m.group(1).upper()
        if t == "RED_BASE":
            return ParsedCommand(text, "attack", scope, target, {"cell": RED_BASE})
        else:
            return ParsedCommand(text, "attack", scope, target, {"enemy_target": t})

    m = re.match(r"^regroup\s+at\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "regroup", scope, target, {"cell": (int(m.group(1)), int(m.group(2)))})

    if re.match(r"^spread$", rest):
        return ParsedCommand(text, "spread", scope, target, {})

    return None


def block5_human_command_interface(env: EnvironmentState, global_state: GlobalState,
                                    first_turn: bool = False) -> HumanPlan:
    if first_turn:
        print("\n" + "="*50)
        print("  BLUE TEAM COMMAND INTERFACE")
        print("="*50)
    else:
        print("\nEnter updated commands or press Enter to keep current strategy.")

    print("\nCommands:")
    print("  all move to <row> <col>        e.g.  all move to 4 4")
    print("  all attack red_base")
    print("  all hold region CENTER")
    print("  all hold region R1")
    print("  all regroup at <row> <col>")
    print("  all spread")
    print("  robot B1 move to <row> <col>")
    print("  0  -> show situation summary")
    print("  9  -> show top 3 strategies")
    print("  Enter or 'done' -> finish\n")

    commands: List[ParsedCommand] = []

    while True:
        text = input("BLUE CMD > ").strip()

        if text == "0":
            print("\n" + generate_llm_summary(env, global_state))
            win  = calculate_winning_rate(global_state)
            sit  = determine_situation_tag(global_state, env.max_steps)
            print(f"Win Rate: {win:.1%} | Situation: {sit}\n")
            continue

        if text == "9":
            print("\n" + generate_top3_strategies(global_state) + "\n")
            continue

        if text == "" or text.lower() == "done":
            break

        parsed = parse_blue_command(text)
        if parsed is None:
            print("  Invalid command. Try: all move to 4 4")
            continue

        commands.append(parsed)
        print(f"  Accepted: {parsed.action} -> {parsed.params}")

    return HumanPlan(parsed_commands=commands)


# =========================================================
# BLOCK 6 — BLUE STRATEGY ASSIGNMENT
# =========================================================

def command_applies(cmd: ParsedCommand, robot_id: str) -> bool:
    return cmd.scope == "all" or cmd.target == robot_id


def block6_blue_strategy_assignment(env: EnvironmentState,
                                     human_plan: HumanPlan) -> Dict[RobotId, Cell]:
    blue_targets: Dict[RobotId, Cell] = {}
    for rid, robot in env.blue_robots.items():
        if robot.alive:
            blue_targets[rid] = robot.position

    for cmd in human_plan.parsed_commands:
        for rid, robot in env.blue_robots.items():
            if not robot.alive or not command_applies(cmd, rid):
                continue
            if cmd.action in ("move", "defend", "regroup"):
                blue_targets[rid] = cmd.params["cell"]
            elif cmd.action == "hold":
                region_cells = REGIONS.get(cmd.params["region"], [robot.position])
                blue_targets[rid] = nearest_cell(robot.position, region_cells)
            elif cmd.action == "attack":
                if "enemy_target" in cmd.params:
                    eid = cmd.params["enemy_target"]
                    if eid in env.red_robots and env.red_robots[eid].alive:
                        blue_targets[rid] = env.red_robots[eid].position
                    else:
                        blue_targets[rid] = RED_BASE
                else:
                    blue_targets[rid] = cmd.params["cell"]
            elif cmd.action == "spread":
                spread_cells = [(1, 8), (8, 1), (1, 5), (5, 1), (8, 5), (5, 8)]
                idx = list(env.blue_robots.keys()).index(rid) % len(spread_cells)
                blue_targets[rid] = spread_cells[idx]

    return blue_targets


# =========================================================
# BLOCK 7 — RED AUTONOMOUS STRATEGY
# =========================================================

def block7_red_autonomous_strategy(env: EnvironmentState) -> Dict[RobotId, Cell]:
    red_targets: Dict[RobotId, Cell] = {}
    blue_alive  = [r for r in env.blue_robots.values() if r.alive]
    center_cells = REGIONS["CENTER"]

    for rid, red_robot in env.red_robots.items():
        if not red_robot.alive:
            continue
        if not blue_alive:
            red_targets[rid] = BLUE_BASE
            continue
        nearest_blue = min(blue_alive, key=lambda b: manhattan(red_robot.position, b.position))
        if manhattan(red_robot.position, nearest_blue.position) <= 4:
            red_targets[rid] = nearest_blue.position
        else:
            red_targets[rid] = nearest_cell(red_robot.position, center_cells)

    return red_targets


# =========================================================
# BLOCK 8 — A* PATH PLANNING
# =========================================================

def astar(start: Cell, goal: Cell) -> List[Cell]:
    if not is_free(start) or not is_free(goal):
        return []
    open_heap: List[Tuple[int, Cell]] = []
    heapq.heappush(open_heap, (0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int]    = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for nb in get_neighbors(current):
            tentative = g_score[current] + 1
            if nb not in g_score or tentative < g_score[nb]:
                came_from[nb] = current
                g_score[nb]   = tentative
                heapq.heappush(open_heap, (tentative + manhattan(nb, goal), nb))
    return []


def block8_path_planning(env: EnvironmentState,
                          blue_targets: Dict[RobotId, Cell],
                          red_targets:  Dict[RobotId, Cell]) -> Dict[RobotId, List[Cell]]:
    all_paths: Dict[RobotId, List[Cell]] = {}
    for rid, robot in env.blue_robots.items():
        if robot.alive:
            all_paths[rid] = astar(robot.position, blue_targets.get(rid, robot.position))
    for rid, robot in env.red_robots.items():
        if robot.alive:
            all_paths[rid] = astar(robot.position, red_targets.get(rid, robot.position))
    return all_paths


# =========================================================
# BLOCK 9 — POSITIONING & EXECUTION
# =========================================================

def system_positioning_adjust(current: Dict[RobotId, Cell],
                               proposed: Dict[RobotId, Cell],
                               cfg: PositioningConfig) -> Dict[RobotId, Cell]:
    final = dict(proposed)
    ids   = list(final.keys())
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            a, b = ids[i], ids[j]
            if final[a] == final[b]:
                final[b] = current[b]
            elif manhattan(final[a], final[b]) < cfg.min_separation:
                final[b] = current[b]
    return final


def next_step_from_path(path: List[Cell], current: Cell) -> Cell:
    return path[1] if len(path) >= 2 else current


def resolve_combat(env: EnvironmentState, metrics: Metrics):
    blue_alive = [r for r in env.blue_robots.values() if r.alive]
    red_alive  = [r for r in env.red_robots.values()  if r.alive]

    for b in blue_alive:
        for r in red_alive:
            if not b.alive or not r.alive:
                continue
            if manhattan(b.position, r.position) <= 1:
                if random.random() < 0.55:
                    r.alive = False
                    r.deaths += 1
                    b.kills  += 1
                    metrics.kills[b.robot_id]  += 1
                    metrics.deaths[r.robot_id] += 1
                    print(f"  {b.robot_id} (BLUE) eliminated {r.robot_id} (RED)")
                else:
                    b.alive = False
                    b.deaths += 1
                    r.kills  += 1
                    metrics.kills[r.robot_id]  += 1
                    metrics.deaths[b.robot_id] += 1
                    print(f"  {r.robot_id} (RED) eliminated {b.robot_id} (BLUE)")


def block9_positioning_and_execution(env: EnvironmentState,
                                      all_paths: Dict[RobotId, List[Cell]],
                                      blue_targets: Dict[RobotId, Cell],
                                      metrics: Metrics,
                                      human_plan: HumanPlan) -> EnvironmentState:
    current_pos  = {}
    proposed_pos = {}

    for rid, robot in {**env.blue_robots, **env.red_robots}.items():
        if robot.alive:
            current_pos[rid]  = robot.position
            proposed_pos[rid] = next_step_from_path(all_paths.get(rid, []), robot.position)

    adjusted = system_positioning_adjust(current_pos, proposed_pos, PositioningConfig())

    for rid, robot in env.blue_robots.items():
        if robot.alive:
            old = robot.position
            robot.position      = adjusted.get(rid, robot.position)
            robot.last_position = old
            robot.distance_covered += euclidean(old, robot.position)

    for rid, robot in env.red_robots.items():
        if robot.alive:
            old = robot.position
            robot.position      = adjusted.get(rid, robot.position)
            robot.last_position = old
            robot.distance_covered += euclidean(old, robot.position)

    _eps = 1e-6
    if human_plan.parsed_commands:
        for rid, robot in env.blue_robots.items():
            if not robot.alive or rid not in blue_targets:
                metrics.command_prev_dist.pop(rid, None)
                metrics.command_last_tgt.pop(rid, None)
                continue
            tgt  = blue_targets[rid]
            tkey = (int(tgt[0]), int(tgt[1]))
            d    = float(euclidean(robot.position, tgt))
            if metrics.command_last_tgt.get(rid) != tkey:
                metrics.command_last_tgt[rid]  = tkey
                metrics.command_prev_dist[rid] = None
            prev = metrics.command_prev_dist.get(rid)
            if prev is None:
                metrics.command_prev_dist[rid] = d
                continue
            if robot.position == tgt or d <= _eps:
                metrics.command_compliance[rid] += 1
            elif d > prev + _eps:
                metrics.command_violations[rid] += 1
            elif d < prev - _eps:
                metrics.command_compliance[rid] += 1
            metrics.command_prev_dist[rid] = d

    resolve_combat(env, metrics)

    for rid, robot in env.blue_robots.items():
        if robot.alive and robot.position in REGIONS["CENTER"]:
            metrics.objective_captures[rid] += 1

    return env


# =========================================================
# BLOCK 10 — STRATEGIC CHANGE DETECTION
# =========================================================

def block10_strategic_change_detection(env: EnvironmentState,
                                        global_state: GlobalState) -> bool:
    if global_state.alive_blue <= 1:
        return True
    if global_state.alive_red  <= 1:
        return True
    if global_state.contested_regions:
        return True
    return False


# =========================================================
# BLOCK 11 — METRICS UPDATE
# =========================================================

def block11_metrics_update(env: EnvironmentState, metrics: Metrics,
                            global_state: Optional[GlobalState] = None) -> Dict[str, Any]:
    player_rows = []
    for robot in list(env.blue_robots.values()) + list(env.red_robots.values()):
        kills  = metrics.kills[robot.robot_id]
        deaths = max(1, metrics.deaths[robot.robot_id])
        player_rows.append({
            "robot_id":           robot.robot_id,
            "team":               robot.team,
            "alive":              robot.alive,
            "kills":              kills,
            "deaths":             metrics.deaths[robot.robot_id],
            "kd":                 round(kills / deaths, 2),
            "distance_covered":   round(robot.distance_covered, 2),
            "objective_captures": metrics.objective_captures[robot.robot_id],
            "command_compliance": metrics.command_compliance[robot.robot_id],
            "command_violations": metrics.command_violations[robot.robot_id],
        })

    def team_synergy(team_robots):
        alive = [r for r in team_robots.values() if r.alive]
        if len(alive) < 2:
            return 0.0
        dists = [manhattan(alive[i].position, alive[j].position)
                 for i in range(len(alive)) for j in range(i+1, len(alive))]
        return round(1.0 / (1.0 + sum(dists)/len(dists)), 3)

    win_rate = 0.5
    situation = "unknown"
    if global_state:
        win_rate  = calculate_winning_rate(global_state)
        situation = determine_situation_tag(global_state, env.max_steps)
        metrics.winning_rate_history.append((env.timestep, win_rate))
        metrics.situation_history.append((env.timestep, situation))

    return {
        "player_metrics": player_rows,
        "blue_synergy":   team_synergy(env.blue_robots),
        "red_synergy":    team_synergy(env.red_robots),
        "winning_rate":   round(win_rate, 3),
        "situation_tag":  situation,
    }


# =========================================================
# BLOCK 12 — TERMINATION CHECK
# =========================================================

def block12_termination_check(env: EnvironmentState) -> bool:
    blue_alive = any(r.alive for r in env.blue_robots.values())
    red_alive  = any(r.alive for r in env.red_robots.values())
    if not blue_alive or not red_alive:
        env.done = True
    elif env.timestep >= env.max_steps:
        env.done = True
    else:
        env.done = False
    return env.done


# =========================================================
# DISPLAY HELPERS
# =========================================================

def print_grid(env: EnvironmentState):
    board = [["." if GRID[x][y] == 0 else "#" for y in range(GRID_W)]
             for x in range(GRID_H)]
    for rid, r in env.blue_robots.items():
        if r.alive:
            x, y = r.position
            board[x][y] = rid[-1]      # "1", "2", "3"
    for rid, r in env.red_robots.items():
        if r.alive:
            x, y = r.position
            board[x][y] = rid[-1].lower()   # "1", "2", "3" (lowercase = red)
    print("\nGrid (BLUE=1/2/3  RED=1/2/3-lower  #=wall):")
    for row in board:
        print(" ".join(row))


def print_status(env: EnvironmentState, summary: str, metrics_out: Dict[str, Any]):
    print(f"\n{'='*50}")
    print(f"  STEP {env.timestep}")
    print(f"{'='*50}")
    print(summary)
    print("\nBLUE TEAM:")
    for r in env.blue_robots.values():
        status = "ALIVE" if r.alive else "dead"
        print(f"  {r.robot_id}: pos={r.position}  {status}  kills={r.kills}  deaths={r.deaths}")
    print("\nRED TEAM:")
    for r in env.red_robots.values():
        status = "ALIVE" if r.alive else "dead"
        print(f"  {r.robot_id}: pos={r.position}  {status}  kills={r.kills}  deaths={r.deaths}")
    print(f"\nBLUE Win Rate : {metrics_out.get('winning_rate', 'N/A')}")
    print(f"Situation     : {metrics_out.get('situation_tag', 'N/A')}")
    print(f"Team Synergy  : BLUE={metrics_out['blue_synergy']}  RED={metrics_out['red_synergy']}")
    print_grid(env)


def export_plotting_data(metrics: Metrics, plotting_history: List[PlottingData],
                          filename: str = "plot_data.json"):
    data = {
        "winning_rate_trajectory": metrics.winning_rate_history,
        "situation_timeline":      metrics.situation_history,
        "detailed_steps":          [p.to_dict() for p in plotting_history],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[PlotData] Saved to {filename}  ({len(plotting_history)} timesteps)")


# =========================================================
# MAIN — GRID WORLD WITH TRUST FRAMEWORK
# =========================================================

def main():
    random.seed(42)
    plotting_history: List[PlottingData] = []

    print("\n" + "="*50)
    print("  MULTI-AGENT SIMULATION + TRUST FRAMEWORK")
    print("  Based on: Arjun Vooturi — AI Security Research")
    print("="*50)

    # Block 1: initialize environment
    env     = block1_environment_initialization()
    metrics = Metrics()

    # TRUST FRAMEWORK: initialize registry + verifier
    trust_registry, verifier = init_trust_for_grid(env)

    # Initial observation & state
    local_obs0    = block2_local_observation(env)
    global_state0 = block3_global_state_encoding(env, local_obs0)

    # First human command input
    human_plan = block5_human_command_interface(env, global_state0, first_turn=True)

    # ---- MAIN LOOP ----
    while True:
        env.timestep += 1

        local_obs    = block2_local_observation(env)
        global_state = block3_global_state_encoding(env, local_obs)
        summary      = block4_system_summary(global_state)

        # Block 10: replan if strategy change detected
        if env.timestep > 1:
            if block10_strategic_change_detection(env, global_state):
                print("\n[Block 10] Strategic change detected — replan?")
                new_plan = block5_human_command_interface(env, global_state, first_turn=False)
                if new_plan.parsed_commands:
                    human_plan = new_plan

        # Block 6: assign blue targets from human plan
        blue_targets = block6_blue_strategy_assignment(env, human_plan)

        # TRUST FRAMEWORK STEP 1: adversarial agent injects bad outputs (PPT Section 4)
        simulate_adversarial_outputs(trust_registry, verifier, env, env.timestep)

        # TRUST FRAMEWORK STEP 2: Verifier checks every blue target
        print(f"\n  [Verifier] Checking all blue targets at t={env.timestep}:")
        for rid, tgt in blue_targets.items():
            score = trust_registry.score(rid)
            status = "LOW-TRUST -> will verify" if trust_registry.is_low_trust(rid) else "trusted  -> fast-pass"
            print(f"    {rid}: target={tgt}  score={score:.2f}  [{status}]")

        blue_targets = trust_filter_blue_targets(
            trust_registry, verifier, blue_targets, env, env.timestep
        )

        # Block 7: red autonomous targets
        red_targets = block7_red_autonomous_strategy(env)

        # Block 8: path planning
        all_paths = block8_path_planning(env, blue_targets, red_targets)

        # Block 9: move + combat
        env = block9_positioning_and_execution(
            env, all_paths, blue_targets, metrics, human_plan
        )

        # TRUST FRAMEWORK STEP 3: update trust from combat outcome
        update_trust_after_combat(trust_registry, verifier, env, env.timestep)
        print_trust_status(trust_registry, env.timestep)

        # Block 11: metrics
        metrics_out = block11_metrics_update(env, metrics, global_state)

        # Collect plotting data
        total_comp = sum(metrics.command_compliance[r] for r in env.blue_robots)
        total_viol = sum(metrics.command_violations[r]  for r in env.blue_robots)
        score      = sum(metrics.objective_captures[r]  for r in env.blue_robots)
        max_threat = max(global_state.threat_map.values()) if global_state.threat_map else 0.0

        plotting_history.append(PlottingData(
            timestep=env.timestep,
            winning_rate=metrics_out["winning_rate"],
            situation_tag=metrics_out["situation_tag"],
            strategy_suggested="manual",
            total_compliance=total_comp,
            total_violations=total_viol,
            score=score,
            blue_alive=global_state.alive_blue,
            red_alive=global_state.alive_red,
            contested_regions=global_state.contested_regions,
            threat_level=max_threat,
        ))

        print_status(env, summary, metrics_out)

        if block12_termination_check(env):
            break

    # ---- END OF GAME ----
    export_plotting_data(metrics, plotting_history, "grid_world_plot_data.json")

    # TRUST FRAMEWORK: final report + export
    print_trust_report(trust_registry)
    export_trust_data(trust_registry, "grid_world_trust_data.json")

    print("\n===== FINAL ROBOT METRICS =====")
    for row in metrics_out["player_metrics"]:
        print(row)

    blue_alive = any(r.alive for r in env.blue_robots.values())
    red_alive  = any(r.alive for r in env.red_robots.values())

    if blue_alive and not red_alive:
        print("\n>>> BLUE team wins!")
    elif red_alive and not blue_alive:
        print("\n>>> RED team wins!")
    else:
        print("\n>>> Match ended by time limit / draw.")


if __name__ == "__main__":
    main()