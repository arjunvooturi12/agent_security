"""
sim_test_with_trust.py
======================
This is my CIS 6372 Information Assurance project implementation.
I built a multi-agent simulation to test how trust management can
protect agent communication from adversarial behavior.

My approach:
- I assign each agent a trust score that changes based on what they do
- If an agent's score drops below 0.5, my Verifier blocks their outputs
- I test this with one adversarial agent (B2) injecting bad data 60% of the time
- I measure 4 metrics to see if my defense actually works

How to run:
    python sim_test_with_trust.py

Everything is in this one file — no other imports needed.

Author: Arjun Vooturi
Course: CIS 6372 Information Assurance
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
# CUSTOM TYPES
# I defined these to make the code easier to read
# =========================================================

Cell    = Tuple[int, int]   # a grid position, e.g. (3, 5)
RobotId = str               # robot name like "B1" or "R2"


# =========================================================
# ROBOT DATA CLASS
# Each robot on the field has these properties
# =========================================================

@dataclass
class Robot:
    robot_id: RobotId
    team: str               # either "BLUE" (my team) or "RED" (opponent)
    position: Cell
    hp: int = 100
    alive: bool = True
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    distance_covered: float = 0.0
    last_position: Optional[Cell] = None


# =========================================================
# COMMAND PARSING
# I parse natural language commands like "all move to 4 4"
# into structured objects so the simulation can act on them
# =========================================================

@dataclass
class ParsedCommand:
    raw_text: str   # the original string the user typed
    action: str     # what to do: move, hold, attack, defend, spread, regroup
    scope: str      # "all" = all robots, "robot" = specific robot
    target: str     # "all" or a robot ID like "B1"
    params: Dict[str, Any] = field(default_factory=dict)  # extra info like destination cell


# =========================================================
# LOCAL OBSERVATION
# What one robot can see around it within a 3-cell radius
# =========================================================

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


# =========================================================
# GLOBAL STATE
# The full picture of the battlefield at any given timestep
# This is used for centralized training / decision making (CTDE)
# =========================================================

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
    threat_map: Dict[RobotId, float]  # how dangerous each red robot is


# =========================================================
# HUMAN PLAN
# Stores the list of commands I typed in during the simulation
# =========================================================

@dataclass
class HumanPlan:
    parsed_commands: List[ParsedCommand] = field(default_factory=list)


# =========================================================
# METRICS
# I track kills, deaths, compliance, and more for each robot
# =========================================================

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


# =========================================================
# PLOTTING DATA
# I collect this every step so I can graph things later
# =========================================================

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
    # minimum cells of separation between two friendly robots
    min_separation: int = 1


# =========================================================
# MAP / ENVIRONMENT SETUP
# I designed a 10x10 grid with walls (1 = wall, 0 = open)
# BLUE starts top-left, RED starts bottom-right
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

BLUE_BASE = (0, 0)  # top-left corner
RED_BASE  = (9, 9)  # bottom-right corner

# I defined 3 strategic regions that matter for scoring
REGIONS = {
    "R1":     [(1, 1), (1, 2), (2, 1), (2, 2)],   # near BLUE base
    "R2":     [(7, 7), (7, 8), (8, 7), (8, 8)],   # near RED base
    "CENTER": [(4, 4), (4, 5), (5, 4), (5, 5)],   # middle of the map
}


@dataclass
class EnvironmentState:
    # holds the full state of the simulation at any moment
    blue_robots: Dict[RobotId, Robot]
    red_robots:  Dict[RobotId, Robot]
    timestep:  int  = 0
    max_steps: int  = 20   # game ends after 20 steps
    done:      bool = False


# =========================================================
# MY TRUST FRAMEWORK
# This is the core of my CIS 6372 project.
#
# The idea: instead of trusting all agents blindly,
# I track a score for each agent. When they do something
# right, the score goes up. When they do something wrong,
# it goes down. If it falls below 0.5, I flag them and
# my Verifier agent takes over to check their actions.
#
# Score rules (based on my PPT Section 3 design):
#   Start:    0.7  (neutral - new agent, not yet proven)
#   Correct:  +0.1 (small reward, trust is slow to build)
#   Incorrect: -0.2 (bigger penalty, trust breaks fast)
#   Threshold: 0.5 (below this = Verifier steps in)
# =========================================================

TRUST_INITIAL         = 0.7   # every agent starts here
TRUST_CORRECT_DELTA   = +0.1  # reward for doing the right thing
TRUST_INCORRECT_DELTA = -0.2  # penalty for bad/adversarial output
TRUST_THRESHOLD       = 0.5   # the cutoff: below this triggers verification
TRUST_MIN             = 0.0   # score can't go below 0
TRUST_MAX             = 1.0   # score can't go above 1


@dataclass
class AgentTrustRecord:
    """
    I store one of these for every BLUE agent.
    It tracks their current trust score and the full history.
    """
    agent_id: str
    team: str
    score: float = TRUST_INITIAL
    rounds_active: int = 0
    correct_outputs: int = 0
    incorrect_outputs: int = 0
    verifier_intercepts: int = 0   # how many times I blocked this agent
    converged_at_round: Optional[int] = None   # which round they got flagged
    score_history: List[Tuple[int, float]] = field(default_factory=list)

    def update(self, correct: bool, timestep: int) -> float:
        """
        I call this after every agent action to update their trust score.
        If they were correct, score goes up a little.
        If they were wrong, score drops more (asymmetric by design).
        """
        self.rounds_active += 1
        if correct:
            self.correct_outputs += 1
            self.score = min(TRUST_MAX, self.score + TRUST_CORRECT_DELTA)
        else:
            self.incorrect_outputs += 1
            self.score = max(TRUST_MIN, self.score + TRUST_INCORRECT_DELTA)

        # save the score at this timestep for plotting later
        self.score_history.append((timestep, round(self.score, 3)))

        # record the first time this agent drops below the threshold
        if self.score < TRUST_THRESHOLD and self.converged_at_round is None:
            self.converged_at_round = timestep

        return self.score

    @property
    def is_low_trust(self) -> bool:
        # returns True if this agent is currently flagged as unreliable
        return self.score < TRUST_THRESHOLD


@dataclass
class VerifierDecision:
    """
    I log every decision my Verifier makes so I can analyze it later.
    This tells me what action was proposed, whether it was accepted,
    and why the decision was made.
    """
    timestep: int
    agent_id: str
    trust_score: float
    action_proposed: str
    accepted: bool
    reason: str


@dataclass
class TrustMetrics:
    """
    These are my 4 evaluation metrics from the PPT Section 4.
    I measure these to show that my framework actually works.
    """
    total_decisions: int = 0
    wrong_accepted: int = 0      # bad outputs that slipped through (Metric 1 numerator)
    correct_accepted: int = 0    # good outputs that passed correctly
    wrong_blocked: int = 0       # bad outputs my Verifier caught and blocked
    verifier_activations: int = 0  # how many times the Verifier had to step in
    task_attempts: int = 0       # total tasks tried (Metric 4 denominator)
    task_successes: int = 0      # tasks completed successfully (Metric 4 numerator)
    convergence_rounds: List[int] = field(default_factory=list)  # Metric 2 data
    history: List[Dict[str, Any]] = field(default_factory=list)  # per-step snapshots

    @property
    def error_propagation_rate(self) -> float:
        """
        Metric 1: what percentage of decisions were corrupted by bad outputs?
        Lower is better — my framework should push this close to zero.
        """
        if self.total_decisions == 0:
            return 0.0
        return round(self.wrong_accepted / self.total_decisions, 4)

    @property
    def mean_trust_convergence(self) -> Optional[float]:
        """
        Metric 2: on average, how many rounds before a bad agent is identified?
        The PPT target is ~4 rounds. Faster detection = better security.
        """
        if not self.convergence_rounds:
            return None
        return round(sum(self.convergence_rounds) / len(self.convergence_rounds), 2)

    @property
    def incorrect_decisions_accepted(self) -> int:
        """
        Metric 3: absolute count of bad outputs that affected the system.
        Without my defense this is high; with it, it should be near zero.
        """
        return self.wrong_accepted

    @property
    def task_success_rate(self) -> float:
        """
        Metric 4: what percentage of tasks still succeeded despite the defense?
        I want this to stay high (above 90%) to show minimal performance cost.
        """
        if self.task_attempts == 0:
            return 1.0
        return round(self.task_successes / self.task_attempts, 4)

    def snapshot(self, timestep: int, trust_records: Dict[str, AgentTrustRecord]):
        """Save a snapshot of all metrics at this timestep for plotting."""
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
        """Print a summary of all 4 metrics at the end of the simulation."""
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
    """
    This is my central trust store — like a database of trust scores.
    The Planner reads from it to know which agents to trust.
    The Verifier writes to it after evaluating each agent's output.
    """

    def __init__(self):
        self._records: Dict[str, AgentTrustRecord] = {}
        self.metrics = TrustMetrics()

    def register(self, agent_id: str, team: str):
        """Add a new agent to the trust system with a starting score of 0.7."""
        if agent_id not in self._records:
            self._records[agent_id] = AgentTrustRecord(agent_id=agent_id, team=team)

    def score(self, agent_id: str) -> float:
        """Look up an agent's current trust score."""
        return self._records[agent_id].score if agent_id in self._records else TRUST_INITIAL

    def is_low_trust(self, agent_id: str) -> bool:
        """Check if an agent is currently flagged (score below 0.5)."""
        return self.score(agent_id) < TRUST_THRESHOLD

    def all_scores(self) -> Dict[str, float]:
        """Get all agents' current scores as a dictionary."""
        return {aid: rec.score for aid, rec in self._records.items()}

    def low_trust_agents(self) -> List[str]:
        """Return a list of agent IDs that are currently flagged as low-trust."""
        return [aid for aid, rec in self._records.items() if rec.is_low_trust]

    def update_trust(self, agent_id: str, correct: bool, timestep: int) -> float:
        """
        Update an agent's trust score after an interaction.
        This is the core of my dynamic trust mechanism.
        """
        if agent_id not in self._records:
            return TRUST_INITIAL
        new_score = self._records[agent_id].update(correct, timestep)
        rec = self._records[agent_id]
        # if this is the first time they crossed below threshold, record it for Metric 2
        if rec.converged_at_round == timestep:
            self.metrics.convergence_rounds.append(timestep)
        return new_score

    def snapshot(self, timestep: int):
        """Take a snapshot of all current scores for plotting."""
        self.metrics.snapshot(timestep, self._records)

    def print_scores(self):
        """Print all agent trust scores to the terminal."""
        print("[TrustRegistry] Agent scores:")
        for aid, rec in sorted(self._records.items()):
            flag = " <<< LOW-TRUST" if rec.is_low_trust else ""
            print(f"  {aid} ({rec.team}): {rec.score:.2f}{flag}")


class VerifierAgent:
    """
    My Verifier Agent is the security layer I designed for this project.

    How it works:
    - If an agent's trust score is >= 0.5, I let their action through immediately
    - If an agent's trust score is < 0.5, I check their action against 4 rules:
        1. Flag carrier moving away from base? Block it.
        2. Defend-assigned agent trying to attack? Block it.
        3. Attack-assigned agent retreating to own base? Block it.
        4. Hold-assigned agent trying to spread? Block it.
    - If a blocked agent would be left with nothing to do, I give them a safe
      fallback: hold their current position.

    This design is inspired by the Verifier Agent concept in my PPT (Section 3)
    and the defense pipeline idea from Hossain et al. (2025).
    """

    def __init__(self, registry: TrustRegistry):
        self.registry  = registry
        self.decisions: List[VerifierDecision] = []  # log of every decision made

    def verify(self, agent_id: str, proposed_action: str,
               context: Dict[str, Any], timestep: int) -> Tuple[bool, str]:
        """
        The main verification function.
        Returns (accepted, reason) for each agent action.

        Fast path: trusted agents skip verification entirely.
        Slow path: low-trust agents get their action checked against my rules.
        """
        trust = self.registry.score(agent_id)
        self.registry.metrics.total_decisions += 1

        if trust >= TRUST_THRESHOLD:
            # agent is trusted — let the action through without checking
            self._record(timestep, agent_id, trust, proposed_action, True, "trusted")
            self.registry.metrics.correct_accepted += 1
            self.registry.metrics.task_successes += 1
            self.registry.metrics.task_attempts += 1
            self.registry.update_trust(agent_id, correct=True, timestep=timestep)
            return True, "trusted"

        # agent is low-trust — I need to evaluate their action
        self.registry.metrics.verifier_activations += 1
        self.registry.metrics.task_attempts += 1
        accepted, reason = self._evaluate(proposed_action, context)

        if accepted:
            # action passed my checks — allow it and give a small trust boost
            self.registry.metrics.correct_accepted += 1
            self.registry.metrics.task_successes += 1
            self.registry.update_trust(agent_id, correct=True, timestep=timestep)
        else:
            # action failed my checks — block it and penalize trust further
            self.registry.metrics.wrong_blocked += 1
            self.registry.update_trust(agent_id, correct=False, timestep=timestep)

        self._record(timestep, agent_id, trust, proposed_action, accepted, reason)
        return accepted, reason

    def _evaluate(self, proposed_action: str, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        My 4 alignment rules for catching adversarial behavior.
        I designed these based on what an adversarial agent would do
        in a capture-the-flag / grid-world scenario.
        """
        strategic_goal = context.get("strategic_goal", "")
        has_flag       = context.get("has_flag", False)
        own_base       = context.get("own_base")
        target_pos     = context.get("target_position")
        current_pos    = context.get("current_position")
        action_lower   = proposed_action.lower()

        # Rule 1: if the agent has the flag and is moving away from base, block it
        # (a compromised agent might try to carry the flag in the wrong direction)
        if has_flag and own_base and target_pos and current_pos:
            dist_now = _trust_dist(current_pos, own_base)
            dist_tgt = _trust_dist(target_pos, own_base)
            if dist_tgt > dist_now + 2:
                return False, "flag-carrier moving away from base"

        # Rule 2: if agent is assigned to defend but is trying to attack, block it
        # (goal mismatch — adversarial agents might try to abandon defense)
        if strategic_goal == "defend" and "attack" in action_lower:
            return False, "defend-agent issuing attack (goal mismatch)"

        # Rule 3: if agent is assigned to attack but is retreating to own base, block it
        # (adversarial retreat could expose the team)
        if strategic_goal == "attack" and own_base and target_pos:
            if own_base == target_pos:
                return False, "attack-agent retreating to own base"

        # Rule 4: if agent is assigned to hold a zone but is trying to spread, block it
        # (spreading abandons the zone this agent was supposed to protect)
        if strategic_goal == "hold" and "spread" in action_lower:
            return False, "hold-agent spreading (abandons zone)"

        # passed all checks — action looks legitimate
        return True, "verified-ok"

    def verify_blue_targets(self, blue_targets: Dict[str, Any],
                            env_context: Dict[str, Any],
                            timestep: int) -> Tuple[Dict[str, Any], List[str]]:
        """
        I run every BLUE agent's movement target through the Verifier.
        Blocked agents are given a safe fallback (stay put) instead.
        Returns: (filtered targets, list of blocked agent IDs)
        """
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
                # safe fallback: agent stays at their current position
                fallback = env_context.get("current_positions", {}).get(rid, target)
                filtered[rid] = fallback
        return filtered, blocked

    def report_outcome(self, agent_id: str, was_correct: bool, timestep: int):
        """
        I call this after the step resolves to update trust based on what actually happened.
        For example: if an agent got eliminated in combat, I treat that as a bad outcome.
        """
        if was_correct:
            self.registry.metrics.task_successes += 1
        else:
            self.registry.metrics.wrong_accepted += 1
        self.registry.update_trust(agent_id, correct=was_correct, timestep=timestep)

    def _record(self, timestep, agent_id, trust, action, accepted, reason):
        """Internal helper: save this decision to the log."""
        self.decisions.append(VerifierDecision(
            timestep=timestep, agent_id=agent_id,
            trust_score=round(trust, 3), action_proposed=action,
            accepted=accepted, reason=reason,
        ))


def _trust_dist(a, b) -> float:
    """Euclidean distance helper used in Verifier rules."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def init_trust_for_grid(env: EnvironmentState) -> Tuple[TrustRegistry, VerifierAgent]:
    """
    I call this once at the start of the simulation to set up
    the trust registry and verifier for all BLUE agents.
    RED agents are autonomous so I don't track their trust.
    """
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
    """
    I use this after block6 assigns targets to run every target
    through the Verifier before passing it to path planning.
    This is where my trust defense actually intercepts bad behavior.
    """
    current_positions = {rid: r.position for rid, r in env.blue_robots.items() if r.alive}
    env_context = {
        "own_base": BLUE_BASE,
        "enemy_base": RED_BASE,
        "current_positions": current_positions,
        "team": "BLUE",
    }
    filtered, _ = verifier.verify_blue_targets(blue_targets, env_context, timestep)
    registry.snapshot(timestep)  # save current state for later analysis
    return filtered


def update_trust_after_combat(registry: TrustRegistry, verifier: VerifierAgent,
                               env: EnvironmentState, timestep: int):
    """
    After each combat step, I update trust scores based on outcomes.
    An agent that got eliminated → bad outcome → trust penalty.
    An agent that scored a kill → good outcome → trust boost.
    """
    for rid, robot in env.blue_robots.items():
        if rid not in registry.all_scores():
            continue
        if not robot.alive:
            # agent was eliminated — this counts as a failed action
            verifier.report_outcome(rid, was_correct=False, timestep=timestep)
        elif robot.kills > 0:
            # agent scored a kill — this counts as a successful action
            verifier.report_outcome(rid, was_correct=True, timestep=timestep)


# =========================================================
# ADVERSARIAL AGENT SIMULATION
#
# This is how I recreate the PPT Section 4 experiment.
# I configure B2 to randomly inject bad outputs at 60% probability.
# This simulates prompt injection or model misalignment.
#
# Without my trust framework, these bad outputs would all get through.
# With my framework, B2 should be detected and flagged by round 4.
# =========================================================

ADVERSARIAL_AGENT = "B2"   # I chose B2 as the bad actor for my experiment
ADVERSARIAL_PROB  = 0.6    # 60% error rate — matches my PPT evaluation setup


def simulate_adversarial_outputs(registry: TrustRegistry,
                                  verifier: VerifierAgent,
                                  env: EnvironmentState,
                                  timestep: int,
                                  adversarial_agent: str = ADVERSARIAL_AGENT,
                                  error_prob: float = ADVERSARIAL_PROB):
    """
    I run this at the start of every timestep to simulate what would happen
    if one agent was adversarial. B2 produces wrong outputs 60% of the time.
    B1 and B3 always produce correct outputs (they are trusted agents).

    This is the core of my PPT Section 4 experiment — I need to show that
    my framework catches B2 automatically without a human flagging it.
    """
    print(f"\n  [Adversarial Sim] Checking {adversarial_agent} (error_prob={error_prob:.0%}):")

    for rid in list(env.blue_robots.keys()):
        robot = env.blue_robots[rid]
        if not robot.alive:
            continue

        if rid == adversarial_agent:
            # B2 is my adversarial agent — it randomly injects bad outputs
            is_bad = random.random() < error_prob
            if is_bad:
                registry.update_trust(rid, correct=False, timestep=timestep)
                registry.metrics.wrong_accepted += 1
                print(f"    {rid} [ADVERSARIAL]: injected BAD output  "
                      f"-> score now {registry.score(rid):.2f}"
                      + (" <<< FLAGGED" if registry.is_low_trust(rid) else ""))
            else:
                # even adversarial agents produce correct outputs sometimes
                registry.update_trust(rid, correct=True, timestep=timestep)
                print(f"    {rid} [ADVERSARIAL]: produced good output  "
                      f"-> score now {registry.score(rid):.2f}")
        else:
            # B1 and B3 are honest agents — always correct
            registry.update_trust(rid, correct=True, timestep=timestep)
            print(f"    {rid} [trusted]:     correct output          "
                  f"-> score now {registry.score(rid):.2f}")


def print_trust_status(registry: TrustRegistry, timestep: int):
    """Print a one-line trust status summary for this timestep."""
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
    """Print the full trust metrics report at the end of the game."""
    print(registry.metrics.report())
    print("\n[TrustFramework] Per-agent final summary:")
    for aid, rec in sorted(registry._records.items()):
        conv = f"flagged @ t={rec.converged_at_round}" if rec.converged_at_round else "never flagged"
        print(f"  {aid}: score={rec.score:.2f} | "
              f"correct={rec.correct_outputs} wrong={rec.incorrect_outputs} | {conv}")


def export_trust_data(registry: TrustRegistry, filename: str = "trust_data.json"):
    """
    Save all trust metrics to a JSON file for analysis and plotting.
    I use this data to generate the charts in my paper.
    """
    data = {
        "metrics": {
            "error_propagation_rate":        registry.metrics.error_propagation_rate,
            "mean_trust_convergence_rounds":  registry.metrics.mean_trust_convergence,
            "incorrect_decisions_accepted":   registry.metrics.incorrect_decisions_accepted,
            "task_success_rate":              registry.metrics.task_success_rate,
            "verifier_activations":           registry.metrics.verifier_activations,
            "total_decisions":                registry.metrics.total_decisions,
        },
        "agent_records": {
            aid: {
                "team":              rec.team,
                "final_score":       rec.score,
                "correct_outputs":   rec.correct_outputs,
                "incorrect_outputs": rec.incorrect_outputs,
                "converged_at_round": rec.converged_at_round,
                "score_history":     rec.score_history,
            }
            for aid, rec in registry._records.items()
        },
        "history": registry.metrics.history,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[TrustFramework] Trust data saved to {filename}")


# =========================================================
# GRID HELPER FUNCTIONS
# Basic utilities for working with the 10x10 map
# =========================================================

def in_bounds(cell: Cell) -> bool:
    """Check if a cell is within the grid boundaries."""
    x, y = cell
    return 0 <= x < GRID_H and 0 <= y < GRID_W


def is_free(cell: Cell) -> bool:
    """Check if a cell is within bounds and not a wall."""
    x, y = cell
    return in_bounds(cell) and GRID[x][y] == 0


def manhattan(a: Cell, b: Cell) -> int:
    """Manhattan distance — I use this for nearby detection and pathfinding."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Cell, b: Cell) -> float:
    """Euclidean distance — I use this for distance tracking."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def get_neighbors(cell: Cell) -> List[Cell]:
    """Return the 4 adjacent cells that are free (no walls, in bounds)."""
    x, y = cell
    candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
    return [c for c in candidates if is_free(c)]


def nearest_cell(start: Cell, cells: List[Cell]) -> Cell:
    """Find the closest cell in a list using Manhattan distance."""
    return min(cells, key=lambda c: manhattan(start, c))


def visible_obstacles(position: Cell, radius: int = 2) -> List[Cell]:
    """Return all wall cells within radius steps of the given position."""
    px, py = position
    out = []
    for x in range(max(0, px - radius), min(GRID_H, px + radius + 1)):
        for y in range(max(0, py - radius), min(GRID_W, py + radius + 1)):
            if GRID[x][y] == 1:
                out.append((x, y))
    return out


def visible_regions(position: Cell, radius: int = 2) -> List[str]:
    """Return the names of strategic regions visible from this position."""
    out = []
    for region_name, cells in REGIONS.items():
        for c in cells:
            if manhattan(position, c) <= radius:
                out.append(region_name)
                break
    return out


# =========================================================
# BLOCK 1 — ENVIRONMENT INITIALIZATION
# Set up the 3 vs 3 battle: BLUE starts top-left, RED bottom-right
# =========================================================

def block1_environment_initialization() -> EnvironmentState:
    """
    I initialize the simulation with 3 BLUE robots and 3 RED robots.
    BLUE agents are the ones I control (and test my trust framework on).
    RED agents are autonomous opponents using a simple heuristic.
    """
    blue_robots = {
        "B1": Robot("B1", "BLUE", (0, 0)),
        "B2": Robot("B2", "BLUE", (0, 1)),  # B2 will be my adversarial agent
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
# Each robot can only see what's within 3 cells of them
# =========================================================

def block2_local_observation(env: EnvironmentState) -> Dict[RobotId, LocalObservation]:
    """
    I build a local view for each robot — they can only sense teammates,
    enemies, and obstacles within a radius of 3 cells.
    This simulates realistic sensor limitations in multi-agent systems.
    """
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
# BLOCK 3 — GLOBAL STATE ENCODING (CTDE)
# Central view of the battlefield for strategic decisions
# =========================================================

def block3_global_state_encoding(env: EnvironmentState,
                                  local_obs: Dict[RobotId, LocalObservation]) -> GlobalState:
    """
    I build the full global state from local observations.
    This is the Centralized Training / Decentralized Execution (CTDE) pattern.
    The central controller (Planner) can see everything; individual robots only see locally.
    I also compute which regions are contested and how dangerous each RED robot is.
    """
    blue_positions = {rid: r.position for rid, r in env.blue_robots.items() if r.alive}
    red_positions  = {rid: r.position for rid, r in env.red_robots.items()  if r.alive}

    # a region is contested if both a BLUE and RED robot are in it
    contested = []
    for region_name, cells in REGIONS.items():
        blue_in = any(pos in cells for pos in blue_positions.values())
        red_in  = any(pos in cells for pos in red_positions.values())
        if blue_in and red_in:
            contested.append(region_name)

    # threat = kills / (deaths + 1) — higher means more dangerous RED robot
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
# BLOCK 4 — SYSTEM SUMMARY + STRATEGY GENERATION
# Shows me the current battlefield situation and suggests strategies
# =========================================================

def block4_system_summary(global_state: GlobalState) -> str:
    """Build a quick one-line summary of the current game state."""
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
    """
    I estimate my team's chance of winning based on:
    - Force ratio (how many BLUE vs RED are alive)
    - Whether I'm contesting key regions
    - How dangerous the remaining RED robots are
    Returns a value between 0.05 and 0.95.
    """
    blue  = global_state.alive_blue
    red   = global_state.alive_red
    total = blue + red
    if total == 0:
        return 0.5
    base = blue / total
    adj  = 0.03 * len(global_state.contested_regions)  # bonus for contesting zones
    if global_state.threat_map:
        avg_threat = sum(global_state.threat_map.values()) / len(global_state.threat_map)
        if avg_threat > 1.5:
            adj -= 0.05 * (avg_threat - 1.0)  # penalty for high RED threat
    return max(0.05, min(0.95, base + adj))


def determine_situation_tag(global_state: GlobalState, max_steps: int = 20) -> str:
    """
    I tag the current game situation so I can categorize it in analysis.
    Examples: 'early_game_balanced', 'outnumbered_contested', 'end_game_critical_blue'
    """
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
    """
    Generate a situation report I can print when the user presses '0'.
    This replaces the OpenAI API call — everything is done locally with rules.
    """
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

    # pick the most relevant situation description
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
    """
    Show 3 strategy suggestions when the user presses '9'.
    These are rule-based — no external API needed.
    """
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
# This is where I type my commands during the simulation
# =========================================================

def parse_blue_command(text: str) -> Optional[ParsedCommand]:
    """
    Parse a natural language command string into a structured ParsedCommand.
    I support: move, hold, defend, attack, regroup, spread
    Commands can apply to all robots ('all') or a specific one ('robot B1')
    """
    s = text.strip().lower()

    # check if command targets all robots or just one
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

    # match the action type
    m = re.match(r"^move\s+to\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "move", scope, target,
                             {"cell": (int(m.group(1)), int(m.group(2)))})

    m = re.match(r"^hold\s+region\s+([a-zA-Z0-9_]+)$", rest)
    if m:
        return ParsedCommand(text, "hold", scope, target,
                             {"region": m.group(1).upper()})

    m = re.match(r"^defend\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "defend", scope, target,
                             {"cell": (int(m.group(1)), int(m.group(2)))})

    m = re.match(r"^attack\s+([a-zA-Z0-9_]+)$", rest)
    if m:
        t = m.group(1).upper()
        if t == "RED_BASE":
            return ParsedCommand(text, "attack", scope, target, {"cell": RED_BASE})
        else:
            # attack a specific enemy robot by ID
            return ParsedCommand(text, "attack", scope, target, {"enemy_target": t})

    m = re.match(r"^regroup\s+at\s+(\d+)\s+(\d+)$", rest)
    if m:
        return ParsedCommand(text, "regroup", scope, target,
                             {"cell": (int(m.group(1)), int(m.group(2)))})

    if re.match(r"^spread$", rest):
        return ParsedCommand(text, "spread", scope, target, {})

    return None  # command didn't match any pattern


def block5_human_command_interface(env: EnvironmentState, global_state: GlobalState,
                                    first_turn: bool = False) -> HumanPlan:
    """
    My command terminal — I type instructions here during the simulation.
    Press 0 to see the current situation, 9 for strategy suggestions,
    or just press Enter to keep the current plan.
    """
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
# Converts my commands into specific target cells for each robot
# =========================================================

def command_applies(cmd: ParsedCommand, robot_id: str) -> bool:
    """Check if a command applies to a specific robot (either 'all' or matching ID)."""
    return cmd.scope == "all" or cmd.target == robot_id


def block6_blue_strategy_assignment(env: EnvironmentState,
                                     human_plan: HumanPlan) -> Dict[RobotId, Cell]:
    """
    I translate my high-level commands into movement targets for each BLUE robot.
    Default: each robot stays at its current position if no command was given.
    This is where the Planner dispatches subtasks to Executor agents.
    """
    # start with each robot targeting its own position (hold by default)
    blue_targets: Dict[RobotId, Cell] = {}
    for rid, robot in env.blue_robots.items():
        if robot.alive:
            blue_targets[rid] = robot.position

    # now apply each command to the relevant robots
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
                    # targeting a specific enemy robot
                    eid = cmd.params["enemy_target"]
                    if eid in env.red_robots and env.red_robots[eid].alive:
                        blue_targets[rid] = env.red_robots[eid].position
                    else:
                        blue_targets[rid] = RED_BASE  # fallback if target is dead
                else:
                    # targeting the red base directly
                    blue_targets[rid] = cmd.params["cell"]
            elif cmd.action == "spread":
                # distribute robots to different pre-defined spread positions
                spread_cells = [(1, 8), (8, 1), (1, 5), (5, 1), (8, 5), (5, 8)]
                idx = list(env.blue_robots.keys()).index(rid) % len(spread_cells)
                blue_targets[rid] = spread_cells[idx]

    return blue_targets


# =========================================================
# BLOCK 7 — RED AUTONOMOUS STRATEGY
# The opponent AI — I didn't design this to be smart,
# just enough to provide a realistic adversarial challenge
# =========================================================

def block7_red_autonomous_strategy(env: EnvironmentState) -> Dict[RobotId, Cell]:
    """
    Simple heuristic for RED robots:
    - If a BLUE robot is within 4 cells, chase it
    - Otherwise, move toward the center of the map
    - If all BLUE robots are gone, head to BLUE base
    """
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
            # close enough to chase — go after the nearest BLUE robot
            red_targets[rid] = nearest_blue.position
        else:
            # too far — move toward center to establish map control
            red_targets[rid] = nearest_cell(red_robot.position, center_cells)

    return red_targets


# =========================================================
# BLOCK 8 — A* PATHFINDING
# I use A* so robots find optimal paths around walls
# =========================================================

def astar(start: Cell, goal: Cell) -> List[Cell]:
    """
    Standard A* search algorithm with Manhattan distance as the heuristic.
    Returns the full path from start to goal, or empty list if no path exists.
    I use this so robots automatically navigate around walls.
    """
    if not is_free(start) or not is_free(goal):
        return []
    open_heap: List[Tuple[int, Cell]] = []
    heapq.heappush(open_heap, (0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, int]    = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            # reconstruct the path by following came_from back to start
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
                f = tentative + manhattan(nb, goal)  # f = g + h
                heapq.heappush(open_heap, (f, nb))
    return []  # no path found


def block8_path_planning(env: EnvironmentState,
                          blue_targets: Dict[RobotId, Cell],
                          red_targets:  Dict[RobotId, Cell]) -> Dict[RobotId, List[Cell]]:
    """Run A* for every robot (both BLUE and RED) toward their assigned targets."""
    all_paths: Dict[RobotId, List[Cell]] = {}
    for rid, robot in env.blue_robots.items():
        if robot.alive:
            all_paths[rid] = astar(robot.position, blue_targets.get(rid, robot.position))
    for rid, robot in env.red_robots.items():
        if robot.alive:
            all_paths[rid] = astar(robot.position, red_targets.get(rid, robot.position))
    return all_paths


# =========================================================
# BLOCK 9 — MOVEMENT + COMBAT EXECUTION
# Move all robots one step along their paths, then resolve fights
# =========================================================

def system_positioning_adjust(current: Dict[RobotId, Cell],
                               proposed: Dict[RobotId, Cell],
                               cfg: PositioningConfig) -> Dict[RobotId, Cell]:
    """
    Prevent two robots from occupying the same cell.
    If a collision would happen, the second robot holds its current position.
    """
    final = dict(proposed)
    ids   = list(final.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if final[a] == final[b]:
                # direct collision — second robot stays
                final[b] = current[b]
            elif manhattan(final[a], final[b]) < cfg.min_separation:
                # too close — second robot stays
                final[b] = current[b]
    return final


def next_step_from_path(path: List[Cell], current: Cell) -> Cell:
    """Take one step along the path. If path is empty or too short, stay put."""
    return path[1] if len(path) >= 2 else current


def resolve_combat(env: EnvironmentState, metrics: Metrics):
    """
    If a BLUE and RED robot are adjacent (Manhattan distance <= 1), they fight.
    BLUE has a 55% win chance per encounter — slight advantage to my team.
    This is a simple probabilistic combat model.
    """
    blue_alive = [r for r in env.blue_robots.values() if r.alive]
    red_alive  = [r for r in env.red_robots.values()  if r.alive]

    for b in blue_alive:
        for r in red_alive:
            if not b.alive or not r.alive:
                continue  # skip already-eliminated robots
            if manhattan(b.position, r.position) <= 1:
                if random.random() < 0.55:
                    # BLUE wins this fight
                    r.alive = False; r.deaths += 1; b.kills += 1
                    metrics.kills[b.robot_id]  += 1
                    metrics.deaths[r.robot_id] += 1
                    print(f"  {b.robot_id} (BLUE) eliminated {r.robot_id} (RED)")
                else:
                    # RED wins this fight
                    b.alive = False; b.deaths += 1; r.kills += 1
                    metrics.kills[r.robot_id]  += 1
                    metrics.deaths[b.robot_id] += 1
                    print(f"  {r.robot_id} (RED) eliminated {b.robot_id} (BLUE)")


def block9_positioning_and_execution(env: EnvironmentState,
                                      all_paths: Dict[RobotId, List[Cell]],
                                      blue_targets: Dict[RobotId, Cell],
                                      metrics: Metrics,
                                      human_plan: HumanPlan) -> EnvironmentState:
    """
    Step 1: Compute proposed next positions for all robots.
    Step 2: Resolve any positioning conflicts.
    Step 3: Move all robots.
    Step 4: Track command compliance (is each BLUE robot moving toward its target?).
    Step 5: Resolve combat for any adjacent robot pairs.
    Step 6: Give objective capture credit for robots in the CENTER region.
    """
    current_pos  = {}
    proposed_pos = {}

    for rid, robot in {**env.blue_robots, **env.red_robots}.items():
        if robot.alive:
            current_pos[rid]  = robot.position
            proposed_pos[rid] = next_step_from_path(all_paths.get(rid, []), robot.position)

    # fix any collisions before moving
    adjusted = system_positioning_adjust(current_pos, proposed_pos, PositioningConfig())

    # actually move all BLUE robots
    for rid, robot in env.blue_robots.items():
        if robot.alive:
            old = robot.position
            robot.position      = adjusted.get(rid, robot.position)
            robot.last_position = old
            robot.distance_covered += euclidean(old, robot.position)

    # actually move all RED robots
    for rid, robot in env.red_robots.items():
        if robot.alive:
            old = robot.position
            robot.position      = adjusted.get(rid, robot.position)
            robot.last_position = old
            robot.distance_covered += euclidean(old, robot.position)

    # track whether each BLUE robot is obeying my commands
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
                # new target — reset the previous distance tracker
                metrics.command_last_tgt[rid]  = tkey
                metrics.command_prev_dist[rid] = None
            prev = metrics.command_prev_dist.get(rid)
            if prev is None:
                metrics.command_prev_dist[rid] = d
                continue
            if robot.position == tgt or d <= _eps:
                metrics.command_compliance[rid] += 1   # reached target
            elif d > prev + _eps:
                metrics.command_violations[rid]  += 1  # moving away from target
            elif d < prev - _eps:
                metrics.command_compliance[rid]  += 1  # getting closer to target
            metrics.command_prev_dist[rid] = d

    # run combat for all adjacent robot pairs
    resolve_combat(env, metrics)

    # give objective capture credit for BLUE robots in the CENTER region
    for rid, robot in env.blue_robots.items():
        if robot.alive and robot.position in REGIONS["CENTER"]:
            metrics.objective_captures[rid] += 1

    return env


# =========================================================
# BLOCK 10 — STRATEGIC CHANGE DETECTION
# I use this to trigger a replan when the situation changes significantly
# =========================================================

def block10_strategic_change_detection(env: EnvironmentState,
                                        global_state: GlobalState) -> bool:
    """
    I trigger a replan if:
    - My team is down to 1 robot (critical situation)
    - The enemy is down to 1 robot (finishing opportunity)
    - A region becomes contested (need to respond tactically)
    """
    if global_state.alive_blue <= 1:
        return True   # I'm nearly wiped out — need to adjust
    if global_state.alive_red  <= 1:
        return True   # enemy nearly defeated — time to finish
    if global_state.contested_regions:
        return True   # zones are being fought over — replan needed
    return False


# =========================================================
# BLOCK 11 — METRICS UPDATE
# I calculate win rate, synergy, and per-robot stats every step
# =========================================================

def block11_metrics_update(env: EnvironmentState, metrics: Metrics,
                            global_state: Optional[GlobalState] = None) -> Dict[str, Any]:
    """
    Update and return all game metrics for this timestep.
    Team synergy measures how well my robots are positioned together.
    Win rate estimates my probability of winning based on current state.
    """
    player_rows = []
    for robot in list(env.blue_robots.values()) + list(env.red_robots.values()):
        kills  = metrics.kills[robot.robot_id]
        deaths = max(1, metrics.deaths[robot.robot_id])  # avoid division by zero
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
        """
        Measure how well grouped a team is.
        High synergy = robots close together (coordinated).
        Low synergy = robots spread out far (disorganized).
        Formula: 1 / (1 + average pairwise distance)
        """
        alive = [r for r in team_robots.values() if r.alive]
        if len(alive) < 2:
            return 0.0
        dists = [manhattan(alive[i].position, alive[j].position)
                 for i in range(len(alive)) for j in range(i + 1, len(alive))]
        return round(1.0 / (1.0 + sum(dists) / len(dists)), 3)

    win_rate  = 0.5
    situation = "unknown"
    if global_state:
        win_rate  = calculate_winning_rate(global_state)
        situation = determine_situation_tag(global_state, env.max_steps)
        # log for plotting
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
# The game ends when one team is eliminated or we hit 20 steps
# =========================================================

def block12_termination_check(env: EnvironmentState) -> bool:
    """
    End the simulation if:
    - All BLUE robots are eliminated (RED wins)
    - All RED robots are eliminated (BLUE wins)
    - We reach the maximum number of timesteps (draw)
    """
    blue_alive = any(r.alive for r in env.blue_robots.values())
    red_alive  = any(r.alive for r in env.red_robots.values())
    if not blue_alive or not red_alive:
        env.done = True   # someone was wiped out
    elif env.timestep >= env.max_steps:
        env.done = True   # time limit reached
    else:
        env.done = False
    return env.done


# =========================================================
# DISPLAY HELPERS
# Print the grid and game status to the terminal each step
# =========================================================

def print_grid(env: EnvironmentState):
    """
    Print the current battlefield.
    BLUE robots shown as their number (1, 2, 3).
    RED robots shown as lowercase number (1, 2, 3).
    Walls shown as #, empty cells as .
    """
    board = [["." if GRID[x][y] == 0 else "#" for y in range(GRID_W)]
             for x in range(GRID_H)]
    for rid, r in env.blue_robots.items():
        if r.alive:
            x, y = r.position
            board[x][y] = rid[-1]           # e.g. "1" for B1
    for rid, r in env.red_robots.items():
        if r.alive:
            x, y = r.position
            board[x][y] = rid[-1].lower()   # e.g. "1" for R1 (lowercase = red)
    print("\nGrid (BLUE=1/2/3  RED=1/2/3-lower  #=wall):")
    for row in board:
        print(" ".join(row))


def print_status(env: EnvironmentState, summary: str, metrics_out: Dict[str, Any]):
    """Print the full step summary including robot positions and metrics."""
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
    """Save per-timestep plotting data to JSON for generating graphs in my paper."""
    data = {
        "winning_rate_trajectory": metrics.winning_rate_history,
        "situation_timeline":      metrics.situation_history,
        "detailed_steps":          [p.to_dict() for p in plotting_history],
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[PlotData] Saved to {filename}  ({len(plotting_history)} timesteps)")


# =========================================================
# MAIN — GRID WORLD WITH FULL TRUST FRAMEWORK
#
# This is the entry point for my simulation.
# Steps every round:
#   1. Observe the environment
#   2. Get my strategy commands (or replan if situation changed)
#   3. Run adversarial simulation (B2 injects bad outputs)
#   4. Verifier checks all BLUE targets
#   5. Plan paths with A*
#   6. Execute movement and combat
#   7. Update trust scores based on outcomes
#   8. Record metrics and print status
#   9. Repeat until game ends
# =========================================================

def main():
    random.seed(42)   # I fix the seed so results are reproducible
    plotting_history: List[PlottingData] = []

    print("\n" + "="*50)
    print("  MULTI-AGENT SIMULATION + TRUST FRAMEWORK")
    print("  Arjun Vooturi — CIS 6372 Information Assurance")
    print("="*50)

    # Block 1: set up 3v3 environment
    env     = block1_environment_initialization()
    metrics = Metrics()

    # Initialize my trust framework — register all BLUE agents at score 0.7
    trust_registry, verifier = init_trust_for_grid(env)

    # get the initial game state before any commands are entered
    local_obs0    = block2_local_observation(env)
    global_state0 = block3_global_state_encoding(env, local_obs0)

    # ask me for my first set of commands before the simulation starts
    human_plan = block5_human_command_interface(env, global_state0, first_turn=True)

    # ---- MAIN SIMULATION LOOP ----
    while True:
        env.timestep += 1

        # observe the current state
        local_obs    = block2_local_observation(env)
        global_state = block3_global_state_encoding(env, local_obs)
        summary      = block4_system_summary(global_state)

        # check if the situation changed enough to need a new plan
        if env.timestep > 1:
            if block10_strategic_change_detection(env, global_state):
                print("\n[Block 10] Strategic change detected — replan?")
                new_plan = block5_human_command_interface(env, global_state, first_turn=False)
                if new_plan.parsed_commands:
                    human_plan = new_plan

        # assign movement targets based on my current commands
        blue_targets = block6_blue_strategy_assignment(env, human_plan)

        # Step 1 of trust framework: run adversarial simulation
        # B2 randomly injects bad outputs at 60% probability
        simulate_adversarial_outputs(trust_registry, verifier, env, env.timestep)

        # Step 2 of trust framework: Verifier inspects every BLUE target
        # low-trust agents get their actions checked; trusted agents are fast-passed
        print(f"\n  [Verifier] Checking all blue targets at t={env.timestep}:")
        for rid, tgt in blue_targets.items():
            score  = trust_registry.score(rid)
            status = "LOW-TRUST -> will verify" if trust_registry.is_low_trust(rid) else "trusted  -> fast-pass"
            print(f"    {rid}: target={tgt}  score={score:.2f}  [{status}]")

        blue_targets = trust_filter_blue_targets(
            trust_registry, verifier, blue_targets, env, env.timestep
        )

        # RED robots pick their targets autonomously
        red_targets = block7_red_autonomous_strategy(env)

        # compute optimal paths for everyone using A*
        all_paths = block8_path_planning(env, blue_targets, red_targets)

        # move robots and resolve any combat
        env = block9_positioning_and_execution(
            env, all_paths, blue_targets, metrics, human_plan
        )

        # Step 3 of trust framework: update trust based on combat results
        update_trust_after_combat(trust_registry, verifier, env, env.timestep)
        print_trust_status(trust_registry, env.timestep)

        # calculate and log all metrics for this step
        metrics_out = block11_metrics_update(env, metrics, global_state)

        # collect plotting data
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

        # display current step info
        print_status(env, summary, metrics_out)

        # check if the game is over
        if block12_termination_check(env):
            break

    # ---- GAME OVER — FINAL OUTPUTS ----

    # save plotting data for generating paper charts
    export_plotting_data(metrics, plotting_history, "grid_world_plot_data.json")

    # print and save my trust framework results (the 4 metrics from my PPT)
    print_trust_report(trust_registry)
    export_trust_data(trust_registry, "grid_world_trust_data.json")

    # print final per-robot stats
    print("\n===== FINAL ROBOT METRICS =====")
    for row in metrics_out["player_metrics"]:
        print(row)

    # announce the winner
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
