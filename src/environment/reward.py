"""
src/environment/reward.py
==========================
Reward and constraint cost signals for DisasterResponseBenchmark.

Implements the shared team reward function (Eq. 4 of the paper):

    r(s_t, a_t) = α * R_mitigation(s_t, a_t)
                - β * R_false_alarm(s_t, a_t)
                - η * R_delay(s_t, a_t)

And the per-agent constraint cost C_i(s_t, a_t) used by the Lagrangian
relaxation (Eq. 5–6 of the paper).

Action encoding (from Section 3.3 of the paper):
    0 — no action
    1 — issue warning
    2 — deploy resources
    3 — initiate evacuation

Constraint cost C_i is incurred when an agent takes a high-severity action
(2 or 3) under conditions that do not justify it (low risk threshold), i.e.
unnecessary resource deployment or unwarranted evacuation orders.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.environment.hazard_generator import HazardState


# ---------------------------------------------------------------------------
# Action constants — mirrors action space definition in paper Section 3.3
# ---------------------------------------------------------------------------

ACTION_NO_OP = 0
ACTION_WARN = 1
ACTION_DEPLOY = 2
ACTION_EVACUATE = 3
HIGH_SEVERITY_ACTIONS = {ACTION_DEPLOY, ACTION_EVACUATE}

# Risk threshold below which a high-severity action is considered unnecessary
UNNECESSARY_ACTION_RISK_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RewardInfo:
    """Detailed breakdown of every reward component for logging."""
    total_reward: float
    mitigation: float
    false_alarm: float
    delay: float
    constraint_costs: Dict[int, float]   # {agent_id: cost}
    risk_level: float
    actions: Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Computes the shared team reward and per-agent constraint costs.

    Parameters
    ----------
    alpha : float
        Weight on mitigation reward (α in Eq. 4). Default 1.0.
    beta : float
        Weight on false-alarm penalty (β in Eq. 4). Default 0.5.
    eta : float
        Weight on delay penalty (η in Eq. 4). Default 0.3.
    constraint_threshold : float
        d_i — maximum tolerable cumulative constraint cost per agent.
        Used externally by LagrangianCTDE; stored here for reference.
    """

    N_AGENTS: int = 3

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        eta: float = 0.3,
        constraint_threshold: float = 0.10,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.constraint_threshold = constraint_threshold

        # Track the last step at which a hazard exceeded critical level
        # to compute the delay penalty
        self._hazard_onset_step: int | None = None
        self._last_response_step: int | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset delay-tracking state at the start of each episode."""
        self._hazard_onset_step = None
        self._last_response_step = None

    def compute(
        self,
        hazard_state: HazardState,
        actions: Tuple[int, int, int],
        prev_hazard_state: HazardState | None,
        timestep: int,
    ) -> RewardInfo:
        """
        Compute the full reward signal for a single transition.

        Parameters
        ----------
        hazard_state : HazardState
            Current state s_t after the joint action was executed.
        actions : tuple[int, int, int]
            Joint action (a1, a2, a3) from the three agents.
        prev_hazard_state : HazardState | None
            State s_{t-1} before the action (None at t=0).
        timestep : int
            Current simulation step index.

        Returns
        -------
        RewardInfo
            Total scalar reward plus detailed component breakdown.
        """
        risk = self._compute_risk_level(hazard_state)
        prev_risk = (
            self._compute_risk_level(prev_hazard_state)
            if prev_hazard_state is not None
            else risk
        )

        # Track onset: first step where risk crosses a critical threshold
        if risk > 0.6 and self._hazard_onset_step is None:
            self._hazard_onset_step = timestep

        # --- Mitigation reward R_mitigation ---
        mitigation = self._mitigation_reward(
            risk, prev_risk, actions
        )

        # --- False-alarm penalty R_false_alarm ---
        false_alarm = self._false_alarm_penalty(risk, actions)

        # --- Delay penalty R_delay ---
        delay = self._delay_penalty(risk, actions, timestep)

        # --- Total reward (Eq. 4) ---
        total = (
            self.alpha * mitigation
            - self.beta * false_alarm
            - self.eta * delay
        )

        # --- Per-agent constraint costs C_i ---
        constraint_costs = self._constraint_costs(risk, actions)

        return RewardInfo(
            total_reward=float(total),
            mitigation=float(mitigation),
            false_alarm=float(false_alarm),
            delay=float(delay),
            constraint_costs=constraint_costs,
            risk_level=float(risk),
            actions=actions,
        )

    def compute_violation(
        self,
        constraint_costs: Dict[int, float],
        d_i: float,
    ) -> bool:
        """
        Return True if any agent's constraint cost exceeds d_i.
        Implements the indicator in Eq. 9 (VR metric).
        """
        return any(c > d_i for c in constraint_costs.values())

    # -----------------------------------------------------------------------
    # Reward component implementations
    # -----------------------------------------------------------------------

    def _mitigation_reward(
        self,
        risk: float,
        prev_risk: float,
        actions: Tuple[int, int, int],
    ) -> float:
        """
        R_mitigation is proportional to the reduction in simulated disaster
        impact caused by the joint action.

        A high-severity action (deploy/evacuate) under high risk earns
        positive mitigation reward proportional to the current risk level.
        A warning action earns a smaller mitigation credit.
        No action earns zero mitigation.
        """
        a1, a2, a3 = actions
        max_severity = max(a1, a2, a3)

        # Base mitigation: risk reduction credit for appropriate responses
        if max_severity == ACTION_EVACUATE and risk > 0.5:
            # Full mitigation for evacuation during high risk
            mitigation = risk * 1.0
        elif max_severity == ACTION_DEPLOY and risk > 0.3:
            # Partial mitigation for resource deployment
            mitigation = risk * 0.7
        elif max_severity == ACTION_WARN and risk > 0.1:
            # Small credit for issuing a warning
            mitigation = risk * 0.3
        else:
            # No action or action taken at negligible risk
            mitigation = max(0.0, prev_risk - risk) * 0.5

        return float(np.clip(mitigation, 0.0, 2.0))

    def _false_alarm_penalty(
        self,
        risk: float,
        actions: Tuple[int, int, int],
    ) -> float:
        """
        R_false_alarm penalises high-severity actions triggered under
        low-risk conditions.

        Penalty scales with action severity and inversely with risk.
        A deployment or evacuation when risk < 0.25 is fully penalised.
        """
        penalty = 0.0
        for action in actions:
            if action in HIGH_SEVERITY_ACTIONS and risk < UNNECESSARY_ACTION_RISK_THRESHOLD:
                # Severity multiplier: evacuation > deployment
                severity_weight = 1.0 if action == ACTION_EVACUATE else 0.6
                # Penalty is stronger when risk is lower
                penalty += severity_weight * (UNNECESSARY_ACTION_RISK_THRESHOLD - risk)

        return float(np.clip(penalty, 0.0, 3.0))

    def _delay_penalty(
        self,
        risk: float,
        actions: Tuple[int, int, int],
        timestep: int,
    ) -> float:
        """
        R_delay penalises late interventions relative to hazard onset.

        If a hazard has been active (risk > 0.6) but no response has been
        issued, the delay penalty grows linearly with the number of steps
        elapsed since onset.
        """
        if self._hazard_onset_step is None:
            return 0.0  # no critical hazard yet

        # A response is defined as any action other than no-op
        any_response = any(a > ACTION_NO_OP for a in actions)

        if any_response and risk > 0.3:
            # Record when a response was finally issued
            if self._last_response_step is None:
                self._last_response_step = timestep

        if self._last_response_step is None:
            # Still waiting for a response — incur delay proportional to elapsed time
            delay_steps = timestep - self._hazard_onset_step
            penalty = risk * min(delay_steps / 10.0, 1.0)
        else:
            penalty = 0.0

        return float(np.clip(penalty, 0.0, 1.0))

    # -----------------------------------------------------------------------
    # Constraint cost
    # -----------------------------------------------------------------------

    def _constraint_costs(
        self,
        risk: float,
        actions: Tuple[int, int, int],
    ) -> Dict[int, float]:
        """
        Compute per-agent constraint cost C_i(s_t, a_t).

        C_i = 1.0 if agent i takes a high-severity action (deploy or evacuate)
              when the risk level is below the unnecessary-action threshold.
        C_i = 0.0 otherwise.

        The binary cost is consistent with the constraint threshold d_i = 0.10,
        meaning the policy is penalised if it violates constraints more than
        10% of steps on average (Eq. 5 of paper).
        """
        costs: Dict[int, float] = {}
        for agent_idx, action in enumerate(actions, start=1):
            if (action in HIGH_SEVERITY_ACTIONS
                    and risk < UNNECESSARY_ACTION_RISK_THRESHOLD):
                costs[agent_idx] = 1.0
            else:
                costs[agent_idx] = 0.0
        return costs

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _compute_risk_level(hazard_state: HazardState) -> float:
        """
        Aggregate scalar risk from the HazardState.

        Combines storm probability, rainfall intensity, and river level
        into a single risk score in [0, 1].
        """
        return float(np.clip(
            0.4 * hazard_state.storm_probability
            + 0.35 * hazard_state.rainfall_intensity
            + 0.25 * hazard_state.river_level,
            0.0,
            1.0,
        ))
