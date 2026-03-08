"""
src/orchestration/orchestration.py
=====================================
Layer 3: Orchestration interface — translates joint agent actions into
simulated emergency response callbacks.

This module implements the orchestration layer described in Section 3.4
of the paper:

  "The orchestration interface maps the discrete joint action a_t =
   (a_1, a_2, a_3) produced by the three agents onto a set of emergency
   response primitives. The interface is designed to model the decision
   pipeline of a real-world emergency management system."

Action-to-response mapping (Section 3.3):
    a_i = 0 — no action
    a_i = 1 — issue warning
    a_i = 2 — deploy resources
    a_i = 3 — initiate evacuation

Response types generated:
    - Storm warning bulletin       (from Agent 1, action ≥ 1)
    - Flood mitigation deployment  (from Agent 2, action ≥ 2)
    - Resource pre-positioning     (from Agent 2, action = 2)
    - Evacuation order             (from Agent 3, action = 3)
    - All-clear notification       (when all agents select no-op)

The Orchestrator also tracks response history, computes escalation/de-
escalation transitions, and provides a structured log for post-hoc
analysis and decision accuracy evaluation.

Usage
-----
    from src.orchestration.orchestration import Orchestrator
    from src.algorithms.lagrangian_ctde import LagrangianCTDE

    orch = Orchestrator(policy=ctde_agent)
    response = orch.step(obs_dict)
    print(response.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action and response enums
# ---------------------------------------------------------------------------

class Action(IntEnum):
    NO_OP    = 0
    WARN     = 1
    DEPLOY   = 2
    EVACUATE = 3


class ResponseType(IntEnum):
    """Discrete emergency response categories."""
    ALL_CLEAR              = 0   # no threat detected
    STORM_WATCH            = 1   # elevated storm monitoring
    STORM_WARNING          = 2   # imminent storm risk
    FLOOD_WATCH            = 3   # elevated flood monitoring
    FLOOD_MITIGATION       = 4   # sandbags, pump deployment
    RESOURCE_PREPOSITION   = 5   # pre-position rescue assets
    EVACUATION_ADVISORY    = 6   # voluntary evacuation guidance
    EVACUATION_ORDER       = 7   # mandatory evacuation order
    MULTI_HAZARD_RESPONSE  = 8   # combined storm + flood emergency


# ---------------------------------------------------------------------------
# Response record
# ---------------------------------------------------------------------------

@dataclass
class EmergencyResponse:
    """
    Structured record of a single orchestration decision.

    Produced by Orchestrator.step() for each environment timestep.
    """
    timestep:       int
    joint_action:   Tuple[int, int, int]     # (a_1, a_2, a_3)
    response_types: List[ResponseType]       # active responses
    severity:       int                      # 0–3 aggregate severity
    bulletins:      List[str]                # human-readable alerts
    callbacks_fired: List[str]               # names of callbacks invoked
    risk_level:     float                    # scalar risk from info dict
    timestamp:      float = field(default_factory=time.time)

    def summary(self) -> str:
        """Return a compact single-line summary for logging."""
        names = [r.name for r in self.response_types] or ["ALL_CLEAR"]
        return (
            f"[t={self.timestep:3d}] "
            f"actions={self.joint_action}  "
            f"severity={self.severity}  "
            f"responses={names}  "
            f"risk={self.risk_level:.3f}"
        )

    def is_emergency(self) -> bool:
        """True if any high-severity response is active."""
        return self.severity >= 2

    @property
    def highest_response(self) -> ResponseType:
        if not self.response_types:
            return ResponseType.ALL_CLEAR
        return max(self.response_types)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Translates joint agent actions into simulated emergency responses.

    Acts as the Layer 3 interface in the three-layer architecture:
      Layer 1 → ViT encoder (φ_t)
      Layer 2 → Lagrangian CTDE-PPO (joint action a_t)
      Layer 3 → Orchestrator (emergency response primitives)

    Registers callback hooks for each response type so downstream
    systems (alerting APIs, GIS updates, resource dispatch) can
    subscribe to specific events.

    Parameters
    ----------
    policy
        Any object with a `get_actions(obs_dict, deterministic=True)`
        method returning Dict[int, int]. Compatible with LagrangianCTDE
        and all baseline agents that implement `.act()`.
    risk_threshold_warn   : float  Risk level above which warnings are forced.
    risk_threshold_deploy : float  Risk level above which deployment is forced.
    risk_threshold_evac   : float  Risk level above which evacuation is forced.
    """

    def __init__(
        self,
        policy=None,
        risk_threshold_warn:   float = 0.3,
        risk_threshold_deploy: float = 0.5,
        risk_threshold_evac:   float = 0.7,
    ) -> None:
        self.policy = policy
        self.risk_threshold_warn   = risk_threshold_warn
        self.risk_threshold_deploy = risk_threshold_deploy
        self.risk_threshold_evac   = risk_threshold_evac

        # Response history for post-hoc analysis
        self._history: List[EmergencyResponse] = []
        self._timestep: int = 0

        # Callback registry: ResponseType → list of callable hooks
        self._callbacks: Dict[ResponseType, List[Callable]] = {
            rt: [] for rt in ResponseType
        }

        # Escalation state tracker
        self._prev_severity: int = 0

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def step(
        self,
        obs_dict: Dict[int, np.ndarray],
        info: Optional[Dict[str, Any]] = None,
        deterministic: bool = True,
    ) -> EmergencyResponse:
        """
        Produce an emergency response for one environment timestep.

        Calls the policy to obtain joint action a_t, then maps the
        action tuple to emergency response primitives.

        Parameters
        ----------
        obs_dict : dict[int, np.ndarray]
            Local agent observations keyed by agent_id {1, 2, 3}.
        info : dict | None
            Optional info dict from env.step() containing risk_level.
        deterministic : bool
            Use greedy policy actions (default True for deployment).

        Returns
        -------
        EmergencyResponse
            Structured response record for this timestep.
        """
        # ── Obtain joint action from policy ──────────────────────────────
        if self.policy is not None:
            # LagrangianCTDE exposes get_actions(); baselines expose act()
            if hasattr(self.policy, "get_actions"):
                joint_action_dict = self.policy.get_actions(
                    obs_dict, deterministic=deterministic
                )
            elif hasattr(self.policy, "act"):
                joint_action_dict = self.policy.act(
                    obs_dict, deterministic=deterministic
                )
            else:
                raise AttributeError(
                    "Policy must implement get_actions() or act()."
                )
        else:
            # Fallback: no-op for all agents (for testing without a policy)
            joint_action_dict = {1: 0, 2: 0, 3: 0}

        joint_action = (
            int(joint_action_dict[1]),
            int(joint_action_dict[2]),
            int(joint_action_dict[3]),
        )

        # ── Extract risk level from info dict ─────────────────────────────
        risk_level = 0.0
        if info is not None:
            risk_level = float(info.get("risk_level", 0.0))

        # ── Translate actions to response primitives ──────────────────────
        response = self._translate(joint_action, risk_level, self._timestep)

        # ── Fire registered callbacks ─────────────────────────────────────
        for rt in response.response_types:
            for cb in self._callbacks.get(rt, []):
                try:
                    cb(response)
                except Exception as exc:           # never let a callback crash the loop
                    logger.warning("Callback for %s raised: %s", rt.name, exc)

        # ── Log escalation/de-escalation transitions ──────────────────────
        if response.severity > self._prev_severity:
            logger.info(
                "ESCALATION  t=%d: severity %d → %d  (%s)",
                self._timestep, self._prev_severity, response.severity,
                ", ".join(r.name for r in response.response_types),
            )
        elif response.severity < self._prev_severity:
            logger.info(
                "DE-ESCALATION t=%d: severity %d → %d",
                self._timestep, self._prev_severity, response.severity,
            )

        self._prev_severity = response.severity
        self._history.append(response)
        self._timestep += 1

        return response

    def reset(self) -> None:
        """Reset orchestration state for a new episode."""
        self._history = []
        self._timestep = 0
        self._prev_severity = 0

    # -----------------------------------------------------------------------
    # Callback registration
    # -----------------------------------------------------------------------

    def register_callback(
        self, response_type: ResponseType, callback: Callable[[EmergencyResponse], None]
    ) -> None:
        """
        Register a callback function for a specific response type.

        The callback is invoked every time that response type is activated.

        Example
        -------
        >>> def alert_handler(resp):
        ...     print("ALERT:", resp.bulletins)
        >>> orch.register_callback(ResponseType.EVACUATION_ORDER, alert_handler)
        """
        self._callbacks[response_type].append(callback)

    # -----------------------------------------------------------------------
    # Response history and analytics
    # -----------------------------------------------------------------------

    @property
    def history(self) -> List[EmergencyResponse]:
        return list(self._history)

    def response_rate(self, response_type: ResponseType) -> float:
        """Fraction of timesteps where a given response type was active."""
        if not self._history:
            return 0.0
        count = sum(1 for r in self._history if response_type in r.response_types)
        return count / len(self._history)

    def mean_severity(self) -> float:
        """Mean severity across all timesteps in the current episode."""
        if not self._history:
            return 0.0
        return float(np.mean([r.severity for r in self._history]))

    def escalation_count(self) -> int:
        """Number of escalation events (severity increase)."""
        count = 0
        for i in range(1, len(self._history)):
            if self._history[i].severity > self._history[i - 1].severity:
                count += 1
        return count

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return a structured summary dict for logging and evaluation."""
        if not self._history:
            return {}
        actions_arr = np.array([r.joint_action for r in self._history])
        return {
            "n_steps":           len(self._history),
            "mean_severity":     self.mean_severity(),
            "escalations":       self.escalation_count(),
            "evacuation_rate":   self.response_rate(ResponseType.EVACUATION_ORDER),
            "storm_warning_rate": self.response_rate(ResponseType.STORM_WARNING),
            "flood_deploy_rate": self.response_rate(ResponseType.FLOOD_MITIGATION),
            "all_clear_rate":    self.response_rate(ResponseType.ALL_CLEAR),
            "action_frequencies": {
                f"agent{i}_action{a}":
                    int((actions_arr[:, i - 1] == a).sum())
                for i in range(1, 4)
                for a in range(4)
            },
        }

    # -----------------------------------------------------------------------
    # Core translation logic
    # -----------------------------------------------------------------------

    def _translate(
        self,
        joint_action: Tuple[int, int, int],
        risk_level: float,
        timestep: int,
    ) -> EmergencyResponse:
        """
        Map joint action tuple to emergency response primitives.

        Translation rules (aligned with Table 2 action encoding):

          Agent 1 (Storm Detection):
            action=0 → no storm alert
            action=1 → STORM_WATCH
            action≥2 → STORM_WARNING
            action=3 → STORM_WARNING (+ escalates to EVACUATION_ORDER if Agent 3 = 3)

          Agent 2 (Flood Risk):
            action=0 → no flood alert
            action=1 → FLOOD_WATCH
            action=2 → FLOOD_MITIGATION + RESOURCE_PREPOSITION
            action=3 → FLOOD_MITIGATION + RESOURCE_PREPOSITION (maximum flood response)

          Agent 3 (Evacuation Planning):
            action=0 → no evacuation
            action=1 → EVACUATION_ADVISORY
            action=2 → EVACUATION_ADVISORY + RESOURCE_PREPOSITION
            action=3 → EVACUATION_ORDER

          Combined:
            all no-op         → ALL_CLEAR
            storm≥2 AND flood≥2 AND evac≥2 → MULTI_HAZARD_RESPONSE
        """
        a1, a2, a3 = [Action(a) for a in joint_action]
        responses: List[ResponseType] = []
        bulletins: List[str] = []
        callbacks_fired: List[str] = []

        # ── Agent 1: Storm Detection ──────────────────────────────────────
        if a1 == Action.WARN:
            responses.append(ResponseType.STORM_WATCH)
            bulletins.append(
                "STORM WATCH: Elevated atmospheric activity detected. "
                "Monitor conditions closely."
            )
            callbacks_fired.append("storm_watch_alert")

        elif a1 in (Action.DEPLOY, Action.EVACUATE):
            responses.append(ResponseType.STORM_WARNING)
            bulletins.append(
                "⚠ STORM WARNING: High-probability storm event imminent. "
                "Prepare for impact within the forecast window."
            )
            callbacks_fired.append("storm_warning_broadcast")

        # ── Agent 2: Flood Risk Assessment ───────────────────────────────
        if a2 == Action.WARN:
            responses.append(ResponseType.FLOOD_WATCH)
            bulletins.append(
                "FLOOD WATCH: River levels and rainfall intensity elevated. "
                "Flood risk increasing."
            )
            callbacks_fired.append("flood_watch_alert")

        elif a2 in (Action.DEPLOY, Action.EVACUATE):
            responses.append(ResponseType.FLOOD_MITIGATION)
            responses.append(ResponseType.RESOURCE_PREPOSITION)
            bulletins.append(
                "🚧 FLOOD MITIGATION DEPLOYED: Emergency pumping assets and "
                "flood barriers activated. Rescue resources pre-positioned."
            )
            callbacks_fired.extend(["flood_mitigation_deploy", "resource_preposition"])

        # ── Agent 3: Evacuation Planning ──────────────────────────────────
        if a3 == Action.WARN:
            responses.append(ResponseType.EVACUATION_ADVISORY)
            bulletins.append(
                "EVACUATION ADVISORY: Residents in risk zones advised to "
                "prepare for possible evacuation. Stay alert for updates."
            )
            callbacks_fired.append("evacuation_advisory_issue")

        elif a3 == Action.DEPLOY:
            responses.append(ResponseType.EVACUATION_ADVISORY)
            responses.append(ResponseType.RESOURCE_PREPOSITION)
            bulletins.append(
                "EVACUATION ADVISORY + RESOURCE PREP: High vulnerability "
                "detected. Evacuation corridors being pre-cleared."
            )
            callbacks_fired.extend(["evacuation_advisory_issue", "resource_preposition"])

        elif a3 == Action.EVACUATE:
            responses.append(ResponseType.EVACUATION_ORDER)
            bulletins.append(
                "🚨 MANDATORY EVACUATION ORDER: Immediate evacuation required "
                "for all residents in designated flood and storm risk zones. "
                "Emergency shelters are open and accessible."
            )
            callbacks_fired.append("evacuation_order_issue")

        # ── Combined multi-hazard check ────────────────────────────────────
        if (a1 >= Action.DEPLOY and a2 >= Action.DEPLOY
                and a3 >= Action.DEPLOY):
            # Upgrade all responses to MULTI_HAZARD_RESPONSE
            responses = [ResponseType.MULTI_HAZARD_RESPONSE]
            bulletins.append(
                "🔴 MULTI-HAZARD EMERGENCY: Simultaneous storm and flood threat "
                "with high population vulnerability. Full emergency response "
                "protocol activated. All agencies on standby."
            )
            callbacks_fired.append("multi_hazard_response_activate")

        # ── All-clear ─────────────────────────────────────────────────────
        if not responses:
            responses.append(ResponseType.ALL_CLEAR)
            callbacks_fired.append("all_clear_status")

        # ── Severity score (0–3) ──────────────────────────────────────────
        severity = self._compute_severity(responses, risk_level)

        return EmergencyResponse(
            timestep=timestep,
            joint_action=joint_action,
            response_types=list(set(responses)),  # deduplicate
            severity=severity,
            bulletins=bulletins,
            callbacks_fired=list(set(callbacks_fired)),
            risk_level=risk_level,
        )

    @staticmethod
    def _compute_severity(
        responses: List[ResponseType], risk_level: float
    ) -> int:
        """
        Compute aggregate severity 0–3 from active responses.

          0 — ALL_CLEAR or low risk
          1 — Watch / advisory
          2 — Warning / mitigation
          3 — Mandatory evacuation / multi-hazard
        """
        if ResponseType.MULTI_HAZARD_RESPONSE in responses:
            return 3
        if ResponseType.EVACUATION_ORDER in responses:
            return 3
        if ResponseType.STORM_WARNING in responses or ResponseType.FLOOD_MITIGATION in responses:
            return 2
        if (ResponseType.STORM_WATCH in responses
                or ResponseType.FLOOD_WATCH in responses
                or ResponseType.EVACUATION_ADVISORY in responses):
            return 1
        # Scale with risk even during no-op
        return 1 if risk_level > 0.5 else 0
