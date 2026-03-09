"""
Orchestration layer: translates joint agent actions into emergency response primitives.
"""
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ResponseType(IntEnum):
    NONE               = 0
    STORM_WARNING      = 1
    FLOOD_WATCH        = 2
    EVACUATION_ADVISORY = 3
    MANDATORY_EVACUATION = 4


# Maps (agent_id, action) -> ResponseType
_ACTION_RESPONSE_MAP = {
    (1, 1): ResponseType.STORM_WARNING,
    (1, 2): ResponseType.STORM_WARNING,
    (1, 3): ResponseType.STORM_WARNING,
    (2, 1): ResponseType.FLOOD_WATCH,
    (2, 2): ResponseType.FLOOD_WATCH,
    (2, 3): ResponseType.FLOOD_WATCH,
    (3, 1): ResponseType.EVACUATION_ADVISORY,
    (3, 2): ResponseType.EVACUATION_ADVISORY,
    (3, 3): ResponseType.MANDATORY_EVACUATION,
}

_RESPONSE_NAMES = {
    ResponseType.NONE:               "NONE",
    ResponseType.STORM_WARNING:      "STORM_WARNING",
    ResponseType.FLOOD_WATCH:        "FLOOD_WATCH",
    ResponseType.EVACUATION_ADVISORY: "EVACUATION_ADVISORY",
    ResponseType.MANDATORY_EVACUATION: "MANDATORY_EVACUATION",
}


@dataclass
class OrchestratorResponse:
    t: int
    action_dict: Dict[int, int]
    severity: int
    responses: List[ResponseType]
    risk: float

    def summary(self) -> str:
        resp_names = [_RESPONSE_NAMES[r] for r in self.responses] if self.responses else []
        act_tuple = tuple(self.action_dict.get(i, 0) for i in range(1, 4))
        return (
            f"[t={self.t:3d}] actions={act_tuple}  severity={self.severity}  "
            f"responses={resp_names}  risk={self.risk:.3f}"
        )


class Orchestrator:
    """
    Wraps a MARL policy and translates joint decisions into
    emergency-management response primitives.
    """

    def __init__(self, policy):
        self.policy = policy
        self._t = 0
        self._prev_severity = 0
        self._history: List[OrchestratorResponse] = []

    def reset(self):
        self._t = 0
        self._prev_severity = 0
        self._history = []

    def step(
        self,
        obs_dict: Dict[int, np.ndarray],
        info: Optional[dict] = None,
    ) -> OrchestratorResponse:
        info = info or {}
        severity = int(info.get("severity", 0))
        risk = float(info.get("risk_level", severity / 4.0))

        action_dict = self.policy.get_actions(obs_dict, deterministic=True)

        # Build active responses
        active: List[ResponseType] = []
        seen = set()
        for i in range(1, 4):
            act = action_dict.get(i, 0)
            resp = _ACTION_RESPONSE_MAP.get((i, act), ResponseType.NONE)
            if resp != ResponseType.NONE and resp not in seen:
                active.append(resp)
                seen.add(resp)
        active.sort()

        # Log escalation events
        if severity > self._prev_severity:
            logger.info(
                "ESCALATION  t=%d: severity %d → %d  (%s)",
                self._t,
                self._prev_severity,
                severity,
                _RESPONSE_NAMES.get(active[0], "NONE") if active else "NONE",
            )

        resp_obj = OrchestratorResponse(
            t=self._t,
            action_dict=action_dict,
            severity=severity,
            responses=active,
            risk=risk,
        )
        self._history.append(resp_obj)
        self._prev_severity = severity
        self._t += 1
        return resp_obj

    def get_episode_summary(self) -> dict:
        if not self._history:
            return {}

        severities = [r.severity for r in self._history]
        escalations = sum(
            1
            for a, b in zip(self._history, self._history[1:])
            if b.severity > a.severity
        )
        evac_steps = sum(
            1
            for r in self._history
            if ResponseType.EVACUATION_ADVISORY in r.responses
            or ResponseType.MANDATORY_EVACUATION in r.responses
        )
        storm_steps = sum(
            1 for r in self._history if ResponseType.STORM_WARNING in r.responses
        )
        T = len(self._history)
        return {
            "mean_severity":    float(np.mean(severities)),
            "escalations":      escalations,
            "evacuation_rate":  evac_steps / T,
            "storm_warning_rate": storm_steps / T,
        }
