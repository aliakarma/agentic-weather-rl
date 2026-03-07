"""
Agentic Orchestration Layer — Emergency Action Simulator
==========================================================
Purpose:
    Simulates real-world emergency response actions triggered by the RL agent's
    decisions. All functions are simulated (no real APIs or external services).
    The module is designed to be a drop-in replacement for real service
    integrations (e.g., SMS gateways, emergency dispatch APIs, dashboards).

    In a production deployment, each function would be replaced with a real
    API call to the corresponding emergency communication infrastructure.

Input:
    RiskLevel enum, action index (int), and contextual metadata (dict)

Output:
    ActionResult dataclass containing execution status, timestamp, and logs

Example usage:
    from orchestration.emergency_action_simulator import (
        EmergencyOrchestrator, RiskLevel
    )
    orchestrator = EmergencyOrchestrator(region="Northwest District")
    result = orchestrator.execute_action(action=2, risk_level=RiskLevel.HIGH)
    print(result.log_entry)
"""

import time
import logging
import datetime
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List, Dict, Any

# Configure module-level logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("EmergencyOrchestrator")


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------

class RiskLevel(IntEnum):
    """Qualitative risk classification used to contextualize actions."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Convert a composite risk score in [0, 1] to a RiskLevel."""
        if score < 0.35:
            return cls.LOW
        elif score < 0.65:
            return cls.MEDIUM
        elif score < 0.80:
            return cls.HIGH
        else:
            return cls.CRITICAL


@dataclass
class ActionResult:
    """
    Result of a single simulated emergency action.

    Attributes:
        action_index:   Integer action code (0–3).
        action_name:    Human-readable action label.
        risk_level:     Risk classification at time of action.
        success:        Whether the simulated action executed without error.
        timestamp:      ISO-format timestamp of execution.
        region:         Geographic region for the action.
        log_entry:      Human-readable summary string.
        metadata:       Additional context (message content, recipients, etc.).
    """
    action_index: int
    action_name: str
    risk_level: RiskLevel
    success: bool
    timestamp: str
    region: str
    log_entry: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual simulated action functions
# ---------------------------------------------------------------------------

def send_alert(
    region: str,
    risk_level: RiskLevel,
    message: Optional[str] = None,
) -> ActionResult:
    """
    Simulate sending an automated public alert for a weather risk event.

    In production, this would invoke an SMS gateway, push notification
    service, or IPAWS (Integrated Public Alert and Warning System) API.

    Args:
        region:     Geographic region identifier.
        risk_level: Severity classification of the weather event.
        message:    Custom alert message. Auto-generated if None.

    Returns:
        ActionResult indicating simulated success/failure and log details.
    """
    if message is None:
        message = (
            f"[ALERT] Severe weather warning for {region}. "
            f"Risk level: {risk_level.name}. "
            "Monitor conditions and follow emergency guidance."
        )

    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = (
        f"[ALERT SENT] Region={region}, Risk={risk_level.name}, "
        f"Message='{message[:60]}...', Time={timestamp}"
    )
    logger.info(log_entry)

    return ActionResult(
        action_index=1,
        action_name="Issue Early Warning",
        risk_level=risk_level,
        success=True,
        timestamp=timestamp,
        region=region,
        log_entry=log_entry,
        metadata={"message": message, "channel": "public_alert_system"},
    )


def notify_emergency_services(
    region: str,
    risk_level: RiskLevel,
    resource_type: str = "general",
) -> ActionResult:
    """
    Simulate notifying and mobilizing emergency response teams.

    In production, this would call a dispatch system API (e.g., CAD —
    Computer-Aided Dispatch) to mobilize fire, rescue, or medical units.

    Args:
        region:        Target geographic region.
        risk_level:    Severity level determining response scale.
        resource_type: Type of resources to mobilize ('general', 'flood',
                       'search_rescue', 'medical').

    Returns:
        ActionResult with dispatch details.
    """
    units_dispatched = {
        RiskLevel.LOW: 0,
        RiskLevel.MEDIUM: 2,
        RiskLevel.HIGH: 5,
        RiskLevel.CRITICAL: 10,
    }[risk_level]

    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = (
        f"[EMERGENCY SERVICES NOTIFIED] Region={region}, "
        f"Risk={risk_level.name}, Type={resource_type}, "
        f"Units={units_dispatched}, Time={timestamp}"
    )
    logger.warning(log_entry)

    return ActionResult(
        action_index=2,
        action_name="Prepare Emergency Resources",
        risk_level=risk_level,
        success=True,
        timestamp=timestamp,
        region=region,
        log_entry=log_entry,
        metadata={
            "units_dispatched": units_dispatched,
            "resource_type": resource_type,
            "dispatch_system": "simulated_CAD",
        },
    )


def recommend_evacuation(
    region: str,
    risk_level: RiskLevel,
    evacuation_zones: Optional[List[str]] = None,
) -> ActionResult:
    """
    Simulate issuing an evacuation recommendation or order.

    In production, this would interface with local government emergency
    management systems to issue official evacuation orders, activate
    shelters, and update route guidance systems.

    Args:
        region:           Geographic region.
        risk_level:       Severity level (should be HIGH or CRITICAL).
        evacuation_zones: List of specific zone identifiers to evacuate.
                          Defaults to the entire region.

    Returns:
        ActionResult with evacuation details.
    """
    if evacuation_zones is None:
        evacuation_zones = [f"{region}_Zone_A", f"{region}_Zone_B"]

    order_type = "MANDATORY" if risk_level == RiskLevel.CRITICAL else "VOLUNTARY"
    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = (
        f"[EVACUATION RECOMMENDED] Region={region}, "
        f"Order={order_type}, Risk={risk_level.name}, "
        f"Zones={evacuation_zones}, Time={timestamp}"
    )
    logger.critical(log_entry)

    return ActionResult(
        action_index=3,
        action_name="Recommend Evacuation",
        risk_level=risk_level,
        success=True,
        timestamp=timestamp,
        region=region,
        log_entry=log_entry,
        metadata={
            "order_type": order_type,
            "evacuation_zones": evacuation_zones,
            "shelters_activated": len(evacuation_zones),
        },
    )


def update_disaster_dashboard(
    region: str,
    risk_level: RiskLevel,
    weather_state: Optional[Dict[str, float]] = None,
    action_taken: Optional[str] = None,
) -> ActionResult:
    """
    Simulate updating a real-time disaster management dashboard.

    In production, this would make a REST API call to push new risk
    indicators, action logs, and weather state data to an operational
    monitoring dashboard.

    Args:
        region:        Geographic region.
        risk_level:    Current risk classification.
        weather_state: Dict with keys storm_probability, rainfall_intensity,
                       flood_risk_score, regional_risk_indicator.
        action_taken:  Human-readable description of the action triggered.

    Returns:
        ActionResult with dashboard update details.
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    payload = {
        "region": region,
        "risk_level": risk_level.name,
        "timestamp": timestamp,
        "weather_state": weather_state or {},
        "last_action": action_taken or "None",
    }
    log_entry = (
        f"[DASHBOARD UPDATED] Region={region}, "
        f"Risk={risk_level.name}, Action={action_taken}, Time={timestamp}"
    )
    logger.info(log_entry)

    return ActionResult(
        action_index=-1,  # Internal system action, not in agent's action space
        action_name="Update Disaster Dashboard",
        risk_level=risk_level,
        success=True,
        timestamp=timestamp,
        region=region,
        log_entry=log_entry,
        metadata={"dashboard_payload": payload},
    )


def no_action(region: str, risk_level: RiskLevel) -> ActionResult:
    """
    Represent the agent's decision to take no emergency action.

    Appropriate when the weather state is assessed as low risk.

    Args:
        region:     Geographic region.
        risk_level: Current risk level.

    Returns:
        ActionResult with a 'monitoring' status.
    """
    timestamp = datetime.datetime.utcnow().isoformat()
    log_entry = (
        f"[NO ACTION] Region={region}, Risk={risk_level.name}, "
        f"Status=Monitoring, Time={timestamp}"
    )
    logger.debug(log_entry)

    return ActionResult(
        action_index=0,
        action_name="No Action",
        risk_level=risk_level,
        success=True,
        timestamp=timestamp,
        region=region,
        log_entry=log_entry,
        metadata={"status": "monitoring"},
    )


# ---------------------------------------------------------------------------
# Orchestrator class
# ---------------------------------------------------------------------------

class EmergencyOrchestrator:
    """
    Agentic orchestration layer that maps RL agent decisions to
    simulated real-world emergency response actions.

    Maintains an action log for audit and analysis purposes.

    Args:
        region:  Default geographic region for all actions.
        verbose: If True, print action results to stdout.
    """

    ACTION_MAP = {
        0: "no_action",
        1: "send_alert",
        2: "notify_emergency_services",
        3: "recommend_evacuation",
    }

    def __init__(self, region: str = "Default Region", verbose: bool = True) -> None:
        self.region = region
        self.verbose = verbose
        self.action_log: List[ActionResult] = []

    def execute_action(
        self,
        action: int,
        risk_level: RiskLevel,
        weather_state: Optional[Dict[str, float]] = None,
    ) -> ActionResult:
        """
        Execute the emergency response action selected by the RL agent.

        After executing the primary action, the dashboard is always updated
        to reflect the latest system state.

        Args:
            action:        Integer action index (0–3).
            risk_level:    Current risk classification.
            weather_state: Optional weather state dict for dashboard update.

        Returns:
            ActionResult from the executed action.
        """
        if action not in self.ACTION_MAP:
            raise ValueError(f"Unknown action index: {action}. Must be 0–3.")

        # Execute primary action
        if action == 0:
            result = no_action(self.region, risk_level)
        elif action == 1:
            result = send_alert(self.region, risk_level)
        elif action == 2:
            result = notify_emergency_services(self.region, risk_level)
        elif action == 3:
            result = recommend_evacuation(self.region, risk_level)

        # Always update the dashboard regardless of action type
        update_disaster_dashboard(
            region=self.region,
            risk_level=risk_level,
            weather_state=weather_state,
            action_taken=result.action_name,
        )

        self.action_log.append(result)

        if self.verbose:
            print(f"  ▸ {result.log_entry}")

        return result

    def get_action_log(self) -> List[Dict[str, Any]]:
        """
        Return the full action history as a list of dictionaries.

        Returns:
            List of dicts with keys: action_name, risk_level, timestamp, success.
        """
        return [
            {
                "action_name": r.action_name,
                "risk_level": r.risk_level.name,
                "timestamp": r.timestamp,
                "success": r.success,
                "region": r.region,
            }
            for r in self.action_log
        ]

    def clear_log(self) -> None:
        """Clear the action history log."""
        self.action_log.clear()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Emergency Orchestrator Smoke Test ===")
    orchestrator = EmergencyOrchestrator(region="Southeast District", verbose=True)

    test_cases = [
        (0, RiskLevel.LOW),
        (1, RiskLevel.MEDIUM),
        (2, RiskLevel.HIGH),
        (3, RiskLevel.CRITICAL),
    ]

    weather_state = {
        "storm_probability": 0.85,
        "rainfall_intensity": 0.72,
        "flood_risk_score": 0.68,
        "regional_risk_indicator": 0.60,
    }

    for action, risk in test_cases:
        print(f"\n[Testing action={action}, risk={risk.name}]")
        orchestrator.execute_action(action, risk, weather_state)

    print(f"\nTotal actions logged: {len(orchestrator.action_log)}")
    print("Smoke test passed.")
