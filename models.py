"""
SRE-Bench: Pydantic Data Models
OpenEnv-compliant request/response models, enums, and internal state types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """10 valid actions an agent can take during an episode."""

    INSPECT_LOGS = "inspect_logs"
    CHECK_METRICS = "check_metrics"
    CHECK_SERVICE = "check_service"
    RESTART_SERVICE = "restart_service"
    SCALE_UP = "scale_up"
    ROLLBACK_DEPLOY = "rollback_deploy"
    VERIFY_ENDPOINT = "verify_endpoint"
    RESOLVE = "resolve"
    CHECK_TOPOLOGY = "check_topology"
    ACKNOWLEDGE_ALERT = "acknowledge_alert"


class Severity(str, Enum):
    """Incident severity levels."""

    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class IncidentType(str, Enum):
    """8 incident categories in the SRE-Bench dataset."""

    CPU_SPIKE = "cpu_spike"
    MEMORY_LEAK = "memory_leak"
    NETWORK_PARTITION = "network_partition"
    DEPLOYMENT_FAILURE = "deployment_failure"
    DB_OVERLOAD = "db_overload"
    CASCADE_FAILURE = "cascade_failure"
    CONFIG_ERROR = "config_error"
    DEPENDENCY_TIMEOUT = "dependency_timeout"


class TaskType(str, Enum):
    """Three progressive task difficulty levels."""

    TASK1 = "task1"
    TASK2 = "task2"
    TASK3 = "task3"


# ─── Nested Models ──────────────────────────────────────────────────────────────


class AlertPayload(BaseModel):
    """Alert details as received by on-call engineer."""

    title: str
    service: str
    severity: Severity
    timestamp: str
    runbook_url: str = ""


class ServiceMetrics(BaseModel):
    """Runtime metrics for a service."""

    cpu: float = Field(ge=0.0, le=100.0)
    memory: float = Field(ge=0.0, le=100.0)
    latency_ms: float = Field(ge=0.0)
    error_rate: float = Field(ge=0.0, le=100.0)


class ServiceTopology(BaseModel):
    """NetworkX DiGraph serialized as node/edge lists."""

    nodes: List[str]
    edges: List[List[str]]


# ─── Core API Models ───────────────────────────────────────────────────────────


class SREObservation(BaseModel):
    """Observation returned by env.reset() and accessible via env.state()."""

    incident_id: str
    alert_payload: AlertPayload
    service_topology: ServiceTopology
    logs: List[str] = Field(default_factory=list, max_length=50)
    metrics: ServiceMetrics
    action_history: List[str] = Field(default_factory=list)
    step_number: int = 0
    incident_resolved: bool = False


class SREAction(BaseModel):
    """Action submitted by the agent via POST /step."""

    action_type: ActionType
    target_service: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class SREReward(BaseModel):
    """Reward returned after each step."""

    value: float = Field(ge=-1.0, le=1.0)
    cumulative: float = 0.0
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ─── Dataset / Incident Models ─────────────────────────────────────────────────


class IncidentData(BaseModel):
    """Full incident record stored in data/incidents.json."""

    incident_id: str
    difficulty: str  # easy | medium | hard
    incident_type: IncidentType
    alert_payload: AlertPayload
    service_topology: ServiceTopology
    logs: List[str] = Field(default_factory=list)
    metrics: ServiceMetrics
    gold_root_cause: str
    gold_triggered_by: str
    gold_affected_chain: List[str] = Field(default_factory=list)
    gold_action_sequence: List[str] = Field(default_factory=list)
    red_herrings: List[str] = Field(default_factory=list)


# ─── Internal Episode State ────────────────────────────────────────────────────


class EpisodeState(BaseModel):
    """In-memory state tracking for a running episode."""

    current_incident_id: str = ""
    step_count: int = 0
    action_history: List[str] = Field(default_factory=list)
    system_state: Dict[str, str] = Field(default_factory=dict)
    diagnosis_done: bool = False
    remediation_applied: bool = False
    verification_done: bool = False
    episode_done: bool = False
    cumulative_reward: float = 0.0
    alert_acknowledged: bool = False
    logs_inspected: bool = False
    metrics_checked: bool = False
    topology_checked: bool = False
    services_inspected: List[str] = Field(default_factory=list)
    services_restarted: List[str] = Field(default_factory=list)
    root_cause_identified: Optional[str] = None


# ─── API Request Models ────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    task: TaskType = TaskType.TASK1
    seed: Optional[int] = Field(default=None, ge=0)


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str = "ok"
