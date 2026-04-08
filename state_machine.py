"""
SRE-Bench: Incident State Machine
Manages action preconditions, state transitions, and causal tracking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from models import (
    ActionType,
    EpisodeState,
    IncidentData,
    IncidentType,
)


class PreconditionError(Exception):
    """Raised when an action's precondition is not met."""

    def __init__(self, action: str, reason: str):
        self.action = action
        self.reason = reason
        super().__init__(f"Precondition failed for '{action}': {reason}")


class IncidentStateMachine:
    """
    Tracks system state across action steps within an episode.
    Enforces action preconditions and manages causal relationships.
    """

    def __init__(self, incident: IncidentData):
        self.incident = incident
        self.state = EpisodeState(
            current_incident_id=incident.incident_id,
            system_state={
                node: "healthy"
                for node in incident.service_topology.nodes
            },
        )

        # Mark the primary service and affected chain as degraded
        primary = incident.alert_payload.service
        if primary in self.state.system_state:
            self.state.system_state[primary] = "degraded"

        for svc in incident.gold_affected_chain:
            if svc in self.state.system_state:
                self.state.system_state[svc] = "degraded"

        # Build the NetworkX graph for topology queries
        self._graph = nx.DiGraph()
        self._graph.add_nodes_from(incident.service_topology.nodes)
        self._graph.add_edges_from(
            [tuple(e) for e in incident.service_topology.edges]
        )

        # Track which services have had specific actions applied
        self._action_targets: Dict[str, Set[str]] = {}  # action -> set of services

    @property
    def is_done(self) -> bool:
        return self.state.episode_done

    @property
    def action_history(self) -> List[str]:
        return self.state.action_history

    def check_precondition(self, action: ActionType, target_service: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if the precondition for an action is met.
        Returns (is_valid, reason_if_invalid).
        """
        if self.state.episode_done:
            return False, "Episode is already complete"

        if action == ActionType.RESTART_SERVICE:
            if not self.state.logs_inspected:
                return False, "Must inspect_logs before restart_service"

        elif action == ActionType.SCALE_UP:
            # Requires capacity issue detected (check_metrics done + relevant incident type)
            if not self.state.metrics_checked:
                return False, "Must check_metrics before scale_up"
            if self.incident.incident_type not in (
                IncidentType.CPU_SPIKE,
                IncidentType.DB_OVERLOAD,
                IncidentType.CASCADE_FAILURE,
            ):
                return False, "scale_up only valid for capacity-related incidents"

        elif action == ActionType.ROLLBACK_DEPLOY:
            if not self.state.logs_inspected:
                return False, "Must inspect_logs before rollback_deploy"
            if self.incident.incident_type not in (
                IncidentType.DEPLOYMENT_FAILURE,
                IncidentType.CONFIG_ERROR,
            ):
                return False, "rollback_deploy only valid for deployment/config incidents"

        elif action == ActionType.VERIFY_ENDPOINT:
            if not self.state.remediation_applied:
                return False, "Must apply remediation before verify_endpoint"

        elif action == ActionType.RESOLVE:
            if not self.state.verification_done:
                return False, "Must verify_endpoint before resolve"

        return True, ""

    def execute_action(
        self,
        action: ActionType,
        target_service: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an action, updating state and returning action results.
        Raises PreconditionError if precondition is not met.
        """
        # Check precondition
        is_valid, reason = self.check_precondition(action, target_service)
        if not is_valid:
            raise PreconditionError(action.value, reason)

        # Build action string for history
        action_str = action.value
        if target_service:
            action_str = f"{action.value}:{target_service}"

        self.state.action_history.append(action_str)
        self.state.step_count += 1

        # Execute action-specific logic
        result = self._dispatch_action(action, target_service)

        return result

    def _dispatch_action(
        self,
        action: ActionType,
        target_service: Optional[str],
    ) -> Dict[str, Any]:
        """Route action to handler and return result dictionary."""

        target = target_service or self.incident.alert_payload.service

        if action == ActionType.INSPECT_LOGS:
            return self._handle_inspect_logs(target)
        elif action == ActionType.CHECK_METRICS:
            return self._handle_check_metrics(target)
        elif action == ActionType.CHECK_SERVICE:
            return self._handle_check_service(target)
        elif action == ActionType.RESTART_SERVICE:
            return self._handle_restart_service(target)
        elif action == ActionType.SCALE_UP:
            return self._handle_scale_up(target)
        elif action == ActionType.ROLLBACK_DEPLOY:
            return self._handle_rollback_deploy(target)
        elif action == ActionType.VERIFY_ENDPOINT:
            return self._handle_verify_endpoint(target)
        elif action == ActionType.RESOLVE:
            return self._handle_resolve()
        elif action == ActionType.CHECK_TOPOLOGY:
            return self._handle_check_topology()
        elif action == ActionType.ACKNOWLEDGE_ALERT:
            return self._handle_acknowledge_alert()
        else:
            return {"status": "unknown_action", "message": f"Action {action} not recognized"}

    def _handle_inspect_logs(self, target: str) -> Dict[str, Any]:
        """Return log lines from the target service."""
        self.state.logs_inspected = True
        self.state.diagnosis_done = True

        if target not in self.state.services_inspected:
            self.state.services_inspected.append(target)

        # Return logs (filtered to target service if matching primary)
        primary = self.incident.alert_payload.service
        if target == primary:
            logs = self.incident.logs
        else:
            # Generate some generic logs for other services
            logs = [
                f"[2026-04-06 12:00:{i:02d}] INFO  {target}: Service operating normally"
                if target not in self.incident.gold_affected_chain
                else f"[2026-04-06 12:00:{i:02d}] WARN  {target}: Degraded performance detected"
                for i in range(min(20, len(self.incident.logs)))
            ]

        return {
            "status": "success",
            "service": target,
            "log_count": len(logs),
            "logs": logs,
        }

    def _handle_check_metrics(self, target: str) -> Dict[str, Any]:
        """Return metrics for the target service."""
        self.state.metrics_checked = True

        primary = self.incident.alert_payload.service
        if target == primary:
            metrics = self.incident.metrics.model_dump()
        else:
            # Return normal metrics for non-affected services
            if target in self.incident.gold_affected_chain:
                metrics = {
                    "cpu": round(self.incident.metrics.cpu * 0.7, 1),
                    "memory": round(self.incident.metrics.memory * 0.8, 1),
                    "latency_ms": round(self.incident.metrics.latency_ms * 0.5, 1),
                    "error_rate": round(self.incident.metrics.error_rate * 0.6, 2),
                }
            else:
                metrics = {
                    "cpu": 25.0,
                    "memory": 40.0,
                    "latency_ms": 120.0,
                    "error_rate": 0.5,
                }

        return {
            "status": "success",
            "service": target,
            "metrics": metrics,
        }

    def _handle_check_service(self, target: str) -> Dict[str, Any]:
        """Return health status of the target service."""
        status = self.state.system_state.get(target, "unknown")
        return {
            "status": "success",
            "service": target,
            "health": status,
            "uptime": "2d 14h 23m" if status == "healthy" else "0h 5m",
        }

    def _handle_restart_service(self, target: str) -> Dict[str, Any]:
        """Simulate service restart."""
        self.state.remediation_applied = True
        self.state.services_restarted.append(target)

        # Restart fixes the service if it's the right remediation
        primary = self.incident.alert_payload.service
        if target == primary and self.incident.incident_type in (
            IncidentType.CPU_SPIKE,
            IncidentType.MEMORY_LEAK,
            IncidentType.CASCADE_FAILURE,
            IncidentType.DEPENDENCY_TIMEOUT,
        ):
            self.state.system_state[target] = "healthy"
            return {
                "status": "success",
                "service": target,
                "message": f"Service {target} restarted successfully",
                "new_health": "healthy",
            }
        else:
            return {
                "status": "partial",
                "service": target,
                "message": f"Service {target} restarted but issue persists",
                "new_health": "degraded",
            }

    def _handle_scale_up(self, target: str) -> Dict[str, Any]:
        """Simulate scaling up a service."""
        self.state.remediation_applied = True

        if self.incident.incident_type in (IncidentType.CPU_SPIKE, IncidentType.DB_OVERLOAD):
            self.state.system_state[target] = "healthy"
            return {
                "status": "success",
                "service": target,
                "message": f"Scaled {target} from 3 to 6 replicas",
                "new_replicas": 6,
            }
        else:
            return {
                "status": "partial",
                "service": target,
                "message": f"Scaled {target} but underlying issue not resolved",
                "new_replicas": 6,
            }

    def _handle_rollback_deploy(self, target: str) -> Dict[str, Any]:
        """Simulate rolling back a deployment."""
        self.state.remediation_applied = True

        if self.incident.incident_type in (
            IncidentType.DEPLOYMENT_FAILURE,
            IncidentType.CONFIG_ERROR,
        ):
            self.state.system_state[target] = "healthy"
            return {
                "status": "success",
                "service": target,
                "message": f"Rolled back {target} to previous stable version",
                "rolled_back_to": "v2.3.1",
            }
        else:
            return {
                "status": "partial",
                "service": target,
                "message": f"Rollback completed but issue not caused by deployment",
                "rolled_back_to": "v2.3.1",
            }

    def _handle_verify_endpoint(self, target: str) -> Dict[str, Any]:
        """Verify that the remediation was effective."""
        primary = self.incident.alert_payload.service
        primary_status = self.state.system_state.get(primary, "degraded")

        if primary_status == "healthy":
            self.state.verification_done = True
            return {
                "status": "success",
                "service": target or primary,
                "message": "Endpoint responding correctly — HTTP 200",
                "latency_ms": 45,
                "verified": True,
            }
        else:
            return {
                "status": "failure",
                "service": target or primary,
                "message": "Endpoint still returning errors — HTTP 503",
                "latency_ms": 30000,
                "verified": False,
            }

    def _handle_resolve(self) -> Dict[str, Any]:
        """Mark the episode as resolved."""
        self.state.episode_done = True
        return {
            "status": "success",
            "message": "Incident marked as resolved",
            "total_steps": self.state.step_count,
            "actions_taken": len(self.state.action_history),
        }

    def _handle_check_topology(self) -> Dict[str, Any]:
        """Return the service dependency graph."""
        self.state.topology_checked = True

        return {
            "status": "success",
            "nodes": list(self._graph.nodes()),
            "edges": [list(e) for e in self._graph.edges()],
            "node_count": self._graph.number_of_nodes(),
            "edge_count": self._graph.number_of_edges(),
        }

    def _handle_acknowledge_alert(self) -> Dict[str, Any]:
        """Acknowledge the alert."""
        self.state.alert_acknowledged = True
        return {
            "status": "success",
            "message": f"Alert for {self.incident.alert_payload.service} acknowledged",
            "incident_id": self.incident.incident_id,
        }

    def get_state_dict(self) -> Dict[str, Any]:
        """Return the current state as a dictionary."""
        return {
            "incident_id": self.state.current_incident_id,
            "step_count": self.state.step_count,
            "action_history": self.state.action_history.copy(),
            "system_state": self.state.system_state.copy(),
            "diagnosis_done": self.state.diagnosis_done,
            "remediation_applied": self.state.remediation_applied,
            "verification_done": self.state.verification_done,
            "episode_done": self.state.episode_done,
            "cumulative_reward": self.state.cumulative_reward,
            "alert_acknowledged": self.state.alert_acknowledged,
        }

    def detect_action_loop(self) -> bool:
        """Check if the last action was a repeat (loop detection)."""
        history = self.state.action_history
        if len(history) < 2:
            return False
        return history[-1] == history[-2]

    def get_cascade_path(self, source: str) -> List[str]:
        """Get the cascade failure path from a source service using BFS."""
        if source not in self._graph:
            return [source]

        visited = []
        queue = [source]
        seen = {source}

        while queue:
            current = queue.pop(0)
            visited.append(current)
            for neighbor in self._graph.successors(current):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

        return visited
