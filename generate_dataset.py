"""
SRE-Bench: Synthetic Incident Dataset Generator
Generates 90 incidents (30 easy, 30 medium, 30 hard) using Faker + NetworkX.
Run: python generate_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

from faker import Faker
import networkx as nx

# Seed for reproducibility
GLOBAL_SEED = 42
fake = Faker()
Faker.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ─── Service Catalog ────────────────────────────────────────────────────────────

SERVICES = [
    "api-gateway", "auth-service", "user-service", "payment-service",
    "order-service", "inventory-service", "notification-service",
    "search-service", "analytics-service", "cache-service",
    "database-primary", "database-replica", "message-queue",
    "cdn-edge", "load-balancer", "config-server",
    "logging-service", "monitoring-service", "billing-service",
    "recommendation-engine",
]

INCIDENT_TYPES = [
    "cpu_spike", "memory_leak", "network_partition",
    "deployment_failure", "db_overload", "cascade_failure",
    "config_error", "dependency_timeout",
]

SEVERITY_MAP = {
    "cpu_spike": ["P2", "P3"],
    "memory_leak": ["P2", "P3"],
    "network_partition": ["P1", "P2"],
    "deployment_failure": ["P1", "P2"],
    "db_overload": ["P1", "P2"],
    "cascade_failure": ["P1"],
    "config_error": ["P2", "P3"],
    "dependency_timeout": ["P2", "P3"],
}

RUNBOOK_TEMPLATES = {
    "cpu_spike": "https://runbooks.internal/cpu-spike-{service}",
    "memory_leak": "https://runbooks.internal/memory-leak-{service}",
    "network_partition": "https://runbooks.internal/network-partition",
    "deployment_failure": "https://runbooks.internal/rollback-deploy-{service}",
    "db_overload": "https://runbooks.internal/db-overload-{service}",
    "cascade_failure": "https://runbooks.internal/cascade-failure",
    "config_error": "https://runbooks.internal/config-error-{service}",
    "dependency_timeout": "https://runbooks.internal/dependency-timeout-{service}",
}

# ─── Log Templates ──────────────────────────────────────────────────────────────

LOG_TEMPLATES = {
    "cpu_spike": [
        "[{ts}] WARN  {svc}: CPU usage at {val:.1f}% — threshold 80%",
        "[{ts}] ERROR {svc}: Thread pool exhausted, {n} threads blocked",
        "[{ts}] WARN  {svc}: GC pause {val:.0f}ms exceeds 200ms target",
        "[{ts}] INFO  {svc}: Auto-scaling triggered but no capacity available",
        "[{ts}] ERROR {svc}: Request queue depth {n}, shedding load",
    ],
    "memory_leak": [
        "[{ts}] WARN  {svc}: Heap usage at {val:.1f}% — OOM risk",
        "[{ts}] ERROR {svc}: Allocation failure in connection pool",
        "[{ts}] WARN  {svc}: GC frequency increased 3x in last 5 minutes",
        "[{ts}] INFO  {svc}: Object count {n} exceeds baseline by 400%",
        "[{ts}] ERROR {svc}: OutOfMemoryError in request handler",
    ],
    "network_partition": [
        "[{ts}] ERROR {svc}: Connection to {dep} timed out after 30s",
        "[{ts}] WARN  {svc}: DNS resolution failed for {dep}.internal",
        "[{ts}] ERROR {svc}: TCP RST received from {dep}",
        "[{ts}] WARN  {svc}: Circuit breaker OPEN for {dep}",
        "[{ts}] INFO  {svc}: Retrying connection to {dep} (attempt {n}/5)",
    ],
    "deployment_failure": [
        "[{ts}] ERROR {svc}: Health check failed after deploy v{ver}",
        "[{ts}] WARN  {svc}: Readiness probe failing — 0/{n} pods ready",
        "[{ts}] ERROR {svc}: CrashLoopBackOff — container exit code 137",
        "[{ts}] INFO  {svc}: Deploy v{ver} started at {ts}",
        "[{ts}] ERROR {svc}: Image pull failed for v{ver} — manifest unknown",
    ],
    "db_overload": [
        "[{ts}] ERROR {svc}: Query timeout after {val:.0f}ms — max 5000ms",
        "[{ts}] WARN  {svc}: Connection pool exhausted ({n}/{n} active)",
        "[{ts}] ERROR {svc}: Deadlock detected on table 'transactions'",
        "[{ts}] WARN  {svc}: Replication lag {val:.0f}s exceeds 10s threshold",
        "[{ts}] INFO  {svc}: Slow query log: SELECT * FROM orders WHERE ...",
    ],
    "cascade_failure": [
        "[{ts}] ERROR {svc}: Upstream {dep} returning 503 — propagating failure",
        "[{ts}] WARN  {svc}: Cascade detected: {dep} → {svc} degraded",
        "[{ts}] ERROR {svc}: All retries exhausted for {dep} dependency",
        "[{ts}] WARN  {svc}: Fallback cache MISS — no stale data available",
        "[{ts}] INFO  {svc}: Circuit breaker HALF-OPEN — testing {dep}",
    ],
    "config_error": [
        "[{ts}] ERROR {svc}: ConfigParseError — invalid YAML at line {n}",
        "[{ts}] WARN  {svc}: Environment variable DB_HOST is empty",
        "[{ts}] ERROR {svc}: Feature flag 'enable_v2_api' not found in config",
        "[{ts}] INFO  {svc}: Config reload triggered at {ts}",
        "[{ts}] ERROR {svc}: TLS certificate expired — connection refused",
    ],
    "dependency_timeout": [
        "[{ts}] ERROR {svc}: {dep} response time {val:.0f}ms exceeds SLA 500ms",
        "[{ts}] WARN  {svc}: {dep} health check returned HTTP 503",
        "[{ts}] ERROR {svc}: Timeout waiting for {dep} — circuit breaker OPEN",
        "[{ts}] INFO  {svc}: Degraded mode activated — {dep} unavailable",
        "[{ts}] WARN  {svc}: Retry budget exhausted for {dep} calls",
    ],
}

# ─── Red Herring Templates ──────────────────────────────────────────────────────

RED_HERRING_LOG_TEMPLATES = [
    "[{ts}] INFO  {svc}: Routine deployment v{ver} completed successfully",
    "[{ts}] INFO  {svc}: CPU usage at {val:.1f}% — within normal range",
    "[{ts}] INFO  {svc}: Scheduled maintenance window started",
    "[{ts}] WARN  {svc}: Minor latency increase — {val:.0f}ms (threshold 500ms)",
    "[{ts}] INFO  {svc}: Cache hit rate 98.2% — operating normally",
    "[{ts}] INFO  {svc}: Cron job 'cleanup_old_sessions' completed in {val:.0f}s",
    "[{ts}] INFO  {svc}: Config refresh — no changes detected",
]

# ─── Action Sequences ───────────────────────────────────────────────────────────

GOLD_SEQUENCES = {
    "cpu_spike": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_service", "restart_service", "verify_endpoint", "resolve",
    ],
    "memory_leak": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_service", "restart_service", "verify_endpoint", "resolve",
    ],
    "network_partition": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_topology", "check_service", "verify_endpoint", "resolve",
    ],
    "deployment_failure": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_service", "rollback_deploy", "verify_endpoint", "resolve",
    ],
    "db_overload": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_service", "scale_up", "verify_endpoint", "resolve",
    ],
    "cascade_failure": [
        "acknowledge_alert", "inspect_logs", "check_topology",
        "check_metrics", "check_service", "restart_service",
        "verify_endpoint", "resolve",
    ],
    "config_error": [
        "acknowledge_alert", "inspect_logs", "check_service",
        "rollback_deploy", "verify_endpoint", "resolve",
    ],
    "dependency_timeout": [
        "acknowledge_alert", "inspect_logs", "check_metrics",
        "check_topology", "check_service", "restart_service",
        "verify_endpoint", "resolve",
    ],
}


# ─── Helper Functions ───────────────────────────────────────────────────────────


def make_incident_id(difficulty: str, seed: int) -> str:
    """Generate deterministic incident ID from difficulty + seed."""
    raw = f"{difficulty}-{seed}"
    h = hashlib.sha256(raw.encode()).hexdigest()[:6]
    return f"INC-{difficulty.upper()}-{h}"


def build_topology(
    primary_service: str,
    incident_type: str,
    num_nodes: int = 8,
) -> Tuple[nx.DiGraph, List[str]]:
    """Build a realistic service topology graph and return the affected chain."""
    available = [s for s in SERVICES if s != primary_service]
    selected = random.sample(available, min(num_nodes - 1, len(available)))
    nodes = [primary_service] + selected

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    # Create realistic dependency edges
    # Primary service depends on some downstream services
    downstream = random.sample(selected, min(3, len(selected)))
    for ds in downstream:
        G.add_edge(primary_service, ds)

    # Add some inter-service dependencies
    for _ in range(random.randint(2, 5)):
        src = random.choice(selected)
        tgt = random.choice([n for n in nodes if n != src])
        G.add_edge(src, tgt)

    # Determine affected chain based on incident type
    if incident_type == "cascade_failure":
        # Build a longer chain
        chain = [primary_service]
        current = primary_service
        for _ in range(random.randint(2, 4)):
            successors = list(G.successors(current))
            if successors:
                nxt = random.choice(successors)
                chain.append(nxt)
                current = nxt
            else:
                break
        affected_chain = chain
    else:
        affected_chain = [primary_service] + downstream[:2]

    return G, affected_chain


def generate_logs(
    incident_type: str,
    primary_service: str,
    dependency: str,
    count: int = 50,
    include_red_herrings: bool = False,
) -> List[str]:
    """Generate synthetic log lines for an incident."""
    templates = LOG_TEMPLATES[incident_type]
    base_time = datetime(2026, 4, 6, random.randint(0, 23), random.randint(0, 59))
    logs = []

    for i in range(count):
        ts = (base_time + timedelta(seconds=i * random.randint(1, 30))).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        tmpl = random.choice(templates)
        log_line = tmpl.format(
            ts=ts,
            svc=primary_service,
            dep=dependency,
            val=random.uniform(60, 99),
            n=random.randint(10, 500),
            ver=f"2.{random.randint(1, 9)}.{random.randint(0, 20)}",
        )
        logs.append(log_line)

    # Inject red herrings for hard incidents
    if include_red_herrings:
        other_services = [s for s in SERVICES if s != primary_service]
        for _ in range(random.randint(5, 10)):
            idx = random.randint(0, len(logs) - 1)
            ts = (
                base_time + timedelta(seconds=idx * random.randint(1, 30))
            ).strftime("%Y-%m-%d %H:%M:%S")
            tmpl = random.choice(RED_HERRING_LOG_TEMPLATES)
            herring = tmpl.format(
                ts=ts,
                svc=random.choice(other_services),
                val=random.uniform(5, 40),
                ver=f"1.{random.randint(0, 9)}.{random.randint(0, 20)}",
            )
            logs[idx] = herring

    return logs[:50]


def generate_metrics(incident_type: str) -> Dict[str, float]:
    """Generate metrics matching the incident type."""
    base = {
        "cpu": round(random.uniform(15, 45), 1),
        "memory": round(random.uniform(30, 55), 1),
        "latency_ms": round(random.uniform(50, 200), 1),
        "error_rate": round(random.uniform(0.1, 2.0), 2),
    }

    # Override specific metrics based on incident type
    overrides = {
        "cpu_spike": {"cpu": round(random.uniform(85, 99), 1)},
        "memory_leak": {"memory": round(random.uniform(88, 99), 1)},
        "network_partition": {
            "latency_ms": round(random.uniform(5000, 30000), 1),
            "error_rate": round(random.uniform(25, 60), 2),
        },
        "deployment_failure": {
            "error_rate": round(random.uniform(40, 90), 2),
        },
        "db_overload": {
            "latency_ms": round(random.uniform(3000, 15000), 1),
            "cpu": round(random.uniform(75, 95), 1),
        },
        "cascade_failure": {
            "error_rate": round(random.uniform(30, 70), 2),
            "latency_ms": round(random.uniform(2000, 10000), 1),
        },
        "config_error": {
            "error_rate": round(random.uniform(50, 100), 2),
        },
        "dependency_timeout": {
            "latency_ms": round(random.uniform(5000, 30000), 1),
        },
    }

    base.update(overrides.get(incident_type, {}))
    return base


def generate_red_herrings(
    incident_type: str, primary_service: str
) -> List[str]:
    """Generate red herring descriptions for hard incidents."""
    other_services = [s for s in SERVICES if s != primary_service]
    herrings = []

    herring_pool = [
        f"Routine deployment on {random.choice(other_services)} completed 2h ago",
        f"CPU spike on {random.choice(other_services)} — but within auto-scale range",
        f"Config change on {random.choice(other_services)} — unrelated namespace",
        f"Scheduled cron job on {random.choice(other_services)} running normally",
        f"Memory usage on {random.choice(other_services)} at 65% — normal range",
        f"Minor latency bump on {random.choice(other_services)} — resolved in 30s",
        f"Alert noise: {random.choice(other_services)} pod restart — healthy after",
    ]

    herrings = random.sample(herring_pool, min(3, len(herring_pool)))
    return herrings


ROOT_CAUSE_TEMPLATES = {
    "cpu_spike": "Runaway process on {svc} consuming excessive CPU due to unoptimized query loop",
    "memory_leak": "Memory leak in {svc} connection pool — connections not released after timeout",
    "network_partition": "Network partition between {svc} and {dep} — switch failure in rack-2",
    "deployment_failure": "Failed deployment v{ver} on {svc} — incompatible schema migration",
    "db_overload": "Database overload on {svc} — missing index on high-traffic query path",
    "cascade_failure": "Cascade failure originating from {dep} — propagated to {svc} via dependency chain",
    "config_error": "Configuration error on {svc} — invalid database connection string after config push",
    "dependency_timeout": "Dependency timeout — {dep} unresponsive causing {svc} request backlog",
}


# ─── Main Generator ─────────────────────────────────────────────────────────────


def generate_incident(
    difficulty: str, index: int
) -> Dict[str, Any]:
    """Generate a single incident at the specified difficulty."""
    seed = hash(f"{difficulty}-{index}") % 100000
    random.seed(GLOBAL_SEED + seed)

    incident_type = INCIDENT_TYPES[index % len(INCIDENT_TYPES)]
    primary_service = random.choice(SERVICES[:10])  # Main services
    dependency = random.choice(
        [s for s in SERVICES if s != primary_service]
    )

    num_nodes = {"easy": 5, "medium": 8, "hard": 12}[difficulty]
    G, affected_chain = build_topology(primary_service, incident_type, num_nodes)

    severity = random.choice(SEVERITY_MAP[incident_type])

    # Higher severity for hard incidents
    if difficulty == "hard" and severity not in ("P1",):
        severity = random.choice(["P1", "P2"])

    base_time = datetime(2026, 4, 6, random.randint(0, 23), random.randint(0, 59))

    alert_payload = {
        "title": f"[{severity}] {incident_type.replace('_', ' ').title()} on {primary_service}",
        "service": primary_service,
        "severity": severity,
        "timestamp": base_time.isoformat() + "Z",
        "runbook_url": RUNBOOK_TEMPLATES[incident_type].format(service=primary_service),
    }

    logs = generate_logs(
        incident_type,
        primary_service,
        dependency,
        count=50,
        include_red_herrings=(difficulty == "hard"),
    )

    metrics = generate_metrics(incident_type)

    root_cause = ROOT_CAUSE_TEMPLATES[incident_type].format(
        svc=primary_service,
        dep=dependency,
        ver=f"2.{random.randint(1, 9)}.{random.randint(0, 20)}",
    )

    gold_sequence = GOLD_SEQUENCES[incident_type].copy()

    red_herrings = (
        generate_red_herrings(incident_type, primary_service)
        if difficulty == "hard"
        else []
    )

    topology_data = {
        "nodes": list(G.nodes()),
        "edges": [list(e) for e in G.edges()],
    }

    return {
        "incident_id": make_incident_id(difficulty, index),
        "difficulty": difficulty,
        "incident_type": incident_type,
        "alert_payload": alert_payload,
        "service_topology": topology_data,
        "logs": logs,
        "metrics": metrics,
        "gold_root_cause": root_cause,
        "gold_triggered_by": dependency if incident_type in (
            "cascade_failure", "network_partition", "dependency_timeout"
        ) else primary_service,
        "gold_affected_chain": affected_chain,
        "gold_action_sequence": gold_sequence,
        "red_herrings": red_herrings,
    }


def generate_dataset() -> List[Dict[str, Any]]:
    """Generate the full 90-incident dataset."""
    incidents = []

    for difficulty in ("easy", "medium", "hard"):
        for i in range(30):
            incident = generate_incident(difficulty, i)
            incidents.append(incident)
            print(
                f"  Generated {incident['incident_id']} "
                f"({incident['incident_type']}, {incident['difficulty']})"
            )

    return incidents


def validate_dataset(incidents: List[Dict[str, Any]]) -> bool:
    """Validate the generated dataset against expected schema."""
    assert len(incidents) == 90, f"Expected 90 incidents, got {len(incidents)}"

    difficulties = {"easy": 0, "medium": 0, "hard": 0}
    for inc in incidents:
        difficulties[inc["difficulty"]] += 1
        assert inc["incident_type"] in INCIDENT_TYPES
        assert inc["alert_payload"]["severity"] in ("P1", "P2", "P3", "P4")
        assert len(inc["logs"]) == 50
        assert len(inc["gold_action_sequence"]) > 0
        assert len(inc["service_topology"]["nodes"]) >= 3

        if inc["difficulty"] == "hard":
            assert len(inc["red_herrings"]) > 0

    assert difficulties == {"easy": 30, "medium": 30, "hard": 30}
    print("\n✅ Dataset validation passed!")
    return True


def main():
    """Generate and save the incident dataset."""
    print("🔧 SRE-Bench Dataset Generator")
    print("=" * 50)

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"\nGenerating 90 incidents (30 per difficulty)...\n")
    incidents = generate_dataset()

    print(f"\nValidating dataset...")
    validate_dataset(incidents)

    output_path = data_dir / "incidents.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(incidents, f, indent=2, ensure_ascii=False)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n📁 Saved to {output_path} ({size_kb:.1f} KB)")
    print(f"📊 Total incidents: {len(incidents)}")
    print(f"   Easy:   30 | Medium: 30 | Hard: 30")
    print(f"\n🎉 Dataset generation complete!")


if __name__ == "__main__":
    main()
