"""
SRE-Bench: LLM Agent Baseline (inference.py)
OpenAI-compatible client using HuggingFace Router for SRE incident response.

Usage:
    python inference.py --task task1 --seed 42 --url http://localhost:7860
    python inference.py --task task3 --seed 100 --url http://localhost:7860
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import httpx


# Configuration
BASE_API_URL = "http://localhost:7860"
ENV_NAME = "sre-bench"
DEFAULT_API_URL = os.environ.get("API_BASE_URL", os.environ.get("SRE_BENCH_API_URL", BASE_API_URL))
DEFAULT_MODEL = os.environ.get("MODEL_NAME", os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"))
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN", ""))
MAX_RETRIES = 5
RETRY_DELAYS = [1, 2, 4, 8, 16]


SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) handling on-call incidents.
You must analyze alerts, diagnose root causes, and execute remediation actions.

Available actions:
- inspect_logs: Examine service logs for errors
- check_metrics: View CPU, memory, latency, error rate
- check_service: Check service health status
- restart_service: Restart a service (requires inspect_logs first)
- scale_up: Increase service replicas (for capacity issues)
- rollback_deploy: Revert to previous deployment (for deploy/config issues)
- verify_endpoint: Verify fix was effective (requires remediation first)
- resolve: Close the incident (requires verification first)
- check_topology: View service dependency graph
- acknowledge_alert: Acknowledge the alert

Rules:
1. Always inspect_logs before restart_service
2. Always apply remediation before verify_endpoint
3. Always verify_endpoint before resolve
4. Follow the order: acknowledge -> diagnose -> remediate -> verify -> resolve

Respond with your action in this exact format:
[START]
action_type: <action_name>
target_service: <service_name>
[END]

For final classification (Task 1), respond with:
[START]
incident_type: <type>
severity: <P1|P2|P3|P4>
primary_fault_service: <service_name>
[END]
"""


def _coerce_float(value: Any, fallback: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _sanitize_token(value: Any) -> str:
    """Convert arbitrary values into single-token strings for structured logs."""
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return "null"
    return re.sub(r"\s+", "_", text)


def _format_action_token(action_type: str, target_service: Optional[str]) -> str:
    """Format action token for [STEP] output."""
    action = _sanitize_token(action_type)
    if target_service:
        target = _sanitize_token(target_service).replace("'", "")
        return f"{action}('{target}')"
    return f"{action}()"


def _extract_error_token(info: Dict[str, Any]) -> str:
    """Return a compact error token for [STEP] output."""
    if not isinstance(info, dict):
        return "null"

    if info.get("error"):
        return _sanitize_token(info["error"])

    action_result = info.get("action_result")
    if isinstance(action_result, dict):
        if action_result.get("status") == "precondition_failed":
            value = action_result.get("error") or action_result.get("reason") or "precondition_failed"
            return _sanitize_token(value)

    if info.get("precondition_failed"):
        return "precondition_failed"

    return "null"


def build_observation_prompt(observation: Dict[str, Any], task: str) -> str:
    """Build a structured prompt from the SREObservation."""
    alert = observation.get("alert_payload", {})
    metrics = observation.get("metrics", {})
    logs = observation.get("logs", [])
    topology = observation.get("service_topology", {})
    history = observation.get("action_history", [])
    step = observation.get("step_number", 0)

    prompt = f"""## Current Incident: {observation.get('incident_id', 'N/A')}

### Alert
- Title: {alert.get('title', 'N/A')}
- Service: {alert.get('service', 'N/A')}
- Severity: {alert.get('severity', 'N/A')}
- Time: {alert.get('timestamp', 'N/A')}

### Metrics
- CPU: {metrics.get('cpu', 'N/A')}%
- Memory: {metrics.get('memory', 'N/A')}%
- Latency: {metrics.get('latency_ms', 'N/A')}ms
- Error Rate: {metrics.get('error_rate', 'N/A')}%

### Recent Logs (last 10)
"""
    for log in logs[-10:]:
        prompt += f"  {log}\n"

    prompt += f"""
### Service Topology
- Nodes: {', '.join(topology.get('nodes', [])[:8])}
- Edges: {len(topology.get('edges', []))} connections

### Episode State
- Step: {step}
- Actions taken: {', '.join(history) if history else 'None'}
"""

    if task == "task1":
        prompt += """
### Your Task (Alert Classification)
Classify this incident. Provide:
- incident_type (cpu_spike, memory_leak, network_partition, deployment_failure, db_overload, cascade_failure, config_error, dependency_timeout)
- severity (P1, P2, P3, P4)
- primary_fault_service (the service causing the issue)
"""
    elif task == "task2":
        prompt += """
### Your Task (Root Cause Analysis)
Determine the root cause. Choose your next diagnostic action to investigate.
"""
    else:
        prompt += """
### Your Task (Full Remediation)
Diagnose and remediate this incident. Choose your next action carefully.
Follow the workflow: acknowledge -> diagnose -> remediate -> verify -> resolve.
"""

    return prompt


def parse_action_response(response_text: str) -> Dict[str, str]:
    """Parse the [START]...[END] format from LLM response."""
    pattern = r"\[START\](.*?)\[END\]"
    match = re.search(pattern, response_text, re.DOTALL)

    if not match:
        return _fallback_parse(response_text)

    content = match.group(1).strip()
    result: Dict[str, str] = {}

    for line in content.split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()

    return result


def _fallback_parse(text: str) -> Dict[str, str]:
    """Fallback parser when [START][END] markers are missing."""
    result: Dict[str, str] = {}
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("- ")
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            result[key] = value.strip()

    return result


class SREAgent:
    """LLM-powered SRE agent using HuggingFace Router (OpenAI-compatible)."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        model: str = DEFAULT_MODEL,
        hf_token: str = HF_TOKEN,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.hf_token = hf_token
        self.client = httpx.Client(timeout=30.0)
        self._response_cache: Dict[str, str] = {}
        self.verbose = True

    def _log(self, message: str) -> None:
        """Write optional human-readable logs to stderr."""
        if self.verbose:
            print(message, file=sys.stderr)

    def reset_episode(self, task: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        payload: Dict[str, Any] = {"task": task}
        if seed is not None:
            payload["seed"] = seed

        resp = self.client.post(f"{self.api_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def execute_step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a step in the environment."""
        resp = self.client.post(f"{self.api_url}/step", json=action)
        resp.raise_for_status()
        return resp.json()

    def get_llm_response(self, prompt: str) -> str:
        """
        Get a response from the LLM via HuggingFace Router.
        Falls back to rule-based if no HF token is set.
        """
        if not self.hf_token:
            return self._rule_based_response(prompt)

        cache_key = prompt[:200]
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                response = self._call_hf_router(prompt)
                self._response_cache[cache_key] = response
                return response
            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    self._log(f"[WARN] LLM call failed (attempt {attempt + 1}): {exc}")
                    time.sleep(delay)
                else:
                    self._log(f"[ERROR] LLM call failed after {MAX_RETRIES} attempts")
                    return self._rule_based_response(prompt)

        return self._rule_based_response(prompt)

    def _call_hf_router(self, prompt: str) -> str:
        """Call HuggingFace Router (OpenAI-compatible endpoint)."""
        try:
            from openai import OpenAI

            client = OpenAI(
                base_url="https://api-inference.huggingface.co/v1/",
                api_key=self.hf_token,
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            self._log("[WARN] openai package not installed. Using rule-based agent.")
            return self._rule_based_response(prompt)

    def _rule_based_response(self, prompt: str) -> str:
        """Rule-based fallback agent for when LLM is unavailable."""
        actions_taken: List[str] = []
        if "Actions taken:" in prompt:
            actions_line = prompt.split("Actions taken:")[1].split("\n")[0].strip()
            if actions_line != "None":
                actions_taken = [a.strip() for a in actions_line.split(",")]

        service = "unknown"
        if "Service:" in prompt:
            service = prompt.split("Service:")[1].split("\n")[0].strip()

        if "Alert Classification" in prompt:
            return self._classify_incident(prompt, service)

        return self._next_action(actions_taken, prompt, service)

    def _classify_incident(self, prompt: str, service: str) -> str:
        """Rule-based incident classification."""
        prompt_lower = prompt.lower()

        incident_type = "cpu_spike"
        severity = "P2"

        if "memory" in prompt_lower and ("88" in prompt or "9" in prompt.split("memory")[1][:20]):
            incident_type = "memory_leak"
        elif "latency" in prompt_lower and any(x in prompt for x in ["5000", "10000", "30000"]):
            incident_type = "network_partition" if "partition" in prompt_lower else "dependency_timeout"
            severity = "P1"
        elif "deploy" in prompt_lower or "rollback" in prompt_lower:
            incident_type = "deployment_failure"
            severity = "P1"
        elif "error_rate" in prompt_lower and any(x in prompt for x in ["40", "50", "60", "70", "80", "90"]):
            incident_type = "deployment_failure"
            severity = "P1"
        elif "cpu" in prompt_lower and ("85" in prompt or "9" in prompt.split("cpu")[1][:20]):
            incident_type = "cpu_spike"
        elif "cascade" in prompt_lower:
            incident_type = "cascade_failure"
            severity = "P1"
        elif "config" in prompt_lower:
            incident_type = "config_error"

        if "[P1]" in prompt:
            severity = "P1"
        elif "[P3]" in prompt:
            severity = "P3"
        elif "[P4]" in prompt:
            severity = "P4"

        return f"""[START]
incident_type: {incident_type}
severity: {severity}
primary_fault_service: {service}
[END]"""

    def _next_action(self, actions_taken: List[str], prompt: str, service: str) -> str:
        """Determine the next action based on workflow state."""
        action_names = [a.split(":")[0] for a in actions_taken]

        if "acknowledge_alert" not in action_names:
            action = "acknowledge_alert"
        elif "inspect_logs" not in action_names:
            action = "inspect_logs"
        elif "check_metrics" not in action_names:
            action = "check_metrics"
        elif "check_service" not in action_names:
            action = "check_service"
        elif not any(a in action_names for a in ("restart_service", "scale_up", "rollback_deploy")):
            if "deploy" in prompt.lower() or "config" in prompt.lower():
                action = "rollback_deploy"
            elif "capacity" in prompt.lower() or "overload" in prompt.lower():
                action = "scale_up"
            else:
                action = "restart_service"
        elif "verify_endpoint" not in action_names:
            action = "verify_endpoint"
        else:
            action = "resolve"

        return f"""[START]
action_type: {action}
target_service: {service}
[END]"""

    def run_episode(
        self,
        task: str,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run a complete episode and return results."""
        self.verbose = verbose

        observation = self.reset_episode(task, seed)
        print(f"[START] task={task} env={ENV_NAME} model={self.model}")

        step_num = 0
        done = False
        final_info: Dict[str, Any] = {}
        step_rewards: List[float] = []

        while not done and step_num < 30:
            prompt = build_observation_prompt(observation, task)
            llm_response = self.get_llm_response(prompt)
            parsed = parse_action_response(llm_response)

            if task == "task1":
                action = {
                    "action_type": "inspect_logs",
                    "target_service": observation.get("alert_payload", {}).get("service"),
                    "parameters": parsed,
                }
            else:
                action_type = parsed.get("action_type", "inspect_logs")
                target = parsed.get("target_service", observation.get("alert_payload", {}).get("service"))
                action = {"action_type": action_type, "target_service": target}

            reward_response = self.execute_step(action)
            step_num += 1
            reward_val = max(0.0, _coerce_float(reward_response.get("value", 0.0), 0.0))
            step_rewards.append(reward_val)

            done = bool(reward_response.get("done", False))
            info = reward_response.get("info", {})
            error_token = _extract_error_token(info)
            action_token = _format_action_token(action.get("action_type", "unknown"), action.get("target_service"))

            print(
                f"[STEP] step={step_num} action={action_token} reward={reward_val:.2f} "
                f"done={str(done).lower()} error={error_token}"
            )

            if "action_result" in info:
                observation["action_history"] = observation.get("action_history", []) + [action.get("action_type", "")]
                observation["step_number"] = step_num

            final_info = info

        computed_score = sum(step_rewards) / max(1, len(step_rewards))
        raw_final_score = _coerce_float(final_info.get("final_score", computed_score), computed_score)
        output_score = max(0.0, raw_final_score)
        success = output_score > 0.10
        rewards_csv = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"

        print(
            f"[END] success={str(success).lower()} steps={step_num} "
            f"score={output_score:.2f} rewards={rewards_csv}"
        )

        return {
            "task": task,
            "seed": seed,
            "steps_used": step_num,
            "cumulative_reward": sum(step_rewards),
            "final_score": output_score,
            "raw_final_score": raw_final_score,
            "step_rewards": step_rewards,
            "final_info": final_info,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SRE-Bench LLM Agent Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task1",
        choices=["task1", "task2", "task3"],
        help="Task difficulty level (default: task1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for incident selection (default: 42)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help=f"API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress non-structured logs",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all three tasks sequentially",
    )

    args = parser.parse_args()

    agent = SREAgent(
        api_url=args.url,
        model=args.model,
        hf_token=HF_TOKEN,
    )

    if args.all_tasks:
        results = []
        for task in ["task1", "task2", "task3"]:
            result = agent.run_episode(
                task=task,
                seed=args.seed,
                verbose=not args.quiet,
            )
            results.append(result)

        if not args.quiet:
            print("\n" + "=" * 60, file=sys.stderr)
            print("FINAL RESULTS SUMMARY", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            for result in results:
                print(
                    f"  {result['task']}: score={result['final_score']:.4f} steps={result['steps_used']}",
                    file=sys.stderr,
                )
    else:
        agent.run_episode(
            task=args.task,
            seed=args.seed,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
