"""
SRE-Bench: LLM Agent Baseline (inference.py)
OpenAI-compatible client using HuggingFace Router for SRE incident response.

Usage:
    python inference.py --task task1 --seed 42 --url http://localhost:7860
    python inference.py --task task3 --seed 100 --url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ─── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:7860"
DEFAULT_MODEL = os.environ.get("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_RETRIES = 5
RETRY_DELAYS = [1, 2, 4, 8, 16]  # Exponential backoff


# ─── Prompts ────────────────────────────────────────────────────────────────────

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
4. Follow the order: acknowledge → diagnose → remediate → verify → resolve

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
Follow the workflow: acknowledge → diagnose → remediate → verify → resolve.
"""

    return prompt


def parse_action_response(response_text: str) -> Dict[str, str]:
    """Parse the [START]...[END] format from LLM response."""
    # Find content between [START] and [END]
    pattern = r"\[START\](.*?)\[END\]"
    match = re.search(pattern, response_text, re.DOTALL)

    if not match:
        # Try to parse without markers
        return _fallback_parse(response_text)

    content = match.group(1).strip()
    result = {}

    for line in content.split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()

    return result


def _fallback_parse(text: str) -> Dict[str, str]:
    """Fallback parser when [START][END] markers are missing."""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip().lstrip("- ")
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            result[key] = value.strip()

    return result


# ─── LLM Client ─────────────────────────────────────────────────────────────────


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

    def reset_episode(self, task: str, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the environment for a new episode."""
        payload = {"task": task}
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

        # Cache check
        cache_key = prompt[:200]
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        for attempt, delay in enumerate(RETRY_DELAYS):
            try:
                response = self._call_hf_router(prompt)
                self._response_cache[cache_key] = response
                return response
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠️  LLM call failed (attempt {attempt + 1}): {e}")
                    time.sleep(delay)
                else:
                    print(f"  ❌ LLM call failed after {MAX_RETRIES} attempts")
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
            print("  ⚠️  openai package not installed. Using rule-based agent.")
            return self._rule_based_response(prompt)

    def _rule_based_response(self, prompt: str) -> str:
        """
        Rule-based fallback agent for when LLM is unavailable.
        Follows a sensible incident response workflow.
        """
        # Parse current state from prompt
        actions_taken = []
        if "Actions taken:" in prompt:
            actions_line = prompt.split("Actions taken:")[1].split("\n")[0].strip()
            if actions_line != "None":
                actions_taken = [a.strip() for a in actions_line.split(",")]

        # Extract service name
        service = "unknown"
        if "Service:" in prompt:
            service = prompt.split("Service:")[1].split("\n")[0].strip()

        # Task 1: Classification
        if "Alert Classification" in prompt:
            return self._classify_incident(prompt, service)

        # Task 2/3: Follow diagnosis → remediation workflow
        return self._next_action(actions_taken, prompt, service)

    def _classify_incident(self, prompt: str, service: str) -> str:
        """Rule-based incident classification."""
        prompt_lower = prompt.lower()

        # Detect incident type from metrics and logs
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

        # Check severity from alert title
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

        # Follow the optimal workflow
        if "acknowledge_alert" not in action_names:
            action = "acknowledge_alert"
        elif "inspect_logs" not in action_names:
            action = "inspect_logs"
        elif "check_metrics" not in action_names:
            action = "check_metrics"
        elif "check_service" not in action_names:
            action = "check_service"
        elif not any(a in action_names for a in ("restart_service", "scale_up", "rollback_deploy")):
            # Choose remediation based on incident indicators
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
        if verbose:
            print(f"\n{'='*60}")
            print(f"🚀 SRE-Bench Agent — Task: {task} | Seed: {seed}")
            print(f"{'='*60}")

        # Reset environment
        observation = self.reset_episode(task, seed)
        if verbose:
            alert = observation.get("alert_payload", {})
            print(f"\n📋 Incident: {observation.get('incident_id')}")
            print(f"   Alert: {alert.get('title')}")
            print(f"   Service: {alert.get('service')}")
            print(f"   Severity: {alert.get('severity')}")

        total_reward = 0.0
        step_num = 0
        done = False
        final_info = {}

        while not done and step_num < 30:
            # Build prompt and get response
            prompt = build_observation_prompt(observation, task)
            llm_response = self.get_llm_response(prompt)

            # Parse action from response
            parsed = parse_action_response(llm_response)

            if task == "task1":
                # Task 1: Submit classification as final step
                if verbose:
                    print(f"\n📝 Classification:")
                    for k, v in parsed.items():
                        print(f"   {k}: {v}")

                # For task1, we submit inspect_logs as the required action
                action = {
                    "action_type": "inspect_logs",
                    "target_service": observation.get("alert_payload", {}).get("service"),
                    "parameters": parsed,
                }
            else:
                # Task 2/3: Execute the parsed action
                action_type = parsed.get("action_type", "inspect_logs")
                target = parsed.get("target_service", observation.get("alert_payload", {}).get("service"))

                action = {
                    "action_type": action_type,
                    "target_service": target,
                }

            # Execute step
            reward_response = self.execute_step(action)
            step_num += 1
            reward_val = reward_response.get("value", 0.0)
            total_reward = reward_response.get("cumulative", 0.0)
            done = reward_response.get("done", False)
            info = reward_response.get("info", {})

            if verbose:
                status = "✅" if reward_val >= 0 else "❌"
                print(f"\n  Step {step_num}: {action.get('action_type')} → {status} reward={reward_val:+.4f} (cum={total_reward:.4f})")

            # Update observation with any new info
            if "action_result" in info:
                observation["action_history"] = observation.get("action_history", []) + [
                    action.get("action_type", "")
                ]
                observation["step_number"] = step_num

            final_info = info

        # Print final results
        final_score = final_info.get("final_score", total_reward)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🏁 Episode Complete")
            print(f"   Steps Used: {step_num}")
            print(f"   Cumulative Reward: {total_reward:.4f}")
            print(f"   Final Score: {final_score:.4f}")
            print(f"{'='*60}")

        # Print in submission format
        print(f"\n[START]")
        if task == "task1":
            parsed_final = parse_action_response(llm_response)
            print(f"incident_type: {parsed_final.get('incident_type', 'unknown')}")
            print(f"severity: {parsed_final.get('severity', 'P2')}")
            print(f"primary_fault_service: {parsed_final.get('primary_fault_service', 'unknown')}")
        else:
            print(f"task: {task}")
            print(f"final_score: {final_score:.4f}")
            print(f"steps_used: {step_num}")
            print(f"cumulative_reward: {total_reward:.4f}")
        print(f"[END]")

        return {
            "task": task,
            "seed": seed,
            "steps_used": step_num,
            "cumulative_reward": total_reward,
            "final_score": final_score,
            "final_info": final_info,
        }


# ─── CLI Entry Point ───────────────────────────────────────────────────────────


def main():
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
        help="Suppress verbose output",
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

        print(f"\n{'='*60}")
        print(f"📊 FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r['task']}: score={r['final_score']:.4f} steps={r['steps_used']}")
    else:
        agent.run_episode(
            task=args.task,
            seed=args.seed,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
