"""
inference.py — SecureAI-Guard baseline inference script (OpenEnv compliant).

Environment variables:
    API_BASE_URL  : base URL for the OpenAI-compatible LLM endpoint
    MODEL_NAME    : model name passed to the LLM client
    HF_TOKEN      : API key for the LLM endpoint (also aliased as API_KEY)
    ENV_URL       : URL of the local SecureAI-Guard environment server
                    (default: http://localhost:7860)

Stdout logging format (required by OpenEnv — single lines, no extras):
    [START] task=<task_name> env=SecureAI-Guard model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Usage:
    export API_BASE_URL=https://api-inference.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3-8B-Instruct
    export HF_TOKEN=hf_...
    export ENV_URL=http://localhost:7860
    python inference.py
"""

import json
import os
import sys
from typing import Any, Dict, List

import requests

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
# LLM-only variables — NEVER used for the environment server
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or HF_TOKEN

# Environment server URL — completely separate from the LLM
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = ["basic_security", "trust_management", "adversarial_drift"]
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "1"))
SEED_BASE = int(os.environ.get("SEED_BASE", "42"))

BENCHMARK_NAME = "SecureAI-Guard"


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM client
# ---------------------------------------------------------------------------
def call_llm(prompt: str, system: str = "") -> str:
    """
    Call an OpenAI-compatible endpoint using API_BASE_URL + API_KEY.
    Falls back to a deterministic rule-based decision when the key is
    missing or the endpoint is unreachable.
    """
    if API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
            messages: list = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=256,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            pass  # fall through to rule-based

    # Rule-based fallback — deterministic, no network needed
    return _rule_based_decision(prompt)


def _rule_based_decision(prompt: str) -> str:
    """Deterministic fallback agent using risk keywords."""
    p = prompt.lower()

    HIGH_RISK = [
        "click here", "verify your account", "suspended", "claim now",
        "download", "install", "wire transfer", "credentials", "emergency",
        "send money", "customs fee", "stranded",
    ]
    MEDIUM_RISK = [
        "http://", "https://", "link", "verify", "update", "confirm",
        "invoice", "document", "offer", "free", "loan", "prize",
    ]

    high = sum(1 for kw in HIGH_RISK if kw in p)
    medium = sum(1 for kw in MEDIUM_RISK if kw in p)

    if high >= 2 or (high >= 1 and medium >= 2):
        decision, confidence, reasoning = (
            "block",
            0.92,
            (
                f"Detected {high} high-risk and {medium} medium-risk indicators. "
                "Blocking to protect user from phishing/social-engineering attempt."
            ),
        )
    elif high == 1 or medium >= 3:
        decision, confidence, reasoning = (
            "warn",
            0.75,
            (
                f"Detected {high} high-risk and {medium} medium-risk indicators. "
                "Issuing warning while preserving user trust."
            ),
        )
    elif medium >= 1:
        decision, confidence, reasoning = (
            "investigate",
            0.60,
            (
                f"Detected {medium} medium-risk indicators. "
                "Flagging for investigation — insufficient evidence to block."
            ),
        )
    else:
        decision, confidence, reasoning = (
            "allow",
            0.88,
            "No risk indicators detected. Message appears legitimate and safe to allow.",
        )

    return json.dumps(
        {"decision": decision, "confidence": confidence, "reasoning": reasoning}
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an autonomous security assistant protecting a user's communications. "
    "Analyse the incoming message and decide how to handle it.\n\n"
    "Respond ONLY with valid JSON (no markdown fences):\n"
    '{"decision": "allow|block|warn|investigate", "confidence": 0.0-1.0, "reasoning": "..."}'
)


def build_prompt(obs: Dict[str, Any]) -> str:
    return (
        f"Channel     : {obs['channel']}\n"
        f"Sender      : {obs['sender']}\n"
        f"Content     : {obs['content']}\n"
        f"Risk Score  : {obs['hf_risk_score']:.2f}\n"
        f"User Trust  : {obs['user_trust']:.1f}/100\n"
        f"Sys Fatigue : {obs['system_fatigue']:.1f}/100\n"
        f"Threat Hist : {len(obs.get('threat_history', []))} recent events\n\n"
        "What is your security decision?"
    )


# ---------------------------------------------------------------------------
# Environment helpers — use ENV_URL (NOT API_BASE_URL)
# ---------------------------------------------------------------------------
def env_reset(task_id: str, seed: int) -> Dict[str, Any]:
    url = f"{ENV_URL}/reset"
    resp = requests.post(url, json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{ENV_URL}/step"
    resp = requests.post(url, json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_action(llm_output: str) -> Dict[str, Any]:
    """Parse LLM JSON output into an action dict."""
    # Strip markdown fences if present
    cleaned = llm_output.strip().strip("```json").strip("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: default to investigate
        data = {
            "decision": "investigate",
            "confidence": 0.5,
            "reasoning": "Unable to parse LLM output; defaulting to investigation.",
        }

    # Validate / clamp
    valid_decisions = {"allow", "block", "warn", "investigate"}
    if data.get("decision") not in valid_decisions:
        data["decision"] = "investigate"
    data["confidence"] = float(max(0.0, min(1.0, data.get("confidence", 0.5))))
    if not data.get("reasoning", "").strip():
        data["reasoning"] = "No reasoning provided."
    return data


# ---------------------------------------------------------------------------
# Strict OpenEnv stdout logging helpers
# ---------------------------------------------------------------------------
def _fmt_bool(v: bool) -> str:
    """Return lowercase 'true' or 'false'."""
    return "true" if v else "false"


def _log_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={model_name}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err_str = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={_fmt_bool(done)} error={err_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={_fmt_bool(success)} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def run_episode(task_id: str, seed: int) -> Dict[str, Any]:
    _log_start(task_id, MODEL_NAME)

    step_num = 0
    done = False
    all_rewards: List[float] = []
    final_score: float = 0.0
    success: bool = False
    obs: Dict[str, Any] | None = None

    try:
        reset_data = env_reset(task_id, seed)
        obs = reset_data["observation"]
    except Exception as exc:
        error_msg = str(exc).replace("\n", " ")
        _log_end(False, 0, 0.0, all_rewards)
        return {
            "task_id": task_id,
            "seed": seed,
            "steps": 0,
            "total_reward": 0.0,
            "final_score": 0.0,
            "success": False,
            "error": error_msg,
        }

    while not done:
        step_num += 1

        try:
            # Build prompt and get action from LLM
            prompt = build_prompt(obs)
            llm_output = call_llm(prompt, system=SYSTEM_PROMPT)
            action = parse_action(llm_output)

            # Step environment
            step_data = env_step(action)
            reward_val: float = step_data["reward"]["value"]
            done = step_data["done"]
            all_rewards.append(reward_val)

            _log_step(step_num, action["decision"], reward_val, done, None)

            # Advance observation to the NEW state
            obs = step_data["observation"]

            # Retrieve grade if episode ended
            if done and "grade" in step_data:
                grade_data = step_data["grade"]
                final_score = float(grade_data.get("score", 0.0))
                success = bool(grade_data.get("passed", False))

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ")
            _log_step(step_num, "error", 0.0, True, error_msg)
            done = True

    _log_end(success, step_num, final_score, all_rewards)

    return {
        "task_id": task_id,
        "seed": seed,
        "steps": step_num,
        "total_reward": round(sum(all_rewards), 4),
        "final_score": final_score,
        "success": success,
    }


def main():
    all_results: List[Dict[str, Any]] = []
    global_episode = 0

    for task_id in TASKS:
        for _ in range(EPISODES_PER_TASK):
            global_episode += 1
            seed = SEED_BASE + global_episode
            result = run_episode(task_id, seed)
            all_results.append(result)


if __name__ == "__main__":
    main()
