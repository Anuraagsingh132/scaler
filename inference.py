"""
inference.py — SecureAI-Guard baseline inference script.

Reads environment variables:
    API_BASE_URL  : base URL of the SecureAI-Guard environment server
    MODEL_NAME    : OpenAI-compatible model name for the LLM agent
    HF_TOKEN      : HuggingFace token (optional, passed to model calls)

Logging format (required by OpenEnv):
    [START] ...episode metadata...
    [STEP]  ...per-step data...
    [END]   ...episode summary...

Usage:
    export API_BASE_URL=http://localhost:7860
    export MODEL_NAME=gpt-3.5-turbo
    export HF_TOKEN=hf_...
    python inference.py
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

# ---------------------------------------------------------------------------
# Logging setup — plain stdout so automated validators can parse the tags
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

TASKS = ["basic_security", "trust_management", "adversarial_drift"]
EPISODES_PER_TASK = int(os.environ.get("EPISODES_PER_TASK", "1"))
SEED_BASE = int(os.environ.get("SEED_BASE", "42"))


# ---------------------------------------------------------------------------
# OpenAI-compatible LLM client
# ---------------------------------------------------------------------------
def call_llm(prompt: str, system: str = "") -> str:
    """
    Call an OpenAI-compatible endpoint.
    Falls back to a deterministic rule-based decision when the endpoint
    is unavailable (so the script is always runnable end-to-end).
    """
    openai_base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if openai_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_key, base_url=openai_base)
            messages = []
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
        except Exception as exc:
            logger.warning("LLM call failed (%s). Using rule-based fallback.", exc)

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
# Environment helpers
# ---------------------------------------------------------------------------
def env_reset(task_id: str, seed: int) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/reset"
    resp = requests.post(url, json={"task_id": task_id, "seed": seed}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/step"
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
# Main inference loop
# ---------------------------------------------------------------------------
def run_episode(task_id: str, seed: int, episode_num: int) -> Dict[str, Any]:
    reset_data = env_reset(task_id, seed)
    obs = reset_data["observation"]

    episode_summary: Dict[str, Any] = {
        "task_id": task_id,
        "seed": seed,
        "episode": episode_num,
        "steps": [],
        "total_reward": 0.0,
        "final_score": None,
        "grade": None,
    }

    logger.info(
        "[START] task=%s episode=%d seed=%d model=%s api=%s",
        task_id,
        episode_num,
        seed,
        MODEL_NAME,
        API_BASE_URL,
    )

    step_num = 0
    done = False

    while not done:
        step_num += 1

        # Build prompt and get action from LLM
        prompt = build_prompt(obs)
        llm_output = call_llm(prompt, system=SYSTEM_PROMPT)
        action = parse_action(llm_output)

        # Step environment
        step_data = env_step(action)
        reward_val = step_data["reward"]["value"]
        episode_summary["total_reward"] += reward_val
        done = step_data["done"]

        step_log = {
            "step": step_num,
            "channel": obs["channel"],
            "sender": obs["sender"],
            "decision": action["decision"],
            "confidence": action["confidence"],
            "reward": reward_val,
            "user_trust": step_data["state"]["user_trust"],
            "system_fatigue": step_data["state"]["system_fatigue"],
            "threat_type": step_data["info"].get("threat_type", "unknown"),
            "done": done,
        }
        episode_summary["steps"].append(step_log)

        logger.info(
            "[STEP] step=%d decision=%s confidence=%.2f reward=%.4f "
            "trust=%.1f fatigue=%.1f threat=%s",
            step_num,
            action["decision"],
            action["confidence"],
            reward_val,
            step_data["state"]["user_trust"],
            step_data["state"]["system_fatigue"],
            step_data["info"].get("threat_type", "unknown"),
        )

        # Advance observation
        obs = step_data["observation"]

        # Retrieve grade if episode ended
        if done and "grade" in step_data:
            grade_data = step_data["grade"]
            episode_summary["final_score"] = grade_data.get("score")
            episode_summary["grade"] = grade_data.get("grade")

    episode_summary["total_reward"] = round(episode_summary["total_reward"], 4)

    logger.info(
        "[END] task=%s episode=%d steps=%d total_reward=%.4f score=%s grade=%s",
        task_id,
        episode_num,
        step_num,
        episode_summary["total_reward"],
        episode_summary.get("final_score"),
        episode_summary.get("grade"),
    )

    return episode_summary


def main():
    logger.info("=== SecureAI-Guard Inference ===")
    logger.info("API_BASE_URL : %s", API_BASE_URL)
    logger.info("MODEL_NAME   : %s", MODEL_NAME)
    logger.info("HF_TOKEN     : %s", "set" if HF_TOKEN else "not set")

    all_results = []
    global_episode = 0

    for task_id in TASKS:
        for ep_idx in range(EPISODES_PER_TASK):
            global_episode += 1
            seed = SEED_BASE + global_episode
            try:
                summary = run_episode(task_id, seed, global_episode)
                all_results.append(summary)
            except Exception as exc:
                logger.error("Episode failed: task=%s episode=%d error=%s", task_id, global_episode, exc)

    # Aggregate summary
    if all_results:
        avg_reward = sum(r["total_reward"] for r in all_results) / len(all_results)
        scored = [r for r in all_results if r["final_score"] is not None]
        avg_score = sum(r["final_score"] for r in scored) / len(scored) if scored else None
        logger.info(
            "=== SUMMARY === episodes=%d avg_reward=%.4f avg_score=%s",
            len(all_results),
            avg_reward,
            f"{avg_score:.4f}" if avg_score is not None else "n/a",
        )


if __name__ == "__main__":
    main()
