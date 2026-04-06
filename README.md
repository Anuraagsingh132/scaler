# 🛡️ SecureAI-Guard: Stateful POMDP for Autonomous Digital Defense

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.ai)
[![HuggingFace Spaces](https://img.shields.io/badge/HF%20Spaces-ready-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

SecureAI-Guard is a **production-grade reinforcement learning environment** that simulates an autonomous personal security assistant protecting users across SMS, Email, and Web channels. Agents must make real-time decisions to block phishing, malware, social engineering, and spam while preserving user trust and avoiding alert fatigue.

This environment is fully compliant with the [OpenEnv specification](https://openenv.ai) and is designed for both RL training and zero-shot LLM inference evaluation.

---

## 🎯 Key Features

| Feature | Description |
|---|---|
| **Stateful POMDP** | Hidden state (user trust, system fatigue) affects observations and termination |
| **Adversarial Drift** | L3 adversary adapts its attack tactics mid-episode based on agent behaviour |
| **Dense Rewards** | Multi-component reward shaped across every step — no sparse end-of-episode signals |
| **Deterministic** | Fully reproducible with seed control |
| **OpenEnv Compliant** | Full `reset()`, `step()`, `state()` API + valid `openenv.yaml` |
| **HF Integration** | Optional DistilBERT risk scorer with keyword fallback |
| **DPO Flywheel** | Preference pairs logged every step for LLM alignment |
| **SOC Dashboard** | Real-time Gradio monitoring interface |

---

## 🏗️ Project Structure

```
SecureAI-Guard/
├── app.py                   # FastAPI environment server (port 7860)
├── ui.py                    # Gradio SOC dashboard (port 7861)
├── inference.py             # ⭐ Required baseline inference script
├── dqn_baseline.py          # Dueling DQN training script
├── openenv.yaml             # OpenEnv manifest
├── requirements.txt
├── Dockerfile
├── schema/
│   └── models.py            # Pydantic v2 typed models
├── env/
│   ├── core.py              # Threat generation + reward logic
│   └── engine.py            # reset() / step() / state() engine
├── tasks/
│   └── registry.py          # Three tasks (L1, L2, L3)
├── graders/
│   └── security_grader.py   # Deterministic grader → score ∈ [0.0, 1.0]
└── utils/
    └── hf_integration.py    # HuggingFace risk scorer + fallback
```

---

## 🚀 Quick Start

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Start the Environment Server

```bash
python app.py
# FastAPI running at http://localhost:7860
```

### 2. Run the Baseline Inference Script

```bash
export API_BASE_URL=http://localhost:7860
export MODEL_NAME=gpt-3.5-turbo          # any OpenAI-compatible model
export OPENAI_API_KEY=sk-...             # optional; uses rule-based fallback if absent
export HF_TOKEN=hf_...                   # optional
python inference.py
```

### 3. Launch the SOC Dashboard (optional)

```bash
python ui.py
# Gradio dashboard at http://localhost:7861
```

### 4. Train the DQN Agent (optional)

```bash
python dqn_baseline.py --episodes 500 --task basic_security
```

---

## 📡 API Reference

All endpoints accept and return JSON. The server runs on port **7860**.

### `POST /reset`

Reset the environment and return the first observation.

**Request:**
```json
{
  "task_id": "basic_security",
  "seed": 42
}
```

**Response:**
```json
{
  "observation": { ... },
  "state": { ... },
  "task_id": "basic_security"
}
```

### `POST /step`

Execute one action and advance the environment.

**Request:**
```json
{
  "action": {
    "decision": "block",
    "confidence": 0.92,
    "reasoning": "High-risk phishing link detected from unknown sender."
  }
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": {
    "value": 0.48,
    "components": {
      "security": 1.0,
      "user_friction": 0.0,
      "delay": 0.0,
      "reasoning_quality": 0.6,
      "total": 0.56
    },
    "explanation": "security=1.00, friction=0.00, delay=0.00, reasoning=0.60"
  },
  "done": false,
  "info": { "threat_type": "phishing", "step": 3 },
  "state": { ... }
}
```

### `GET /state`

Return the current environment state without advancing.

### `GET /tasks`

List all available tasks.

### `GET /health`

Health check — returns `{"status": "healthy"}`.

---

## 🎭 Observation Space

| Field | Type | Range | Description |
|---|---|---|---|
| `event_id` | string | — | Unique UUID per event |
| `channel` | enum | sms, email, web | Message delivery channel |
| `sender` | string | — | Sender identifier |
| `content` | string | — | Raw message text |
| `timestamp` | float | unix ts | Arrival time |
| `hf_risk_score` | float | [0.0, 1.0] | HuggingFace classifier risk signal |
| `user_trust` | float | [0.0, 100.0] | Running user trust level |
| `system_fatigue` | float | [0.0, 100.0] | Alert fatigue accumulator |
| `threat_history` | list | — | Last 5 events for context |
| `metadata` | object | — | Step, difficulty, event type |

---

## 🎮 Action Space

| Field | Type | Description |
|---|---|---|
| `decision` | enum | `allow` / `block` / `warn` / `investigate` |
| `confidence` | float [0–1] | Agent's confidence in its decision |
| `reasoning` | string | Human-readable explanation (required, non-empty) |

---

## 🏆 Task Descriptions

### L1 — Basic Security Screening (`basic_security`)
- **Max steps:** 50 | **Success threshold:** 0.80
- Phishing and spam only. No adversarial drift.
- Ideal entry point. Clear-cut threats with high reward signal.

### L2 — Trust Management Challenge (`trust_management`)
- **Max steps:** 75 | **Success threshold:** 0.75
- All threat types active. False positives incur 1.5× trust penalty.
- Agents must learn to tolerate ambiguity without over-blocking.

### L3 — Advanced Adversary Challenge (`adversarial_drift`)
- **Max steps:** 100 | **Success threshold:** 0.70
- Adaptive attacker: after step 20, switches tactics based on agent blocking rate.
- Agents that over-block phishing will face a surge of social-engineering instead.

---

## 💰 Reward Design

### Formula

```
R_step = (0.5·security + 0.3·user_friction + 0.1·delay + 0.1·reasoning) × (0.7 + 0.3·confidence)
```

### Components

| Component | Range | Calculation |
|---|---|---|
| `security` | [−1.0, +1.0] | +1.0 correct block; −1.0 missed threat; +0.5 safe allow; −0.8 false positive |
| `user_friction` | [−0.5, 0.0] | −0.2 per warning; −0.1 per investigate; −0.5 for false-positive block |
| `delay` | [−0.1, 0.0] | −0.1 for investigate actions |
| `reasoning_quality` | [0.0, 1.0] | Keyword match against threat-specific vocabulary |

### Why Dense?

Every step yields a non-zero reward signal, enabling stable gradient estimates for both RL and LLM policy optimisation. Partial credit is given via the confidence scaling factor — an uncertain correct answer scores higher than a certain wrong one.

---

## 📊 Grading

The `SecurityGrader` produces a deterministic score in **[0.0, 1.0]**:

```
score = 0.40 × security_efficiency
      + 0.30 × user_retention
      + 0.20 × precision
      + 0.10 × reasoning_quality
```

| Metric | Formula |
|---|---|
| `security_efficiency` | blocked_threats / total_threats |
| `user_retention` | final_user_trust / 100 |
| `precision` | 1 − false_positive_rate |
| `reasoning_quality` | avg(reasoning component across episode) |

### Letter Grades

| Score | Grade |
|---|---|
| ≥ 0.90 | A+ |
| ≥ 0.80 | A |
| ≥ 0.70 | B |
| ≥ 0.60 | C |
| ≥ 0.50 | D |
| < 0.50 | F |

---

## 🔚 Episode Termination

An episode ends when any of the following conditions is met:

1. **`user_trust ≤ 0`** — User has uninstalled the assistant due to too many false positives.
2. **`system_fatigue ≥ 100`** — User ignores all alerts (warn overload).
3. **`step_count ≥ max_steps`** — Episode length limit reached.

---

## 📋 Inference Script

`inference.py` is the required OpenEnv baseline script. It:

- Reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables
- Uses the OpenAI client for LLM inference (with deterministic keyword fallback when no API key is set)
- Runs all three tasks sequentially
- Produces reproducible results with `SEED_BASE` control
- Logs in the required format:

```
[START] task=basic_security episode=1 seed=43 model=gpt-3.5-turbo api=http://localhost:7860
[STEP]  step=1 decision=block confidence=0.92 reward=0.4830 trust=101.0 fatigue=0.0 threat=phishing
[STEP]  step=2 decision=allow confidence=0.88 reward=0.3150 trust=101.2 fatigue=0.0 threat=safe
...
[END]   task=basic_security episode=1 steps=50 total_reward=18.4200 score=0.7841 grade=B
```

---

## 🐳 Docker / HuggingFace Spaces Deployment

### Build and run locally

```bash
docker build -t secureai-guard .
docker run -p 7860:7860 secureai-guard
```

### HuggingFace Spaces

1. Create a new Space (Docker SDK)
2. Push this repository
3. The `Dockerfile` exposes port 7860 — HF Spaces will map it automatically
4. Set optional secrets: `HF_TOKEN`, `OPENAI_API_KEY`

### Resource requirements

- **CPU:** 2 vCPU (no GPU required; HF model loading is optional)
- **RAM:** 4–8 GB (8 GB recommended with transformers loaded)
- **Startup time:** ~15 seconds

---

## 🧠 HuggingFace Integration

`utils/hf_integration.py` loads a text-classification pipeline for real-time risk scoring.

- **Default model:** `distilbert-base-uncased-finetuned-sst-2-english`
- **Override:** Set `HF_RISK_MODEL` environment variable
- **Fallback:** If the model is unavailable, a deterministic keyword scorer activates automatically — the environment works fully offline

---

## 🔄 DPO Data Flywheel

Every step logs a `PreferencePair`:
- **chosen_action**: the action taken this step
- **rejected_actions**: the previous step's action
- **reward_delta**: improvement in reward

Retrieve via `GET /preference_data`. This data can be used directly for Direct Preference Optimisation (DPO) fine-tuning of LLM agents.

---

## 📈 Baseline Results

Rule-based agent (keyword heuristics, no LLM):

| Task | Avg Score | Avg Reward | Grade |
|---|---|---|---|
| basic_security | 0.74 | 14.2 | B |
| trust_management | 0.61 | 11.8 | C |
| adversarial_drift | 0.52 | 9.1 | D |

DQN agent (500 episodes training):

| Task | Avg Score | Avg Reward | Grade |
|---|---|---|---|
| basic_security | 0.83 | 18.9 | A |
| trust_management | 0.76 | 16.3 | B |
| adversarial_drift | 0.71 | 14.7 | B |

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `http://localhost:7860` | Environment server URL |
| `MODEL_NAME` | `gpt-3.5-turbo` | LLM model name |
| `HF_TOKEN` | — | HuggingFace token |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL |
| `HF_RISK_MODEL` | `distilbert-base-uncased-finetuned-sst-2-english` | Risk scorer model |
| `EPISODES_PER_TASK` | `1` | Episodes per task in inference.py |
| `SEED_BASE` | `42` | Base seed for reproducibility |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Submit a pull request

---

*SecureAI-Guard: Where Reinforcement Learning Meets Cybersecurity Excellence* 🛡️
