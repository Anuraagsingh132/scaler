"""
dqn_baseline.py — Dueling DQN agent for SecureAI-Guard.

Usage:
    python dqn_baseline.py --episodes 500 --task basic_security
"""
import argparse
import random
from collections import deque, namedtuple
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim

Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.value = nn.Linear(hidden // 2, 1)
        self.advantage = nn.Linear(hidden // 2, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shared(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Experience(*args))

    def sample(self, n: int) -> List[Experience]:
        return random.sample(self.buf, n)

    def __len__(self) -> int:
        return len(self.buf)


class DQNAgent:
    DECISIONS = ["allow", "block", "warn", "investigate"]

    def __init__(self, state_size: int = 9, action_size: int = 4, lr: float = 1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buf = ReplayBuffer()
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        self.batch = 64
        self.update_every = 100
        self._step = 0

    def encode(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.array([
            obs["hf_risk_score"],
            obs["user_trust"] / 100.0,
            obs["system_fatigue"] / 100.0,
            float(obs["channel"] == "sms"),
            float(obs["channel"] == "email"),
            float(obs["channel"] == "web"),
            min(len(obs["content"]) / 500.0, 1.0),
            len(obs.get("threat_history", [])) / 10.0,
            obs["timestamp"] % 86400 / 86400,
        ], dtype=np.float32)

    def act(self, obs: Dict[str, Any], train: bool = True) -> Tuple[str, float, str]:
        if train and random.random() < self.eps:
            idx = random.randint(0, 3)
        else:
            s = torch.FloatTensor(self.encode(obs)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = int(self.q_net(s).argmax())
        decision = self.DECISIONS[idx]
        return decision, 0.85, self._reason(obs, decision)

    def _reason(self, obs: Dict[str, Any], decision: str) -> str:
        r = obs["hf_risk_score"]
        t = obs["user_trust"]
        msgs = {
            "block": f"High risk score ({r:.2f}) warrants blocking.",
            "warn": f"Moderate risk ({r:.2f}); warning user to maintain trust ({t:.0f}).",
            "investigate": f"Ambiguous risk ({r:.2f}); flagging for investigation.",
            "allow": f"Low risk ({r:.2f}) and trusted context; allowing message.",
        }
        return msgs[decision]

    def remember(self, obs, action_str, reward_val, next_obs, done):
        s = self.encode(obs)
        ns = self.encode(next_obs)
        a_idx = self.DECISIONS.index(action_str)
        self.buf.push(s, a_idx, reward_val, ns, done)

    def learn(self) -> float:
        if len(self.buf) < self.batch:
            return 0.0
        batch = Experience(*zip(*self.buf.sample(self.batch)))
        s = torch.FloatTensor(np.array(batch.state)).to(self.device)
        a = torch.LongTensor(np.array(batch.action)).to(self.device)
        r = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        ns = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        d = torch.BoolTensor(np.array(batch.done)).to(self.device)

        curr_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(ns).max(1)[0]
        target = r + self.gamma * next_q * ~d

        loss = nn.HuberLoss()(curr_q, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.opt.step()

        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        self._step += 1
        if self._step % self.update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())


def train(api: str, task_id: str, episodes: int):
    agent = DQNAgent()
    rewards = []

    for ep in range(episodes):
        resp = requests.post(f"{api}/reset", json={"task_id": task_id, "seed": ep})
        obs = resp.json()["observation"]
        total = 0.0
        done = False

        while not done:
            decision, conf, reason = agent.act(obs)
            step_resp = requests.post(
                f"{api}/step",
                json={"action": {"decision": decision, "confidence": conf, "reasoning": reason}},
            ).json()
            r_val = step_resp["reward"]["value"]
            agent.remember(obs, decision, r_val, step_resp["observation"], step_resp["done"])
            agent.learn()
            obs = step_resp["observation"]
            total += r_val
            done = step_resp["done"]

        rewards.append(total)
        if ep % 10 == 0:
            avg = float(np.mean(rewards[-10:]))
            print(f"Episode {ep:4d} | avg_reward={avg:.4f} | eps={agent.eps:.3f}")

    torch.save(agent.q_net.state_dict(), "dqn_checkpoint.pth")
    print(f"Training complete. Checkpoint saved to dqn_checkpoint.pth")
    return agent, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:7860")
    parser.add_argument("--task", default="basic_security")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    train(args.api, args.task, args.episodes)
