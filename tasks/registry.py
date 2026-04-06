from typing import Dict, List
from schema.models import TaskDefinition


class TaskRegistry:
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {
            "basic_security": TaskDefinition(
                task_id="basic_security",
                name="Basic Security Screening",
                description=(
                    "L1 — Identify and block clear-cut phishing and spam. "
                    "Only phishing and spam threats appear. No adversarial drift. "
                    "Success requires blocking ≥80% of threats with <20% false-positive rate."
                ),
                difficulty="L1",
                max_steps=50,
                success_threshold=0.8,
                parameters={
                    "threat_types": ["phishing", "spam"],
                    "drift_enabled": False,
                    "safe_ratio": 0.4,
                },
            ),
            "trust_management": TaskDefinition(
                task_id="trust_management",
                name="Trust Management Challenge",
                description=(
                    "L2 — Handle gray-area messages from semi-trusted senders while "
                    "maintaining user trust above 50. Includes malware and social-engineering "
                    "messages. False positives significantly drain trust."
                ),
                difficulty="L2",
                max_steps=75,
                success_threshold=0.75,
                parameters={
                    "threat_types": ["phishing", "malware", "spam", "social_engineering"],
                    "drift_enabled": False,
                    "trust_penalty_multiplier": 1.5,
                    "safe_ratio": 0.3,
                },
            ),
            "adversarial_drift": TaskDefinition(
                task_id="adversarial_drift",
                name="Advanced Adversary Challenge",
                description=(
                    "L3 — Defend against an adaptive attacker that observes agent performance "
                    "and pivots tactics mid-episode (adversarial drift). All threat types active. "
                    "The attacker switches from phishing to social-engineering when blocked too often."
                ),
                difficulty="L3",
                max_steps=100,
                success_threshold=0.7,
                parameters={
                    "threat_types": ["phishing", "malware", "spam", "social_engineering"],
                    "drift_enabled": True,
                    "adaptation_threshold": 5,
                    "drift_start_step": 20,
                },
            ),
        }

    def get_task(self, task_id: str) -> TaskDefinition:
        if task_id not in self.tasks:
            raise ValueError(f"Task '{task_id}' not found. Available: {list(self.tasks)}")
        return self.tasks[task_id]

    def list_tasks(self) -> List[TaskDefinition]:
        return list(self.tasks.values())

    def get_by_difficulty(self, difficulty: str) -> List[TaskDefinition]:
        return [t for t in self.tasks.values() if t.difficulty == difficulty]
