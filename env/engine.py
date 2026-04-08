import random
from typing import Dict, Any, Optional, List

from schema.models import (
    Observation, Action, Reward, State, StepResponse,
    ThreatType, CommunicationChannel, PreferencePair,
)
from utils.hf_integration import HFRiskScorer
from env.core import SecurityEnvironment


class SecureAIGuardEngine(SecurityEnvironment):
    """
    Full environment engine exposing reset(), step(), and state() methods.
    Fully deterministic when seed is provided.
    """

    def __init__(self):
        super().__init__()
        self.hf_scorer = HFRiskScorer()
        self.adversarial_memory: List[Dict[str, Any]] = []
        self.drift_counter = 0
        self.preference_data: List[Dict[str, Any]] = []
        self._seed: Optional[int] = None
        self._task_difficulty: str = "L1"
        self._max_steps: int = 50
        self._current_event: Optional[Dict[str, Any]] = None
        # Determinism: seeded counters replace uuid4/time.time
        self._event_counter: int = 0
        self._logical_time: float = 0.0
        # L3 drift: when set, the next event MUST use this threat type
        self._drifted_threat_type: Optional[ThreatType] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, task_id: Optional[str] = None) -> Observation:
        """Reset environment for a new episode. Returns first observation."""
        self._seed = seed if seed is not None else 42
        self.rng = random.Random(self._seed)
        # Deterministic episode ID derived from seed
        episode_id = f"ep-{self._seed:08x}-{self.rng.getrandbits(32):08x}"
        self.state = State(episode_id=episode_id)
        self.adversarial_memory = []
        self.preference_data = []
        self.drift_counter = 0
        self._event_counter = 0
        self._logical_time = 0.0
        self._drifted_threat_type = None

        # Apply task settings
        from tasks.registry import TaskRegistry
        registry = TaskRegistry()
        if task_id and task_id in registry.tasks:
            task = registry.tasks[task_id]
            self._task_difficulty = task.difficulty
            self._max_steps = task.max_steps
            # Store task params so core.py can access trust_penalty_multiplier
            self._task_params = task.parameters
        else:
            self._task_difficulty = "L1"
            self._max_steps = 50
            self._task_params = {}

        # Generate first event
        self._current_event = self._next_event()
        return self._build_observation(self._current_event)

    def step(self, action: Action) -> StepResponse:
        """Execute one step. Returns observation, reward, done, info, state.

        IMPORTANT: The returned observation reflects the NEW state after the
        action is processed, not the stale pre-action state.
        """
        if self._current_event is None:
            # Auto-init if not reset
            self._current_event = self._next_event()

        threat_type = self._current_event["threat_type"]

        # 1. Build observation for reward calculation (old event context)
        obs_for_reward = self._build_observation(self._current_event)
        reward = self._calculate_reward(action, obs_for_reward, threat_type)
        self._update_state(action, reward, threat_type)

        # Log for DPO
        self._log_preference(action, reward)

        # Adversarial drift for L3
        if self._task_difficulty == "L3":
            self._maybe_drift()

        done = self._check_done()

        info: Dict[str, Any] = {
            "threat_type": threat_type.value,
            "difficulty": self._task_difficulty,
            "adversarial_drift": self.state.adversarial_drift_active,
            "action": action.decision.value,
            "step": self.state.step_count,
        }

        # 2. Advance to next event THEN build fresh observation
        if not done:
            self._current_event = self._next_event()
        # Build observation from the NEW current event with UPDATED state
        new_observation = self._build_observation(self._current_event)

        return StepResponse(
            observation=new_observation,
            reward=reward,
            done=done,
            info=info,
            state=self.state,
        )

    def get_state(self) -> State:
        """Return current environment state snapshot."""
        return self.state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_event(self) -> Dict[str, Any]:
        channels = list(CommunicationChannel)
        channel = self.rng.choice(channels)

        # If drift has set a specific threat type, honour it
        if self._drifted_threat_type is not None:
            threat_type = self._drifted_threat_type
            self._drifted_threat_type = None  # consume once
        else:
            # Threat distribution by phase
            step = self.state.step_count
            if self._task_difficulty == "L1":
                threat_weights = [0.4, 0.0, 0.2, 0.0, 0.4]   # phishing, malware, spam, social_eng, safe
            elif self._task_difficulty == "L2":
                if step < 30:
                    threat_weights = [0.25, 0.1, 0.15, 0.1, 0.4]
                else:
                    threat_weights = [0.2, 0.2, 0.15, 0.2, 0.25]
            else:  # L3
                if step < 20:
                    threat_weights = [0.3, 0.1, 0.1, 0.1, 0.4]
                elif step < 50:
                    threat_weights = [0.2, 0.2, 0.15, 0.2, 0.25]
                else:
                    threat_weights = [0.25, 0.25, 0.1, 0.3, 0.1]

            threat_types = [
                ThreatType.PHISHING, ThreatType.MALWARE, ThreatType.SPAM,
                ThreatType.SOCIAL_ENGINEERING, ThreatType.SAFE,
            ]
            threat_type = self.rng.choices(threat_types, weights=threat_weights, k=1)[0]

        self.current_threat_type = threat_type
        self.state.active_threat_type = threat_type

        event = self._generate_event(threat_type, channel)
        self.adversarial_memory.append({
            "threat_type": threat_type.value,
            "step": self.state.step_count,
        })
        return event

    def _build_observation(self, event: Dict[str, Any]) -> Observation:
        hf_score = self.hf_scorer.score_text(event["content"])
        # Deterministic event ID from seed + counter
        event_id = f"evt-{self._seed:08x}-{self._event_counter:04d}"
        self._event_counter += 1
        # Advance logical time
        self._logical_time += 1.0
        return Observation(
            event_id=event_id,
            channel=event["channel"],
            sender=event["sender"],
            content=event["content"],
            timestamp=self._logical_time,
            hf_risk_score=hf_score,
            user_trust=self.state.user_trust,
            system_fatigue=self.state.system_fatigue,
            threat_history=self.adversarial_memory[-5:],
            metadata={
                "event_type": event["threat_type"].value,
                "step": self.state.step_count,
                "difficulty": self._task_difficulty,
            },
        )

    def _check_done(self) -> bool:
        if self.state.user_trust <= 0:
            return True
        if self.state.system_fatigue >= 100:
            return True
        if self.state.step_count >= self._max_steps:
            return True
        return False

    def _maybe_drift(self):
        """L3 adaptive attacker: mutate upcoming threat type based on agent performance."""
        if self.state.step_count > 20 and self.state.blocked_threats > 5:
            self.state.adversarial_drift_active = True
            self.drift_counter += 1
            if self.state.false_positives > 3:
                self._drifted_threat_type = ThreatType.SOCIAL_ENGINEERING
            elif self.state.blocked_threats / max(self.state.threat_count, 1) > 0.8:
                self._drifted_threat_type = self.rng.choice(
                    [ThreatType.MALWARE, ThreatType.SOCIAL_ENGINEERING]
                )

    def _log_preference(self, action: Action, reward: Reward):
        entry: Dict[str, Any] = {
            "chosen_action": action.model_dump(),
            "reward": reward.value,
            "pair": None,
        }
        if self.preference_data:
            prev = self.preference_data[-1]
            pair = PreferencePair(
                step=self.state.step_count,
                chosen_action=action,
                rejected_actions=[Action(**prev["chosen_action"])],
                reward_delta=reward.value - prev["reward"],
                timestamp=self._logical_time,  # deterministic logical time
            )
            entry["pair"] = pair.model_dump()
        self.preference_data.append(entry)

    def get_preference_data(self) -> List[Dict[str, Any]]:
        return [p["pair"] for p in self.preference_data if p["pair"] is not None]

    def set_difficulty(self, level: str):
        self._task_difficulty = level
        difficulty_steps = {"L1": 50, "L2": 75, "L3": 100}
        self._max_steps = difficulty_steps.get(level, 50)
