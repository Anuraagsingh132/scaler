from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from enum import Enum



class DecisionType(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    INVESTIGATE = "investigate"


class ThreatType(str, Enum):
    PHISHING = "phishing"
    MALWARE = "malware"
    SPAM = "spam"
    SOCIAL_ENGINEERING = "social_engineering"
    SAFE = "safe"


class CommunicationChannel(str, Enum):
    SMS = "sms"
    EMAIL = "email"
    WEB = "web"


class Action(BaseModel):
    decision: DecisionType = Field(..., description="Agent's security decision")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(..., description="Explanation for the decision")

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reasoning cannot be empty")
        return v.strip()


class Observation(BaseModel):
    event_id: str = Field(default="")
    channel: CommunicationChannel
    sender: str
    content: str
    timestamp: float
    hf_risk_score: float = Field(ge=0.0, le=1.0)
    user_trust: float = Field(ge=0.0, le=100.0)
    system_fatigue: float = Field(ge=0.0, le=100.0)
    threat_history: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RewardComponents(BaseModel):
    security: float = Field(description="Security component score")
    user_friction: float = Field(description="User friction penalty")
    delay: float = Field(description="Processing delay penalty")
    reasoning_quality: float = Field(description="Reasoning quality bonus")
    total: float = Field(description="Total reward")


class Reward(BaseModel):
    value: float
    components: RewardComponents
    explanation: str


class State(BaseModel):
    episode_id: str = Field(default="")
    step_count: int = 0
    total_reward: float = 0.0
    user_trust: float = 100.0
    system_fatigue: float = 0.0
    threat_count: int = 0
    blocked_threats: int = 0
    false_positives: int = 0
    active_threat_type: Optional[ThreatType] = None
    adversarial_drift_active: bool = False


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: Literal["L1", "L2", "L3"]
    max_steps: int = 100
    success_threshold: float = 0.7
    parameters: Dict[str, Any] = Field(default_factory=dict)


class PreferencePair(BaseModel):
    step: int
    chosen_action: Action
    rejected_actions: List[Action]
    reward_delta: float
    timestamp: float


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]
    state: State


class GradeResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Final score between 0.0 and 1.0")
    passed: bool
    grade: str
    metrics: Dict[str, float]
    details: Dict[str, Any]
