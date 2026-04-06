import time
import random
from typing import Dict, Any, List, Optional

from schema.models import (
    Observation, Action, Reward, State, RewardComponents,
    CommunicationChannel, ThreatType, DecisionType,
)


THREAT_TEMPLATES: Dict[ThreatType, List[str]] = {
    ThreatType.PHISHING: [
        "Click here to verify your account immediately: http://secure-login-{n}.net",
        "Your account will be suspended! Action required: http://paypal-verify-{n}.com",
        "You've won $1,000,000! Claim now: http://prize-claim-{n}.xyz",
        "Security alert: Confirm your identity now: http://bank-update-{n}.info",
        "Your password has expired. Reset here: http://reset-password-{n}.co",
    ],
    ThreatType.MALWARE: [
        "Download this important document: http://docs-share-{n}.net/invoice.exe",
        "Your invoice is attached: http://files-secure-{n}.com/doc.zip",
        "Install our mandatory security update: http://update-patch-{n}.org/patch.msi",
        "View your shipping label: http://track-package-{n}.info/label.pdf",
    ],
    ThreatType.SOCIAL_ENGINEERING: [
        "Hi, it's me! I need you to urgently send $500 via wire transfer.",
        "This is your manager. I need the VPN credentials ASAP for the audit.",
        "Family emergency! Please call this number immediately: +1-800-{n}.",
        "Your package is held at customs. Pay the $12 fee here: http://customs-{n}.com",
        "I'm stranded abroad and lost my wallet. Can you send money?",
    ],
    ThreatType.SPAM: [
        "Buy now! 90% off everything! Limited time only!",
        "Lose weight in 7 days with this miracle pill. Order now!",
        "You have been pre-approved for a $50,000 loan. Claim it today!",
        "Meet singles in your area tonight. Sign up free!",
    ],
    ThreatType.SAFE: [
        "Meeting confirmed for 3pm tomorrow in conference room B.",
        "Your Amazon order #112-{n} has shipped. Expected delivery: Friday.",
        "Thanks for your message, I'll get back to you by end of day.",
        "Don't forget about dinner tonight at 7pm!",
        "Your monthly statement is ready to view in your online account.",
        "Reminder: dentist appointment tomorrow at 10am.",
    ],
}

SENDER_POOLS: Dict[ThreatType, List[str]] = {
    ThreatType.PHISHING: [
        "security@paypa1.com", "noreply@amazon-verify.com",
        "support@google-account-alert.net", "admin@bankofamerica-secure.org",
    ],
    ThreatType.MALWARE: [
        "documents@share-files.net", "invoices@billing-dept.co",
        "it-support@company-updates.org",
    ],
    ThreatType.SOCIAL_ENGINEERING: [
        "unknown_number_+15550199", "colleague@corp-mail.com",
        "ceo@company-urgent.com", "support@customs-authority.net",
    ],
    ThreatType.SPAM: [
        "promo@mega-deals.com", "offers@flash-sale.net",
        "newsletter@discount-shop.biz",
    ],
    ThreatType.SAFE: [
        "mom@gmail.com", "boss@mycompany.com", "noreply@amazon.com",
        "friend@gmail.com", "calendar@google.com", "no-reply@bank.com",
    ],
}


class SecurityEnvironment:
    """Base environment with threat generation and reward logic."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.state = State()
        self.current_threat_type: ThreatType = ThreatType.SAFE

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def _generate_event(
        self,
        threat_type: ThreatType,
        channel: CommunicationChannel,
    ) -> Dict[str, Any]:
        templates = THREAT_TEMPLATES[threat_type]
        template = self.rng.choice(templates)
        n = self.rng.randint(1000, 9999)
        content = template.format(n=n)

        sender = self.rng.choice(SENDER_POOLS[threat_type])
        return {
            "channel": channel,
            "sender": sender,
            "content": content,
            "threat_type": threat_type,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Reward calculation
    # ------------------------------------------------------------------

    def _calculate_reward(
        self,
        action: Action,
        observation: Observation,
        threat_type: ThreatType,
    ) -> Reward:
        security = self._security_score(action, threat_type)
        user_friction = self._friction_score(action, threat_type)
        delay = -0.1 if action.decision == DecisionType.INVESTIGATE else 0.0
        reasoning_quality = self._reasoning_quality(action.reasoning, threat_type)

        # Dense shaped reward
        w1, w2, w3, w4 = 0.5, 0.3, 0.1, 0.1
        total = (
            w1 * security
            + w2 * user_friction
            + w3 * delay
            + w4 * reasoning_quality
        )
        # Partial credit: scale by confidence
        total *= 0.7 + 0.3 * action.confidence

        components = RewardComponents(
            security=round(security, 4),
            user_friction=round(user_friction, 4),
            delay=round(delay, 4),
            reasoning_quality=round(reasoning_quality, 4),
            total=round(total, 4),
        )
        return Reward(
            value=round(total, 4),
            components=components,
            explanation=(
                f"security={security:.2f}, friction={user_friction:.2f}, "
                f"delay={delay:.2f}, reasoning={reasoning_quality:.2f}"
            ),
        )

    def _security_score(self, action: Action, threat_type: ThreatType) -> float:
        if threat_type != ThreatType.SAFE:
            scores = {
                DecisionType.BLOCK: 1.0,
                DecisionType.INVESTIGATE: 0.7,
                DecisionType.WARN: 0.5,
                DecisionType.ALLOW: -1.0,
            }
        else:
            scores = {
                DecisionType.ALLOW: 0.5,
                DecisionType.WARN: 0.0,
                DecisionType.INVESTIGATE: 0.0,
                DecisionType.BLOCK: -0.8,
            }
        return scores.get(action.decision, 0.0)

    def _friction_score(self, action: Action, threat_type: ThreatType) -> float:
        if action.decision == DecisionType.WARN:
            return -0.2
        if action.decision == DecisionType.INVESTIGATE:
            return -0.1
        if action.decision == DecisionType.BLOCK and threat_type == ThreatType.SAFE:
            return -0.5
        return 0.0

    def _reasoning_quality(self, reasoning: str, threat_type: ThreatType) -> float:
        r = reasoning.lower()
        keyword_map = {
            ThreatType.PHISHING: ["link", "suspicious", "domain", "verification", "urgent", "phish"],
            ThreatType.MALWARE: ["attachment", "download", "executable", "scan", "suspicious", "malware"],
            ThreatType.SOCIAL_ENGINEERING: ["pressure", "emotional", "unusual", "verify", "emergency", "social"],
            ThreatType.SPAM: ["promotional", "unsolicited", "bulk", "spam", "offer"],
            ThreatType.SAFE: ["legitimate", "trusted", "safe", "known", "normal"],
        }
        keywords = keyword_map.get(threat_type, [])
        if not keywords:
            return 0.5
        hits = sum(1 for k in keywords if k in r)
        return min(hits / len(keywords) + 0.2, 1.0)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def _update_state(self, action: Action, reward: Reward, threat_type: ThreatType):
        if threat_type != ThreatType.SAFE:
            self.state.threat_count += 1
            if action.decision == DecisionType.BLOCK:
                self.state.blocked_threats += 1
                self.state.user_trust = min(100.0, self.state.user_trust + 1.0)
            elif action.decision == DecisionType.WARN:
                self.state.user_trust = min(100.0, self.state.user_trust + 0.5)
        else:
            if action.decision == DecisionType.BLOCK:
                self.state.user_trust = max(0.0, self.state.user_trust - 5.0)
                self.state.false_positives += 1
            elif action.decision == DecisionType.ALLOW:
                self.state.user_trust = min(100.0, self.state.user_trust + 0.2)

        if action.decision == DecisionType.WARN:
            self.state.system_fatigue = min(100.0, self.state.system_fatigue + 2.0)
        elif action.decision == DecisionType.INVESTIGATE:
            self.state.system_fatigue = min(100.0, self.state.system_fatigue + 1.0)
        elif action.decision == DecisionType.ALLOW and threat_type == ThreatType.SAFE:
            self.state.system_fatigue = max(0.0, self.state.system_fatigue - 0.5)

        self.state.step_count += 1
        self.state.total_reward = round(self.state.total_reward + reward.value, 4)
