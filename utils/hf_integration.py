import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Risk keywords weighted by severity
HIGH_RISK_KEYWORDS = [
    "click here", "verify your account", "suspend", "suspended", "immediate action",
    "you've won", "claim now", "download", "install", "executable", "wire transfer",
    "urgent", "emergency", "credentials", "password expired", "password has expired",
    "customs fee", "stranded", "send money", "limited time", "pre-approved",
    "reset here", "action required", "confirm your identity",
]

MEDIUM_RISK_KEYWORDS = [
    "click", "verify", "account", "update", "confirm", "link", "http",
    "invoice", "document", "security", "alert", "warning", "expire",
    "offer", "free", "win", "prize", "loan", "approved",
]

# Safe-content patterns: messages matching these are very likely legitimate
SAFE_PATTERNS = [
    "meeting confirmed", "has shipped", "expected delivery",
    "get back to you", "dinner tonight", "appointment",
    "monthly statement", "conference room", "thanks for",
    "don't forget", "reminder:", "order #", "shipped",
    "end of day", "see you", "good morning", "good evening",
    "happy birthday", "thank you",
]


class HFRiskScorer:
    """
    HuggingFace-backed risk scorer with keyword-primary approach.

    The HF model (distilbert-sst2) is a *sentiment* classifier, so it can
    mis-score benign messages (e.g. "dentist appointment" → NEGATIVE sentiment
    → high "risk").  To prevent false positives the scorer uses keyword
    heuristics as the **primary** signal (80 %) and blends the HF model
    score in only as a secondary correction (20 %).
    """

    def __init__(self):
        self.classifier = None
        self._try_load_model()

    def _try_load_model(self):
        hf_token = os.environ.get("HF_TOKEN", "")
        try:
            from transformers import pipeline
            import torch

            model_name = os.environ.get(
                "HF_RISK_MODEL", "distilbert-base-uncased-finetuned-sst-2-english"
            )
            device = 0 if (hasattr(torch, "cuda") and torch.cuda.is_available()) else -1
            self.classifier = pipeline(
                "text-classification", model=model_name, device=device,
                token=hf_token if hf_token else None,
            )
            logger.info("HF model loaded: %s", model_name)
        except Exception as exc:
            logger.warning("HF model unavailable (%s). Using keyword fallback.", exc)
            self.classifier = None

    def score_text(self, text: str) -> float:
        """Return risk score in [0.0, 1.0].

        Uses keyword heuristics as the primary signal (80 %) and the HF
        sentiment model only as a weak secondary signal (20 %).
        """
        kw_score = self._keyword_score(text)

        hf_score = kw_score  # default: same as keyword if model unavailable
        if self.classifier is not None:
            try:
                result = self.classifier(text[:512])[0]
                label = result["label"].upper()
                score = float(result["score"])
                # NEGATIVE / LABEL_0 → risky; POSITIVE / LABEL_1 → safe
                if label in ("NEGATIVE", "LABEL_0"):
                    hf_score = min(score, 1.0)
                else:
                    hf_score = max(1.0 - score, 0.0)
            except Exception as exc:
                logger.warning("Classifier error: %s", exc)

        # Blend: keyword-primary (80%) + HF-secondary (20%)
        blended = 0.80 * kw_score + 0.20 * hf_score
        return round(min(max(blended, 0.0), 1.0), 4)

    def _keyword_score(self, text: str) -> float:
        t = text.lower()

        # Check for safe-content patterns first — if the message looks
        # clearly legitimate, give it a very low risk score.
        safe_hits = sum(1 for p in SAFE_PATTERNS if p in t)
        if safe_hits >= 1:
            high = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in t)
            # Only override if there are no high-risk keywords present
            if high == 0:
                return round(max(0.05 - safe_hits * 0.01, 0.0), 4)

        high = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in t)
        medium = sum(1 for kw in MEDIUM_RISK_KEYWORDS if kw in t)
        score = min(high * 0.25 + medium * 0.08, 1.0)
        return round(score, 4)
