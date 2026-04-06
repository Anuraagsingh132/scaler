from typing import Dict, Any, List
from schema.models import State, Reward, GradeResult


class SecurityGrader:
    """
    Deterministic grader that scores an episode and returns a float in [0.0, 1.0].

    Scoring weights:
        - Security Efficiency  : 0.40  (blocked / total threats)
        - User Retention       : 0.30  (final trust / 100)
        - Precision            : 0.20  (1 - false_positive_rate)
        - Reasoning Quality    : 0.10  (avg reasoning component)
    """

    WEIGHTS = {
        "security": 0.40,
        "trust": 0.30,
        "precision": 0.20,
        "quality": 0.10,
    }

    def grade_episode(
        self,
        final_state: State,
        rewards: List[Reward],
        task_id: str = "basic_security",
    ) -> GradeResult:
        # ---- individual metrics (all in [0, 1]) ----
        security_eff = (
            final_state.blocked_threats / final_state.threat_count
            if final_state.threat_count > 0
            else 0.0
        )

        user_retention = max(0.0, final_state.user_trust) / 100.0

        total_decisions = final_state.threat_count + final_state.false_positives
        fp_rate = (
            final_state.false_positives / total_decisions
            if total_decisions > 0
            else 0.0
        )
        precision = max(0.0, 1.0 - fp_rate)

        avg_reasoning = (
            sum(r.components.reasoning_quality for r in rewards) / len(rewards)
            if rewards
            else 0.0
        )

        # ---- weighted score ----
        score = (
            self.WEIGHTS["security"] * security_eff
            + self.WEIGHTS["trust"] * user_retention
            + self.WEIGHTS["precision"] * precision
            + self.WEIGHTS["quality"] * avg_reasoning
        )
        score = round(min(max(score, 0.0), 1.0), 4)

        from tasks.registry import TaskRegistry
        try:
            threshold = TaskRegistry().get_task(task_id).success_threshold
        except ValueError:
            threshold = 0.7

        return GradeResult(
            score=score,
            passed=score >= threshold,
            grade=self._letter_grade(score),
            metrics={
                "security_efficiency": round(security_eff, 4),
                "user_retention": round(user_retention, 4),
                "false_positive_rate": round(fp_rate, 4),
                "precision": round(precision, 4),
                "reasoning_quality": round(avg_reasoning, 4),
            },
            details={
                "total_steps": final_state.step_count,
                "total_reward": final_state.total_reward,
                "threats_blocked": final_state.blocked_threats,
                "threat_count": final_state.threat_count,
                "false_positives": final_state.false_positives,
                "final_trust": round(final_state.user_trust, 2),
                "final_fatigue": round(final_state.system_fatigue, 2),
                "task_id": task_id,
                "threshold": threshold,
            },
        )

    @staticmethod
    def _letter_grade(score: float) -> str:
        if score >= 0.90:
            return "A+"
        if score >= 0.80:
            return "A"
        if score >= 0.70:
            return "B"
        if score >= 0.60:
            return "C"
        if score >= 0.50:
            return "D"
        return "F"
