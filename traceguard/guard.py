"""Online guard hook simulation."""

from __future__ import annotations

from traceguard.rules import RuleEngine
from traceguard.schema import GuardDecision, TrajectoryCase, TrajectoryStep


class TraceGuard:
    def __init__(self, rules: RuleEngine | None = None) -> None:
        self.rules = rules or RuleEngine()

    def before_tool_call(self, trajectory: TrajectoryCase, candidate_action: TrajectoryStep) -> GuardDecision:
        report = self.rules.evaluate(trajectory, candidate_action=candidate_action)
        assert report is not None
        return GuardDecision(decision=report.decision, report=report, candidate_action=candidate_action)

    def after_tool_observation(self, trajectory: TrajectoryCase, observation: TrajectoryStep) -> GuardDecision:
        extended = trajectory.model_copy(update={"trajectory": trajectory.trajectory + [observation]})
        report = self.rules.evaluate(extended)
        if report is None:
            from traceguard.schema import RiskReport

            report = RiskReport(
                label="safe",
                decision="allow",
                reason="Observation did not trigger a deterministic guard rule.",
                confidence=0.55,
            )
        return GuardDecision(decision=report.decision, report=report, candidate_action=observation)

