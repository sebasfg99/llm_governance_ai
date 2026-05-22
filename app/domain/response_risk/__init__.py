# ============================================================
# app/domain/response_risk — Per-response risk scoring
#
# This package is intentionally separate from app/domain/risk/.
# `domain/risk` answers: "How risky is the AI PROJECT?" (EU AI
# Act tier, project-level risks). `domain/response_risk` answers:
# "How risky is THIS specific response?" (governance signal per
# trace).
# ============================================================

from app.domain.response_risk.scorer import compute_risk
from app.domain.response_risk.models import ResponseRisk, RiskLevel

__all__ = ["compute_risk", "ResponseRisk", "RiskLevel"]
