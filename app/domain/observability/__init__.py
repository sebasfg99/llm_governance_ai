# ============================================================
# app/domain/observability — Cost tracking + unified logger
#
# This package complements the existing TraceService and
# EvalService. It does NOT replace them; it composes them
# along with the response-risk scorer so a single function
# (`log_llm_call`) writes the full audit trail in one place.
# ============================================================

from app.domain.observability.cost import estimate_cost, MODEL_PRICING
from app.domain.observability.logger import log_llm_call, ObservabilityLogger

__all__ = [
    "estimate_cost",
    "MODEL_PRICING",
    "log_llm_call",
    "ObservabilityLogger",
]
