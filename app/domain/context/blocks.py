"""
ContextBlock — typed unit of prompt context.

Each block represents one logical section of the assembled prompt.
Priority values follow a "lower = higher importance" convention so that
sorted(blocks, key=lambda b: b.priority) yields the protection order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------
# Priority tiers
# ------------------------------------------------------------------
PRIORITY_REQUIRED = 0   # system prompt, user question, grounding contract
PRIORITY_CRITICAL = 1   # top RAG chunks by similarity (top 2-3)
PRIORITY_HIGH     = 2   # project memory (L2)
PRIORITY_MEDIUM   = 3   # remaining RAG chunks
PRIORITY_LOW      = 4   # conversation history (recent turns)
PRIORITY_LOWEST   = 5   # reasoning traces, oldest history turns


@dataclass
class ContextBlock:
    name: str
    content: str
    token_count: int
    priority: int
    source_type: str    # "system"|"retrieval"|"memory"|"history"|"trace"|"instruction"|"query"
    required: bool
    metadata: dict = field(default_factory=dict)
