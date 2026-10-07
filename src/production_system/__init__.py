"""Rule-based animal identification using forward chaining."""

from .engine import ForwardChainingEngine, InferenceResult, Rule
from .knowledge_base import ANIMALS, EXAMPLES, FEATURES, RULES, create_engine

__all__ = [
    "ANIMALS",
    "EXAMPLES",
    "FEATURES",
    "RULES",
    "ForwardChainingEngine",
    "InferenceResult",
    "Rule",
    "create_engine",
]
