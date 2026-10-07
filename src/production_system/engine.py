"""A small, deterministic forward-chaining inference engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Rule:
    """An IF-premises-THEN-conclusion production rule."""

    rule_id: str
    premises: frozenset[str]
    conclusion: str
    explanation: str = ""

    @classmethod
    def build(
        cls,
        rule_id: str,
        premises: Iterable[str],
        conclusion: str,
        explanation: str = "",
    ) -> "Rule":
        return cls(rule_id, frozenset(premises), conclusion, explanation)


@dataclass(frozen=True, slots=True)
class FiredRule:
    """One trace entry produced when a rule fires."""

    rule: Rule
    facts_before: frozenset[str]


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """The final working memory and explainable inference trace."""

    initial_facts: frozenset[str]
    final_facts: frozenset[str]
    fired_rules: tuple[FiredRule, ...]
    candidates: tuple[str, ...]

    @property
    def status(self) -> str:
        if len(self.candidates) == 1:
            return "recognized"
        if len(self.candidates) > 1:
            return "ambiguous"
        return "unknown"

    @property
    def animal(self) -> str | None:
        return self.candidates[0] if len(self.candidates) == 1 else None


class ForwardChainingEngine:
    """Apply rules in declared order until the working memory stops changing."""

    def __init__(self, rules: Sequence[Rule], terminal_facts: Iterable[str]):
        ids = [rule.rule_id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Rule IDs must be unique")
        self.rules = tuple(rules)
        self.terminal_facts = frozenset(terminal_facts)

    def infer(self, initial_facts: Iterable[str]) -> InferenceResult:
        initial = frozenset(fact.strip() for fact in initial_facts if fact.strip())
        facts = set(initial)
        fired_ids: set[str] = set()
        trace: list[FiredRule] = []

        changed = True
        while changed:
            changed = False
            for rule in self.rules:
                if rule.rule_id in fired_ids:
                    continue
                if rule.premises <= facts:
                    fired_ids.add(rule.rule_id)
                    if rule.conclusion not in facts:
                        trace.append(FiredRule(rule, frozenset(facts)))
                        facts.add(rule.conclusion)
                        changed = True

        candidates = tuple(sorted(facts & self.terminal_facts))
        return InferenceResult(initial, frozenset(facts), tuple(trace), candidates)
