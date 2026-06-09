import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def trim_to_token_budget(text: str, token_budget: int) -> str:
    max_chars = max(0, token_budget * 4)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "\n...[trimmed]"


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if len(token) > 2}


@dataclass
class MemoryRecord:
    id: str
    memory_type: str
    title: str
    summary: str
    tags: List[str] = field(default_factory=list)
    importance: float = 0.5
    pinned: bool = False
    file_path: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredMemory:
    memory: MemoryRecord
    score: float


class MemoryStore(Protocol):
    def search(self, query: str, memory_types: Optional[Sequence[str]] = None, limit: int = 5) -> List[ScoredMemory]:
        ...


class InMemoryMemoryStore:
    """Local fallback store. Swap this with Redis/Postgres/Chroma adapters later."""

    def __init__(self, memories: Optional[Sequence[MemoryRecord]] = None):
        self.memories = list(memories or [])

    def add(self, memory: MemoryRecord) -> None:
        self.memories.append(memory)

    def search(self, query: str, memory_types: Optional[Sequence[str]] = None, limit: int = 5) -> List[ScoredMemory]:
        query_terms = tokenize(query)
        allowed_types = set(memory_types or [])
        scored = []

        for memory in self.memories:
            if allowed_types and memory.memory_type not in allowed_types:
                continue

            memory_text = " ".join([memory.title, memory.summary, " ".join(memory.tags)])
            memory_terms = tokenize(memory_text)
            overlap = len(query_terms & memory_terms)
            semantic_similarity = overlap / math.sqrt(max(1, len(query_terms)) * max(1, len(memory_terms)))

            age_days = max(0, (datetime.now(timezone.utc) - memory.created_at).days)
            recency_score = 1 / (1 + age_days / 30)
            importance_score = max(0, min(1, memory.importance))
            pinned_score = 0.25 if memory.pinned else 0

            final_score = semantic_similarity + 0.2 * recency_score + 0.4 * importance_score + pinned_score
            if final_score > 0.18:
                scored.append(ScoredMemory(memory=memory, score=final_score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return self._remove_redundant(scored)[:limit]

    def _remove_redundant(self, scored: List[ScoredMemory]) -> List[ScoredMemory]:
        selected = []
        selected_terms: List[set[str]] = []
        for item in scored:
            terms = tokenize(item.memory.summary)
            if not terms:
                selected.append(item)
                continue
            redundancy = max((len(terms & existing) / max(1, len(terms | existing)) for existing in selected_terms), default=0)
            if redundancy < 0.75:
                selected.append(item)
                selected_terms.append(terms)
        return selected


class ContextBuilder:
    """Builds small, task-aware prompt context from summaries and retrieved memories."""

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        total_budget_tokens: int = 9000,
        recent_message_budget_tokens: int = 1200,
        memory_budget_tokens: int = 1500,
        document_budget_tokens: int = 2500,
    ):
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.total_budget_tokens = total_budget_tokens
        self.recent_message_budget_tokens = recent_message_budget_tokens
        self.memory_budget_tokens = memory_budget_tokens
        self.document_budget_tokens = document_budget_tokens

    def classify_intent(self, query: str, format_type: str = "markdown") -> str:
        lowered = query.lower()
        if format_type.lower() == "pptx" or any(word in lowered for word in ["ppt", "slides", "presentation"]):
            return "ppt"
        if any(word in lowered for word in ["calculate", "solve", "equation", "proof", "derivative", "integral"]):
            return "math"
        if any(word in lowered for word in ["csv", "excel", "dataset", "dataframe", "trend", "correlation"]):
            return "data"
        if any(word in lowered for word in ["research", "report", "sources", "citations"]):
            return "research"
        return "chat"

    def build(
        self,
        query: str,
        format_type: str = "markdown",
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_summary: str = "",
        recent_messages: Optional[Sequence[str]] = None,
        document_context: str = "",
        tool_context: str = "",
    ) -> Dict[str, Any]:
        intent = self.classify_intent(query, format_type)
        memory_types = self._memory_types_for_intent(intent)
        memories = self.memory_store.search(query, memory_types=memory_types, limit=5)

        sections = []
        profile_context = self._build_profile_context(user_profile or {}, query)
        if profile_context:
            sections.append(("USER PROFILE", profile_context))
        if conversation_summary:
            sections.append(("CONVERSATION SUMMARY", trim_to_token_budget(conversation_summary, 350)))
        recent_context = self._build_recent_context(recent_messages or [])
        if recent_context:
            sections.append(("RECENT CONVERSATION", recent_context))

        memory_context = self._build_memory_context(memories)
        if memory_context:
            sections.append(("RETRIEVED MEMORY", memory_context))
        if document_context:
            sections.append(("RETRIEVED DOCUMENTS", trim_to_token_budget(document_context, self.document_budget_tokens)))
        if tool_context:
            sections.append(("TOOL RESULTS", trim_to_token_budget(tool_context, 1000)))

        context = "\n\n".join(f"{title}:\n{body}" for title, body in sections)
        return {
            "intent": intent,
            "context": trim_to_token_budget(context, self.total_budget_tokens),
            "selected_memories": memories,
            "token_estimate": estimate_tokens(context),
        }

    def _memory_types_for_intent(self, intent: str) -> List[str]:
        if intent == "ppt":
            return ["user_preference", "project_state", "research_report", "presentation"]
        if intent == "math":
            return ["user_preference", "formula", "project_state"]
        if intent == "data":
            return ["user_preference", "dataset", "research_report", "project_state"]
        if intent == "research":
            return ["user_preference", "research_report", "document_summary", "project_state"]
        return ["user_preference", "project_state", "conversation_summary"]

    def _build_profile_context(self, user_profile: Dict[str, Any], query: str) -> str:
        query_terms = tokenize(query)
        selected = []
        for key, value in user_profile.items():
            key_terms = tokenize(str(key))
            value_terms = tokenize(str(value))
            if key_terms & query_terms or value_terms & query_terms or key in {"name", "preferred_style", "tech_stack"}:
                selected.append(f"- {key}: {value}")
        return trim_to_token_budget("\n".join(selected), 350)

    def _build_recent_context(self, recent_messages: Sequence[str]) -> str:
        recent = list(recent_messages)[-8:]
        return trim_to_token_budget("\n".join(f"- {message}" for message in recent), self.recent_message_budget_tokens)

    def _build_memory_context(self, memories: Sequence[ScoredMemory]) -> str:
        lines = []
        remaining_budget = self.memory_budget_tokens
        for item in memories[:5]:
            memory = item.memory
            memory_text = (
                f"- [{memory.memory_type}] {memory.title} "
                f"(score={item.score:.2f}, tags={', '.join(memory.tags) or 'none'})\n"
                f"  Summary: {memory.summary}\n"
                f"  File: {memory.file_path or 'not stored'}"
            )
            memory_text = trim_to_token_budget(memory_text, min(350, remaining_budget))
            lines.append(memory_text)
            remaining_budget -= estimate_tokens(memory_text)
            if remaining_budget <= 0:
                break
        return "\n".join(lines)
