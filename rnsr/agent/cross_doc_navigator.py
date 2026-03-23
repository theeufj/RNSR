"""
RNSR Cross-Document Navigator

Orchestrates multi-document queries by leveraging the knowledge graph
to find and link entities across documents.

This navigator handles queries like:
- "What happens to Person X mentioned in Document A in Document B?"
- "Compare the terms in Contract 1 and Contract 2"
- "Trace the timeline of events across all documents"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import structlog

from rnsr.extraction.models import Entity, EntityType, Relationship, RelationType
from rnsr.indexing.knowledge_graph import KnowledgeGraph
from rnsr.indexing.kv_store import KVStore
from rnsr.models import SkeletonNode

logger = structlog.get_logger(__name__)


# =============================================================================
# Cross-Document Query Models
# =============================================================================


@dataclass
class CrossDocQuery:
    """A decomposed cross-document query."""
    
    original_query: str
    entities_mentioned: list[str] = field(default_factory=list)
    documents_mentioned: list[str] = field(default_factory=list)
    query_type: str = "general"  # general, comparison, timeline, entity_tracking
    sub_queries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentResult:
    """Result from querying a single document."""
    
    doc_id: str
    doc_title: str
    answer: str
    evidence: list[str] = field(default_factory=list)
    entities_found: list[Entity] = field(default_factory=list)
    confidence: float = 0.0
    nodes_visited: int = 0
    iterations: int = 0


@dataclass
class CrossDocAnswer:
    """Final synthesized answer from cross-document query."""
    
    query: str
    answer: str
    document_results: list[DocumentResult] = field(default_factory=list)
    entities_involved: list[Entity] = field(default_factory=list)
    relationships_used: list[Relationship] = field(default_factory=list)
    confidence: float = 0.0
    trace: list[dict[str, Any]] = field(default_factory=list)
    total_nodes_visited: int = 0
    total_iterations: int = 0


# =============================================================================
# Entity Extraction from Query
# =============================================================================


QUERY_ENTITY_EXTRACTION_PROMPT = """Analyze this query and extract entities that need to be tracked across documents.

Query: {query}
{conversation_context}
{available_documents}
{document_coverage}
Extract:
1. People mentioned (names, roles)
2. Organizations mentioned
3. Documents or sections referenced — IMPORTANT disambiguation rules:
   a. If previous Q&A context exists, check whether the current question CONTINUES discussing the same document or SHIFTS to a different one. Look for topic changes — e.g. if prior questions were about a driver licence application but this question asks about "consent orders" or "the Court section", it is shifting to a different document.
   b. If the question uses generic phrases like "the document", "the letter", "this application", "this form", consider ALL available documents (listed above) and determine which one the question most likely refers to based on the question's specific content (e.g. keywords, legal concepts mentioned).
   c. When a question asks about something that clearly matches a specific document title (e.g. "Agreement for Sale of Shares"), reference that document even if the conversation was previously about a different document.
   d. If the question asks about content that exists in multiple documents (e.g. "What is the reference number on the letter?" when there are multiple letters), try to determine which document is most relevant from the question's context and conversation flow.
   e. DOCUMENT SHIFT DETECTION: When the conversation has asked MULTIPLE consecutive questions answered from the same document and the current question uses a GENERIC reference ("the document", "the letter", "this form") or asks a generic identity/meta question (title, date, reference number, court details, page count) that could apply to ANY document, prefer referencing a document that has NOT YET been discussed (see document coverage above). In a sequential Q&A review session, generic questions after exhausting one document's content typically signal a transition to the next document.
   f. TOPIC CONTINUITY: Conversely, when the current question asks about specific details that logically continue the same topic as recent questions (e.g. asking about "the recipient's address" after asking "who is the letter addressed to"), keep referencing the SAME document even if many questions have been asked about it.
   g. REFERENCE RESOLUTION: References like "this application", "this document", "the letter" ALWAYS refer to the MOST RECENTLY discussed document from the conversation context, unless the question introduces content that clearly belongs to a different document. If Q6 was about "Application for Consent Orders", then Q7's "this application" refers to the consent orders — NOT an application discussed in Q1-Q5. Always check the LAST Q&A pair first.
   h. SEQUENTIAL REVIEW — LETTER/DOCUMENT TYPE SELECTION: When transitioning to a new document type (e.g., from a costs agreement to "the letter") and MULTIPLE documents of that type exist, prefer the one whose title/topic is most thematically connected to what was just discussed. For example, after discussing a costs agreement, an "Invoice Cover Letter" is more relevant than a "Survey letter" or a general correspondence letter. Include the most likely document in documents_referenced.
4. Key legal concepts or events
5. Dates or time periods

OUTPUT FORMAT (JSON):
```json
{{
    "entities": [
        {{"name": "John Smith", "type": "PERSON", "role": "defendant"}},
        {{"name": "Contract A", "type": "DOCUMENT"}}
    ],
    "query_type": "entity_tracking|comparison|timeline|general",
    "documents_referenced": ["Document A", "Document B"]
}}
```

Respond with JSON only:"""


# =============================================================================
_STOP_WORDS = frozenset({
    "the", "is", "in", "at", "of", "and", "or", "to", "a", "an", "for",
    "on", "with", "by", "from", "as", "into", "this", "that", "it",
    "its", "are", "was", "were", "be", "been", "has", "have", "had",
    "do", "does", "did", "not", "but", "what", "which", "who", "whom",
    "how", "when", "where", "why", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "under", "about", "each",
    "made", "does", "document", "documents",
})


# Cross-Document Navigator
# =============================================================================


class CrossDocNavigator:
    """
    Orchestrates multi-document queries using the knowledge graph.
    
    Workflow:
    1. Extract entities from the query
    2. Resolve entities to documents via knowledge graph
    3. Plan retrieval across documents
    4. Execute per-document navigation
    5. Synthesize cross-document answer
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        document_navigators: dict[str, Any] | None = None,
        llm_fn: Callable[[str], str] | None = None,
    ):
        """
        Initialize the cross-document navigator.
        
        Args:
            knowledge_graph: Knowledge graph with entities and relationships.
            document_navigators: Dict mapping doc_id to navigator instances.
            llm_fn: LLM function for synthesis.
        """
        self.kg = knowledge_graph
        self.navigators = document_navigators or {}
        self._llm_fn = llm_fn
        
        # Cache for document content stores
        self._kv_stores: dict[str, KVStore] = {}
        self._skeletons: dict[str, dict[str, SkeletonNode]] = {}
        self._doc_titles: dict[str, str] = {}
    
    def set_llm_function(self, llm_fn: Callable[[str], str]) -> None:
        """Set the LLM function."""
        self._llm_fn = llm_fn

    def _get_doc_title(self, doc_id: str) -> str:
        """Resolve doc_id to a human-readable title."""
        return self._doc_titles.get(doc_id, doc_id)
    
    def register_document(
        self,
        doc_id: str,
        skeleton: dict[str, SkeletonNode],
        kv_store: KVStore,
        navigator: Any = None,
        title: str | None = None,
    ) -> None:
        """
        Register a document's resources for cross-document queries.
        
        Args:
            doc_id: Document ID.
            skeleton: Skeleton index for the document.
            kv_store: KV store with document content.
            navigator: Optional pre-configured navigator.
            title: Human-readable document title / filename.
        """
        self._skeletons[doc_id] = skeleton
        self._kv_stores[doc_id] = kv_store
        if title:
            self._doc_titles[doc_id] = title
        
        if navigator:
            self.navigators[doc_id] = navigator
        
        logger.info("document_registered", doc_id=doc_id, title=title or doc_id)
    
    def query(
        self,
        question: str,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> CrossDocAnswer:
        """
        Execute a cross-document query.
        
        Args:
            question: The user's question.
            conversation_context: Previous Q&A pairs for resolving
                ambiguous references (e.g. "this application").
                Each dict has ``question`` and ``answer`` keys.
            
        Returns:
            CrossDocAnswer with synthesized result.
        """
        trace = []
        
        # Step 1: Extract entities from query
        trace.append({
            "step": "extract_entities",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        query_analysis = self._analyze_query(question, conversation_context)
        
        trace.append({
            "step": "query_analyzed",
            "entities": query_analysis.entities_mentioned,
            "type": query_analysis.query_type,
        })
        
        # Step 2: Resolve entities to documents
        trace.append({
            "step": "resolve_entities",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        doc_entities = self._resolve_entities_to_documents(query_analysis)
        
        trace.append({
            "step": "entities_resolved",
            "doc_count": len(doc_entities),
            "documents": list(doc_entities.keys()),
        })
        
        # Step 3: Plan retrieval
        trace.append({
            "step": "plan_retrieval",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        retrieval_plan = self._plan_retrieval(
            question, query_analysis, doc_entities,
            conversation_context=conversation_context,
        )
        
        # Step 4: Execute per-document navigation
        trace.append({
            "step": "execute_navigation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        document_results = self._execute_navigation(retrieval_plan)
        
        trace.append({
            "step": "navigation_complete",
            "results_count": len(document_results),
        })
        
        # Step 5: Synthesize cross-document answer
        trace.append({
            "step": "synthesize",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        title_scores = {
            t["doc_id"]: t.get("title_score", 0)
            for t in retrieval_plan
        }
        answer = self._synthesize_answer(
            question,
            query_analysis,
            document_results,
            doc_entities,
            title_scores=title_scores,
            conversation_context=conversation_context,
        )
        
        answer.trace = trace
        
        logger.info(
            "cross_doc_query_complete",
            query=question[:100],
            documents=len(document_results),
            confidence=answer.confidence,
        )
        
        return answer
    
    def _analyze_query(
        self,
        question: str,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> CrossDocQuery:
        """
        Analyze the query to extract entities and determine query type.
        
        Args:
            question: The user's question.
            conversation_context: Recent Q&A pairs for disambiguation.
            
        Returns:
            CrossDocQuery with extracted information.
        """
        result = CrossDocQuery(original_query=question)
        
        if not self._llm_fn:
            # Basic extraction without LLM
            result.query_type = "general"
            return result
        
        try:
            ctx_block = ""
            if conversation_context:
                recent = conversation_context[-3:]
                lines = ["Previous Q&A context (most recent last):"]
                for pair in recent:
                    lines.append(f"  Q: {pair['question']}")
                    answer_preview = pair["answer"][:200]
                    lines.append(f"  A: {answer_preview}")
                    if pair.get("source_document"):
                        lines.append(f"  Source document: {pair['source_document']}")
                ctx_block = "\n".join(lines) + "\n"

            docs_block = ""
            if self._doc_titles:
                doc_lines = ["Available documents in this collection:"]
                for doc_id, title in self._doc_titles.items():
                    doc_lines.append(f"  - {title}")
                docs_block = "\n".join(doc_lines) + "\n"

            coverage_block = self._build_document_coverage(conversation_context)

            prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(
                query=question,
                conversation_context=ctx_block,
                available_documents=docs_block,
                document_coverage=coverage_block,
            )
            response = self._llm_fn(prompt)
            
            # Parse JSON response
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{[\s\S]*\}', response)
                json_str = json_match.group(0) if json_match else "{}"
            
            parsed = json.loads(json_str)
            
            # Extract entity names
            entities = parsed.get("entities", [])
            result.entities_mentioned = [e.get("name", "") for e in entities if e.get("name")]
            result.query_type = parsed.get("query_type", "general")
            result.documents_mentioned = parsed.get("documents_referenced", [])
            
            logger.debug(
                "query_analyzed",
                entities=result.entities_mentioned,
                type=result.query_type,
            )
            
        except Exception as e:
            logger.warning("query_analysis_failed", error=str(e))
        
        return result
    
    def _resolve_entities_to_documents(
        self,
        query: CrossDocQuery,
    ) -> dict[str, list[Entity]]:
        """
        Resolve mentioned entities to their appearances in documents.
        
        Args:
            query: Analyzed query with entity mentions.
            
        Returns:
            Dict mapping doc_id to list of entities found.
        """
        doc_entities: dict[str, list[Entity]] = {}
        
        for entity_name in query.entities_mentioned:
            # Search knowledge graph for this entity
            entities = self.kg.find_entities_by_name(entity_name, fuzzy=True)
            
            for entity in entities:
                # Get all documents where this entity appears
                for doc_id in entity.document_ids:
                    if doc_id not in doc_entities:
                        doc_entities[doc_id] = []
                    if entity not in doc_entities[doc_id]:
                        doc_entities[doc_id].append(entity)
                
                # Also check linked entities across documents
                linked = self.kg.find_entity_across_documents(entity.id)
                for linked_entity in linked:
                    for doc_id in linked_entity.document_ids:
                        if doc_id not in doc_entities:
                            doc_entities[doc_id] = []
                        if linked_entity not in doc_entities[doc_id]:
                            doc_entities[doc_id].append(linked_entity)
        
        return doc_entities
    
    def _resolve_docs_mentioned(
        self, documents_mentioned: list[str],
    ) -> set[str]:
        """Fuzzy-match ``documents_mentioned`` strings against registered doc titles.

        Returns a set of doc_ids whose titles match at least one mention.
        """
        matched: set[str] = set()
        if not documents_mentioned:
            return matched
        for mention in documents_mentioned:
            mention_lower = mention.lower()
            mention_words = set(re.findall(r"[a-z0-9]+", mention_lower))
            for doc_id, title in self._doc_titles.items():
                title_lower = title.lower()
                if mention_lower in title_lower or title_lower in mention_lower:
                    matched.add(doc_id)
                    continue
                title_words = set(re.findall(r"[a-z0-9]+", title_lower))
                if len(mention_words & title_words) >= 2:
                    matched.add(doc_id)
        if matched:
            logger.info(
                "docs_mentioned_resolved",
                mentions=documents_mentioned,
                matched_doc_ids=list(matched),
            )
        return matched

    def _compute_title_score(
        self,
        doc_id: str,
        question: str,
        documents_mentioned: list[str] | None = None,
        *,
        is_generic_reference: bool = False,
    ) -> float:
        """Score a document's relevance based on title vs query term overlap.

        Higher scores mean the document title better matches the question.
        Used to prioritise navigation order so the most relevant document
        is queried first.

        When *is_generic_reference* is True the boost from
        ``documents_mentioned`` is reduced because the LLM's resolution
        of a generic phrase like "the letter" to a specific document
        title is unreliable.
        """
        title = self._doc_titles.get(doc_id, "").lower()
        if not title:
            return 0.0

        title_words = set(re.findall(r"[a-z0-9]+", title))
        query_words = {
            w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in self._STOPWORDS and len(w) > 2
        }

        overlap = title_words & query_words
        score = len(overlap) * 2.0

        exact_boost = 3.0 if is_generic_reference else 10.0
        partial_boost = 2.0 if is_generic_reference else 5.0

        if documents_mentioned:
            best_mention_boost = 0.0
            for mention in documents_mentioned:
                mention_lower = mention.lower()
                if mention_lower in title or title in mention_lower:
                    best_mention_boost = max(best_mention_boost, exact_boost)
                else:
                    mention_words = set(re.findall(r"[a-z0-9]+", mention_lower))
                    mention_overlap = title_words & mention_words
                    if len(mention_overlap) >= 2:
                        best_mention_boost = max(best_mention_boost, partial_boost)
            score += best_mention_boost

        return score

    def _plan_retrieval(
        self,
        question: str,
        query: CrossDocQuery,
        doc_entities: dict[str, list[Entity]],
        conversation_context: list[dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Plan the retrieval strategy for each document.

        Documents with entity matches get entity-focused sub-queries.
        All other registered documents are still searched with the
        original question so that no document is silently skipped.

        Tasks are sorted by title relevance so the most likely target
        document is navigated first.  When conversation context is
        available, recently-discussed documents are deprioritized for
        generic-reference questions to encourage exploration.

        Args:
            question: Original question.
            query: Analyzed query.
            doc_entities: Entities by document.
            conversation_context: Prior Q&A pairs with source_document.

        Returns:
            List of retrieval tasks sorted by title relevance.
        """
        tasks = []
        planned_doc_ids: set[str] = set()

        is_generic = bool(self._GENERIC_REF_RE.search(question))

        for doc_id, entities in doc_entities.items():
            entity_names = [e.canonical_name for e in entities]

            if query.query_type == "entity_tracking":
                entity_hint = ", ".join(entity_names[:3])
                sub_query = f"{question}\n\nFocus on entities: {entity_hint}"
            elif query.query_type == "comparison":
                sub_query = f"Extract the relevant details for comparison: {question}"
            elif query.query_type == "timeline":
                entity_hint = ", ".join(entity_names[:3])
                sub_query = f"{question}\n\nKey entities: {entity_hint}"
            else:
                sub_query = question

            target_nodes = set()
            for entity in entities:
                target_nodes.update(entity.node_ids)

            title_score = self._compute_title_score(
                doc_id, question, query.documents_mentioned,
                is_generic_reference=is_generic,
            )
            tasks.append({
                "doc_id": doc_id,
                "sub_query": sub_query,
                "entities": entities,
                "target_nodes": list(target_nodes),
                "title_score": title_score,
            })
            planned_doc_ids.add(doc_id)

        all_registered = set(self.navigators.keys()) | set(self._kv_stores.keys())
        _SMALL_COLLECTION_THRESHOLD = 10
        skip_filtering = len(all_registered) <= _SMALL_COLLECTION_THRESHOLD

        mentioned_doc_ids = self._resolve_docs_mentioned(query.documents_mentioned)

        for doc_id in all_registered - planned_doc_ids:
            force_include = doc_id in mentioned_doc_ids
            if (
                not force_include
                and not skip_filtering
                and not self._doc_has_keyword_overlap(doc_id, question)
            ):
                logger.debug(
                    "fallback_doc_skipped",
                    doc_id=doc_id,
                    reason="no keyword overlap with query",
                )
                continue

            title_score = self._compute_title_score(
                doc_id, question, query.documents_mentioned,
                is_generic_reference=is_generic,
            )
            tasks.append({
                "doc_id": doc_id,
                "sub_query": question,
                "entities": [],
                "target_nodes": [],
                "title_score": title_score,
            })
            logger.debug(
                "fallback_doc_included",
                doc_id=doc_id,
                title_score=title_score,
                reason="documents_mentioned match" if force_include
                       else ("small collection — querying all docs" if skip_filtering
                             else "keyword overlap found, searching with original query"),
            )

        self._apply_freshness_adjustments(tasks, question, conversation_context)
        self._apply_continuity_boost(tasks, question, conversation_context)

        tasks.sort(key=lambda t: t.get("title_score", 0), reverse=True)

        if tasks:
            logger.info(
                "retrieval_plan_ordered",
                order=[(t["doc_id"], self._doc_titles.get(t["doc_id"], ""), t.get("title_score", 0)) for t in tasks[:5]],
            )
        
        return tasks

    def _apply_freshness_adjustments(
        self,
        tasks: list[dict[str, Any]],
        question: str,
        conversation_context: list[dict[str, str]] | None,
    ) -> None:
        """Adjust title scores based on conversation history.

        Only activates when the question uses generic references (e.g.
        "the document", "the letter").  Applies three mechanisms:

        1. **Dominant-document boost** — the document that was the source
           for the majority of recent answers gets a proportional boost
           and is exempt from penalty (handles multi-question sequences
           about the same document).
        2. **Freshness penalty** — documents discussed many times but NOT
           dominant receive a penalty to encourage progression.
        3. **Transition boost** — when a single undiscussed document
           remains and the most-recent document has been used >=2 times
           consecutively, the undiscussed document is boosted to match
           the top score (natural document progression).
        """
        if not conversation_context or not self._doc_titles:
            return

        if not self._GENERIC_REF_RE.search(question):
            return

        title_to_docid: dict[str, str] = {}
        for doc_id, title in self._doc_titles.items():
            title_to_docid[title.lower()] = doc_id

        discussed_doc_ids: set[str] = set()
        discussion_counts: dict[str, int] = {}

        for pair in conversation_context:
            src = pair.get("source_document", "")
            if not src:
                continue
            matched_id = self._match_source_to_docid(src, title_to_docid)
            if matched_id:
                discussed_doc_ids.add(matched_id)
                discussion_counts[matched_id] = (
                    discussion_counts.get(matched_id, 0) + 1
                )

        recent_reversed = list(reversed(
            [did for pair in conversation_context
             for did in [self._match_source_to_docid(
                 pair.get("source_document", ""), title_to_docid)]
             if did]
        ))
        most_recent = recent_reversed[0] if recent_reversed else None
        most_recent_consecutive = 0
        if most_recent:
            for did in recent_reversed:
                if did == most_recent:
                    most_recent_consecutive += 1
                else:
                    break

        window = min(5, len(recent_reversed))
        recent_window = recent_reversed[:window] if window else []
        window_counts: dict[str, int] = {}
        for did in recent_window:
            window_counts[did] = window_counts.get(did, 0) + 1
        dominant = max(window_counts, key=window_counts.get) if window_counts else None
        dominant_ratio = window_counts.get(dominant, 0) / window if dominant and window else 0

        exempt_from_penalty: set[str] = set()
        if most_recent and most_recent_consecutive < 3:
            exempt_from_penalty.add(most_recent)
        if dominant and dominant_ratio >= 0.5:
            exempt_from_penalty.add(dominant)
            dominant_boost = dominant_ratio * 5.0
            for task in tasks:
                if task["doc_id"] == dominant:
                    task["title_score"] = task.get("title_score", 0) + dominant_boost
                    logger.info(
                        "dominant_doc_boost",
                        doc_id=dominant,
                        title=self._doc_titles.get(dominant, ""),
                        boost=round(dominant_boost, 1),
                        ratio=round(dominant_ratio, 2),
                        window=window,
                    )
                    break

        for task in tasks:
            if task["doc_id"] in exempt_from_penalty:
                continue
            count = discussion_counts.get(task["doc_id"], 0)
            if count >= 2:
                penalty = -1.5 * count
                task["title_score"] = task.get("title_score", 0) + penalty
                logger.info(
                    "freshness_penalty_applied",
                    doc_id=task["doc_id"],
                    title=self._doc_titles.get(task["doc_id"], ""),
                    penalty=penalty,
                    discussion_count=count,
                )

        all_task_ids = {t["doc_id"] for t in tasks}
        undiscussed = all_task_ids - discussed_doc_ids

        if most_recent_consecutive >= 3:
            if undiscussed:
                for task in tasks:
                    if task["doc_id"] in undiscussed:
                        task["title_score"] = task.get("title_score", 0) + 1.0
        elif most_recent:
            if not (dominant and dominant != most_recent and dominant_ratio >= 0.5):
                for task in tasks:
                    if task["doc_id"] == most_recent:
                        task["title_score"] = task.get("title_score", 0) + 2.0
                        logger.info(
                            "continuity_boost_in_freshness",
                            doc_id=most_recent,
                            title=self._doc_titles.get(most_recent, ""),
                            boost=2.0,
                            consecutive=most_recent_consecutive,
                        )
                        break

        if len(undiscussed) > 1:
            max_undiscussed = max(
                (t.get("title_score", 0) for t in tasks
                 if t["doc_id"] in undiscussed),
                default=0,
            )
            for task in tasks:
                if (
                    task["doc_id"] in undiscussed
                    and task.get("title_score", 0) < max_undiscussed
                ):
                    task["title_score"] = max_undiscussed
                    logger.info(
                        "undiscussed_score_equalized",
                        doc_id=task["doc_id"],
                        title=self._doc_titles.get(task["doc_id"], ""),
                        new_score=max_undiscussed,
                    )

        if len(undiscussed) == 1 and most_recent_consecutive >= 2:
            top_score = max(
                (t.get("title_score", 0) for t in tasks), default=0,
            )
            for task in tasks:
                if task["doc_id"] in undiscussed:
                    if task.get("title_score", 0) < top_score:
                        task["title_score"] = top_score
                        logger.info(
                            "single_undiscussed_transition_boost",
                            doc_id=task["doc_id"],
                            title=self._doc_titles.get(task["doc_id"], ""),
                            new_score=top_score,
                        )
                    break

    @staticmethod
    def _match_source_to_docid(
        src: str, title_to_docid: dict[str, str],
    ) -> str | None:
        if not src:
            return None
        src_lower = src.lower()
        for t_lower, did in title_to_docid.items():
            if src_lower in t_lower or t_lower in src_lower:
                return did
        return None

    def _apply_continuity_boost(
        self,
        tasks: list[dict[str, Any]],
        question: str,
        conversation_context: list[dict[str, str]] | None,
    ) -> None:
        """Boost the most-recently-discussed document for topic continuity.

        Activates when the question does NOT signal a document switch
        (no generic reference pattern) AND no document has positive
        title_score from keyword matching.  This handles implicit
        continuation where consecutive questions ask about properties
        of the same document without explicitly naming it.
        """
        if not conversation_context or not self._doc_titles:
            return
        if self._GENERIC_REF_RE.search(question):
            return
        max_ts = max((t.get("title_score", 0) for t in tasks), default=0)
        if max_ts > 0:
            return

        recent_srcs = [
            p.get("source_document", "")
            for p in conversation_context[-3:]
            if p.get("source_document")
        ]
        if not recent_srcs or len(set(recent_srcs)) != 1:
            return

        cont_lower = recent_srcs[0].lower()
        for task in tasks:
            title = self._doc_titles.get(task["doc_id"], "").lower()
            if cont_lower in title or title in cont_lower:
                task["title_score"] = task.get("title_score", 0) + 5.0
                logger.info(
                    "continuity_boost_applied",
                    doc_id=task["doc_id"],
                    title=self._doc_titles.get(task["doc_id"], ""),
                    boost=5.0,
                )
                break

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "it", "this", "that",
        "with", "from", "by", "as", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "what", "when", "where", "who", "how", "which", "there", "about",
    })

    _GENERIC_REF_RE = re.compile(
        r'\b(?:the|this)\s+(?:document|letter|form|application|agreement|contract)\b'
        r'|\bwhat\s+is\s+the\s+(?:title|date|heading)\b'
        r'|\bhow\s+many\s+pages\b'
        r'|\b(?:the|this)\s+\S+\s+section\b'
        r'|\bunder\s+(?:the|which)\b'
        r"|\b(?:the|this)\s+(?:recipient|sender|addressee|signatory|author)(?:['\u2019]s)?\s",
        re.IGNORECASE,
    )

    def _doc_has_keyword_overlap(self, doc_id: str, question: str) -> bool:
        """Check if a document's title or skeleton has any keyword overlap with the query."""
        skeleton = self._skeletons.get(doc_id)
        if not skeleton:
            return True

        query_words = {
            w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in self._STOPWORDS and len(w) > 2
        }
        if not query_words:
            return True

        doc_text_parts: list[str] = []
        title = self._doc_titles.get(doc_id, "")
        if title:
            doc_text_parts.append(title.lower())
        for node in skeleton.values():
            doc_text_parts.append(node.header.lower())
            doc_text_parts.append(node.summary.lower())
        doc_text = " ".join(doc_text_parts)
        doc_words = set(re.findall(r"[a-z0-9]+", doc_text))

        return bool(query_words & doc_words)

    _EARLY_TERMINATION_CONFIDENCE = 0.9

    def _execute_navigation(
        self,
        tasks: list[dict[str, Any]],
    ) -> list[DocumentResult]:
        """
        Execute navigation for each document task.

        Stops early when a result exceeds the confidence threshold,
        skipping remaining documents to save time.
        
        Args:
            tasks: List of retrieval tasks.
            
        Returns:
            List of per-document results.
        """
        results = []

        max_title_score = max(
            (t.get("title_score", 0) for t in tasks), default=0,
        )
        for task in tasks:
            task["_is_primary"] = (
                max_title_score > 0
                and task.get("title_score", 0) == max_title_score
            )

        for i, task in enumerate(tasks):
            doc_id = task["doc_id"]
            
            if doc_id in self.navigators:
                navigator = self.navigators[doc_id]
                result = self._navigate_with_navigator(task, navigator)
            elif doc_id in self._kv_stores:
                result = self._direct_content_retrieval(task)
            else:
                logger.warning("no_navigator_for_doc", doc_id=doc_id)
                result = DocumentResult(
                    doc_id=doc_id,
                    doc_title=self._get_doc_title(doc_id),
                    answer="Document not accessible",
                    confidence=0.0,
                )
            
            results.append(result)

            # Only use early termination for large collections (>10 docs).
            # For small collections the cost of querying all docs is low and
            # skipping docs risks missing the correct answer in a different
            # document that also matches the query.
            if (
                len(tasks) > 10
                and result.confidence >= self._EARLY_TERMINATION_CONFIDENCE
                and result.answer
                and not self._is_negative_answer(result.answer)
                and i < len(tasks) - 1
            ):
                skipped = [t["doc_id"] for t in tasks[i + 1 :]]
                logger.info(
                    "early_termination",
                    confident_doc=doc_id,
                    confidence=result.confidence,
                    skipped_docs=skipped,
                )
                break
        
        return results
    
    def _navigate_with_navigator(
        self,
        task: dict[str, Any],
        navigator: Any,
    ) -> DocumentResult:
        """
        Execute navigation using a document navigator.
        
        Args:
            task: Retrieval task.
            navigator: Document navigator instance.
            
        Returns:
            DocumentResult.
        """
        doc_id = task["doc_id"]
        
        try:
            resolver = getattr(self, "_kg_resolver", None)
            if resolver is not None:
                resolution = resolver.try_resolve(
                    task["sub_query"], doc_ids=[doc_id]
                )
                if resolution.resolved and resolution.answer:
                    return DocumentResult(
                        doc_id=doc_id,
                        doc_title=self._get_doc_title(doc_id),
                        answer=resolution.answer,
                        entities_found=task["entities"],
                        confidence=resolution.confidence,
                        nodes_visited=0,
                        iterations=0,
                    )
                nav_metadata: dict[str, Any] | None = None
                if resolution.entity_node_ids:
                    nav_metadata = {
                        "entity_priority_nodes": resolution.entity_node_ids
                    }
            else:
                nav_metadata = None

            if task.get("_is_primary") or task.get("title_score", 0) > 0:
                if nav_metadata is None:
                    nav_metadata = {}
                nav_metadata["primary_document"] = bool(task.get("_is_primary"))
                nav_metadata["title_score"] = task.get("title_score", 0)

            nav_result = navigator.navigate(
                task["sub_query"], metadata=nav_metadata
            )
            
            visited = nav_result.get("visited_nodes", [])
            return DocumentResult(
                doc_id=doc_id,
                doc_title=self._get_doc_title(doc_id),
                answer=nav_result.get("answer", ""),
                evidence=nav_result.get("variables", []),
                entities_found=task["entities"],
                confidence=nav_result.get("confidence", 0.5),
                nodes_visited=len(visited) if isinstance(visited, list) else 0,
                iterations=nav_result.get("iteration", 0),
            )
            
        except Exception as e:
            logger.error("navigation_failed", doc_id=doc_id, error=str(e))
            return DocumentResult(
                doc_id=doc_id,
                doc_title=self._get_doc_title(doc_id),
                answer=f"Error: {str(e)}",
                confidence=0.0,
            )
    
    def _direct_content_retrieval(
        self,
        task: dict[str, Any],
    ) -> DocumentResult:
        """
        Retrieve content directly from target nodes.
        
        Args:
            task: Retrieval task.
            
        Returns:
            DocumentResult.
        """
        doc_id = task["doc_id"]
        kv_store = self._kv_stores.get(doc_id)
        
        if not kv_store:
            return DocumentResult(
                doc_id=doc_id,
                doc_title=self._get_doc_title(doc_id),
                answer="Content not available",
                confidence=0.0,
            )
        
        # Retrieve content from target nodes
        evidence = []
        for node_id in task["target_nodes"]:
            content = kv_store.get(node_id)
            if content:
                evidence.append(content)
        
        if not evidence:
            return DocumentResult(
                doc_id=doc_id,
                doc_title=self._get_doc_title(doc_id),
                answer="No relevant content found",
                confidence=0.0,
            )
        
        # Synthesize answer from evidence if we have LLM
        if self._llm_fn:
            entity_names = [e.canonical_name for e in task["entities"]]
            
            synthesis_prompt = f"""Based on the following content, answer the question.

Question: {task['sub_query']}

Focus on: {', '.join(entity_names)}

Content:
{chr(10).join(f'--- Section ---{chr(10)}{e}' for e in evidence)}

Answer:"""
            
            try:
                answer = self._llm_fn(synthesis_prompt)
            except Exception as e:
                answer = f"Error synthesizing: {str(e)}"
        else:
            answer = "\n\n".join(evidence)
        
        return DocumentResult(
            doc_id=doc_id,
            doc_title=self._get_doc_title(doc_id),
            answer=answer,
            evidence=evidence,
            entities_found=task["entities"],
            confidence=0.7 if evidence else 0.0,
        )
    
    _NEGATIVE_ANSWER_PATTERNS = (
        "i cannot answer",
        "cannot answer",
        "cannot be determined",
        "cannot determine",
        "no relevant content found",
        "no relevant documents found",
        "content not available",
        "document not accessible",
        "error during navigation",
        "error during synthesis",
        "unable to determine",
        "unable to find",
        "unable to answer",
        "unable to identify",
        "insufficient information",
        "not contain this information",
        "does not contain",
        "do not contain",
        "no information found",
        "no information available",
        "no information was found",
        "information not found",
        "information gap",
        "not found in",
        "question status:** **unanswered",
    )

    @classmethod
    def _is_negative_answer(cls, answer: str) -> bool:
        """True if *answer* is an unanswerable / hedged boilerplate response."""
        if not answer:
            return True
        lower = answer.lower()
        return any(pat in lower for pat in cls._NEGATIVE_ANSWER_PATTERNS)

    def _format_results_with_priority(
        self,
        results: list[DocumentResult],
        title_scores: dict[str, float] | None = None,
        question: str = "",
    ) -> str:
        """Format per-document results with relevance labels and confidence.

        When title scores are tied (or nearly tied for generic-reference
        questions), a lightweight answer-relevance heuristic (keyword
        overlap between the question and each answer) is used to break
        the tie so the synthesis LLM sees the most relevant answer first
        and clearly labelled.

        If multiple substantive results share the same (or near) title
        score, they are labelled "TIED CANDIDATE" so the synthesis LLM
        must decide which answer best addresses the question on semantic
        merit rather than being biased by a misleading "PRIMARY SOURCE"
        label.
        """
        if not results:
            return ""

        def _answer_relevance(answer: str, q: str) -> float:
            """Score how directly an answer addresses the question words."""
            if not q or not answer:
                return 0.0
            q_words = {
                w for w in re.sub(r"[^\w\s]", "", q.lower()).split()
                if len(w) > 2 and w not in _STOP_WORDS
            }
            a_lower = answer.lower()
            if not q_words:
                return 0.0
            return sum(1 for w in q_words if w in a_lower) / len(q_words)

        def _sort_key(r: DocumentResult) -> tuple[float, float, float]:
            ts = title_scores.get(r.doc_id, 0) if title_scores else 0
            ar = _answer_relevance(r.answer, question)
            return (ts, ar, r.confidence)

        scored = sorted(results, key=_sort_key, reverse=True)

        top_ts = title_scores.get(scored[0].doc_id, 0) if title_scores and scored else 0

        tie_threshold = 1.0 if self._GENERIC_REF_RE.search(question) else 0.0

        tied_at_top = sum(
            1 for r in scored
            if title_scores
            and abs(title_scores.get(r.doc_id, 0) - top_ts) <= tie_threshold
            and title_scores.get(r.doc_id, 0) > 0
        ) if title_scores else 0

        parts: list[str] = []
        for r in scored:
            ts = title_scores.get(r.doc_id, 0) if title_scores else 0
            near_top = abs(ts - top_ts) <= tie_threshold
            if tied_at_top > 1 and near_top and ts > 0:
                label = "TIED CANDIDATE — evaluate answer quality"
            elif near_top and top_ts > 0 and tied_at_top <= 1:
                label = "PRIMARY SOURCE (highest relevance)"
            elif title_scores and ts > 0:
                label = "SUPPORTING SOURCE"
            else:
                label = "Document"
            conf_tag = f" [confidence: {r.confidence:.2f}]" if r.confidence > 0 else ""
            parts.append(
                f"{label}: {r.doc_title}{conf_tag}\nFindings: {r.answer}"
            )
        return "\n\n".join(parts)

    def _synthesize_answer(
        self,
        question: str,
        query: CrossDocQuery,
        results: list[DocumentResult],
        doc_entities: dict[str, list[Entity]],
        *,
        title_scores: dict[str, float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> CrossDocAnswer:
        """
        Synthesize the final cross-document answer.

        Results are ordered by title relevance so the most relevant
        document appears first and is tagged as PRIMARY SOURCE.

        Args:
            question: Original question.
            query: Analyzed query.
            results: Per-document results.
            doc_entities: Entities by document.
            title_scores: Per-document title relevance scores.

        Returns:
            Final CrossDocAnswer.
        """
        if not results:
            return CrossDocAnswer(
                query=question,
                answer="No relevant documents found for this query.",
                confidence=0.0,
            )

        useful_results = [
            r for r in results if not self._is_negative_answer(r.answer)
        ]
        if not useful_results:
            useful_results = results

        def _answer_relevance(answer: str) -> float:
            q_words = {
                w for w in re.sub(r"[^\w\s]", "", question.lower()).split()
                if len(w) > 2 and w not in _STOP_WORDS
            }
            if not q_words or not answer:
                return 0.0
            a_lower = answer.lower()
            return sum(1 for w in q_words if w in a_lower) / len(q_words)

        useful_results = sorted(
            useful_results,
            key=lambda r: (
                title_scores.get(r.doc_id, 0) if title_scores else 0,
                _answer_relevance(r.answer),
                r.confidence,
            ),
            reverse=True,
        )

        all_entities = []
        for entities in doc_entities.values():
            all_entities.extend(entities)

        relationships: list[Relationship] = []
        entity_ids = {e.id for e in all_entities}
        for entity_id in entity_ids:
            rels = self.kg.get_entity_relationships(entity_id)
            for rel in rels:
                if rel.target_id in entity_ids or rel.source_id in entity_ids:
                    if rel not in relationships:
                        relationships.append(rel)

        entity_context = self._build_entity_context(
            all_entities, relationships, doc_entities,
        )

        best_confidence = max(r.confidence for r in results) if results else 0.0

        if not self._llm_fn:
            answer = self._simple_synthesis(question, useful_results)
        elif query.query_type == "comparison":
            answer = self._synthesize_comparison(
                question, useful_results, entity_context,
                title_scores=title_scores,
                conversation_context=conversation_context,
            )
        elif query.query_type == "timeline":
            answer = self._synthesize_timeline(
                question, useful_results, all_entities, entity_context,
                title_scores=title_scores,
                conversation_context=conversation_context,
            )
        elif query.query_type == "entity_tracking":
            answer = self._synthesize_entity_tracking(
                question, useful_results, all_entities, entity_context,
                title_scores=title_scores,
                conversation_context=conversation_context,
            )
        else:
            answer = self._synthesize_general(
                question, useful_results, entity_context,
                title_scores=title_scores,
                conversation_context=conversation_context,
            )
        
        total_nodes = sum(r.nodes_visited for r in results)
        total_iters = sum(r.iterations for r in results)

        return CrossDocAnswer(
            query=question,
            answer=answer,
            document_results=results,
            entities_involved=list({e.id: e for e in all_entities}.values()),
            relationships_used=relationships,
            confidence=best_confidence,
            total_nodes_visited=total_nodes,
            total_iterations=total_iters,
        )

    def _build_entity_context(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        doc_entities: dict[str, list[Entity]],
    ) -> str:
        """Build a textual KG context block for the synthesis prompt.

        Includes entity-to-document mapping, entity relationships, and
        co-mention information so the LLM can disambiguate conflicting
        answers across documents.
        """
        lines: list[str] = []

        # Map entities to their documents (using human-readable titles)
        doc_entity_map: dict[str, list[str]] = {}
        for doc_id, ents in doc_entities.items():
            title = self._get_doc_title(doc_id)
            doc_entity_map[title] = [e.canonical_name for e in ents]
        if doc_entity_map:
            lines.append("Entity-Document mapping:")
            for title, names in doc_entity_map.items():
                lines.append(f"  - {title}: {', '.join(names)}")

        # Relationships
        if relationships:
            lines.append("Entity relationships:")
            seen: set[str] = set()
            for rel in relationships[:20]:
                key = f"{rel.source_id}|{rel.type.value}|{rel.target_id}"
                if key not in seen:
                    seen.add(key)
                    lines.append(
                        f"  - {rel.source_id} → {rel.type.value} → {rel.target_id}"
                    )

        # Co-mentions (entities that appear together)
        for entity in entities[:8]:
            co = self.kg.get_entities_mentioned_together(entity.id)
            if co:
                related = [e.canonical_name for e, _ in co[:5]]
                lines.append(
                    f"  - {entity.canonical_name} co-occurs with: {', '.join(related)}"
                )

        if not lines:
            return ""
        return "\n".join(lines)
    
    def _simple_synthesis(
        self,
        question: str,
        results: list[DocumentResult],
    ) -> str:
        """Simple synthesis without LLM."""
        parts = []
        for result in results:
            if result.answer:
                parts.append(f"**{result.doc_title}**:\n{result.answer}")
        return "\n\n".join(parts) if parts else "No answers found."

    def _format_conversation_context(
        self,
        conversation_context: list[dict[str, str]] | None,
    ) -> str:
        """Build a conversation context block for synthesis prompts.

        Includes the 3 most recent Q&A pairs plus a document coverage
        summary showing which documents have/haven't been discussed.
        """
        if not conversation_context:
            return ""
        recent = conversation_context[-3:]
        lines = ["\nConversation context (previous Q&A, most recent last):"]
        for pair in recent:
            lines.append(f"  Q: {pair['question']}")
            lines.append(f"  A: {pair['answer'][:150]}")
            if pair.get("source_document"):
                lines.append(f"  (from: {pair['source_document']})")

        coverage = self._build_document_coverage(conversation_context)
        if coverage:
            lines.append(coverage)
        return "\n".join(lines) + "\n"

    def _build_document_coverage(
        self,
        conversation_context: list[dict[str, str]] | None,
    ) -> str:
        """Summarise which documents have/haven't been discussed yet.

        Used both in query analysis and synthesis prompts so the LLM
        knows which documents are fresh and which have already been
        covered by prior Q&A pairs.
        """
        if not conversation_context or not self._doc_titles:
            return ""

        discussed_titles: dict[str, int] = {}
        for pair in conversation_context:
            src = pair.get("source_document", "")
            if src:
                discussed_titles[src] = discussed_titles.get(src, 0) + 1

        all_titles = set(self._doc_titles.values())

        discussed_matched: set[str] = set()
        for title in all_titles:
            t_lower = title.lower()
            for src_title in discussed_titles:
                s_lower = src_title.lower()
                if s_lower in t_lower or t_lower in s_lower:
                    discussed_matched.add(title)
                    break

        not_discussed = all_titles - discussed_matched

        if not discussed_matched and not not_discussed:
            return ""

        lines = ["Document coverage in this conversation:"]
        if discussed_matched:
            d_list = ", ".join(sorted(discussed_matched))
            lines.append(f"  ALREADY DISCUSSED: {d_list}")
        if not_discussed:
            nd_list = ", ".join(sorted(not_discussed))
            lines.append(f"  NOT YET DISCUSSED: {nd_list}")
        lines.append(
            "When the question uses generic references, prefer answering "
            "from NOT YET DISCUSSED documents."
        )
        return "\n".join(lines)
    
    _CROSS_DOC_RULES = """RULES (STRICTLY FOLLOW ALL):
1. Give a DIRECT, CONCISE answer — start with the answer itself, not analysis or preamble.
2. FORBIDDEN: markdown headers (#, ##), section titles, bullet lists, "Overview", "Summary", "Conclusion" sections. Write plain prose only.
3. Do NOT hedge or say "cannot be determined" when the information IS present in the findings.
4. If multiple documents provide the same answer, state it once — do not repeat per document.
5. When the question asks about a SPECIFIC item (e.g. "the reference number on the letter", "the recipient's address") and multiple documents contain similar items, choose the answer from the document that the question is most likely referring to. Consider:
   - Which document the PRIMARY SOURCE label points to.
   - Whether the question specifies any distinguishing details (dates, names, types).
   - The conversation context: if prior questions established a specific document as the topic, the current question likely refers to that same document UNLESS the question introduces new content that better matches a different document.
   - Give ONE definitive answer, not multiple alternatives.
6. When documents DISAGREE, apply these tiebreakers IN ORDER:
   a. Prefer the source labelled "PRIMARY SOURCE (highest relevance)" — it was selected by relevance scoring.
   b. When sources are labelled "TIED CANDIDATE", you MUST judge which answer most directly and specifically answers the question. Ignore confidence scores for tied candidates — they only measure how easy the answer was to find, NOT how correct it is. Instead, prefer the answer that:
      - Is SPECIFIC and CONCRETE over one that is GENERIC or SELF-REFERENTIAL. For example, "That my driver licence disqualification is removed" is a specific order, while "orders in terms of the draft Consent Orders" is self-referential (it just says "the orders that are in the orders document") and gives no substantive information.
      - States a specific rule, form, provision, fact, name, amount, or date that IS the answer, over one that references these things generically or procedurally.
      - Directly addresses the question's intent rather than providing procedural or administrative context from a different document type.
      - Comes from the document whose type/purpose most closely matches what the question is asking about.
   c. Do NOT pick an answer just because it has higher confidence or because more documents mention it.
7. Keep the answer under 3 sentences unless the question requires more detail.
8. NEVER wrap the answer in "Entity Tracking", "Timeline", "Analysis", "Comprehensive", or any report-style formatting.
9. Ignore any formatting in the document findings — rewrite in plain, concise prose.
10. DOCUMENT PROGRESSION: When the conversation context shows that previous questions were answered from specific documents and the "Document coverage" section lists documents as NOT YET DISCUSSED, consider whether the current question is about an undiscussed document. Generic references ("the document", "the letter") after a series of questions from the same document often indicate a shift to a new, undiscussed document. Prefer answers from NOT YET DISCUSSED documents when the question is generic.
11. METADATA CONTINUATION: When consecutive questions ask about properties of the same correspondence (e.g. Q1: "What is the date?", Q2: "What is the name of the firm?", Q3: "What is the reference number?"), they ALL refer to the SAME document. Use conversation context to identify which document was the source for the most recent answer and continue using that document. Do NOT switch to a different document for follow-up metadata questions.
12. THEMATIC LETTER SELECTION: When transitioning from one document type to another (e.g. costs agreement → "the letter") and multiple letters exist, prefer the letter whose topic is most thematically connected to what was just discussed. For example, after billing/costs discussions, an "Invoice Cover Letter" is more relevant than a "Survey letter"."""

    def _synthesize_comparison(
        self,
        question: str,
        results: list[DocumentResult],
        entity_context: str = "",
        *,
        title_scores: dict[str, float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """Synthesize a comparison answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = self._format_results_with_priority(results, title_scores, question)
        conv_block = self._format_conversation_context(conversation_context)

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question by comparing information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}{conv_block}
Question: {question}

Document findings:
{results_text}

Answer:"""

        try:
            return self._llm_fn(prompt)
        except Exception as e:
            return f"Error: {str(e)}\n\n{self._simple_synthesis(question, results)}"

    def _synthesize_timeline(
        self,
        question: str,
        results: list[DocumentResult],
        entities: list[Entity],
        entity_context: str = "",
        *,
        title_scores: dict[str, float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """Synthesize a timeline answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = self._format_results_with_priority(results, title_scores, question)
        conv_block = self._format_conversation_context(conversation_context)

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}{conv_block}
Question: {question}

Document findings:
{results_text}

Answer:"""

        try:
            return self._llm_fn(prompt)
        except Exception as e:
            return f"Error: {str(e)}\n\n{self._simple_synthesis(question, results)}"

    def _synthesize_entity_tracking(
        self,
        question: str,
        results: list[DocumentResult],
        entities: list[Entity],
        entity_context: str = "",
        *,
        title_scores: dict[str, float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """Synthesize an entity tracking answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = self._format_results_with_priority(results, title_scores, question)
        conv_block = self._format_conversation_context(conversation_context)

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}{conv_block}
Question: {question}

Document findings:
{results_text}

Answer:"""

        try:
            return self._llm_fn(prompt)
        except Exception as e:
            return f"Error: {str(e)}\n\n{self._simple_synthesis(question, results)}"

    def _synthesize_general(
        self,
        question: str,
        results: list[DocumentResult],
        entity_context: str = "",
        *,
        title_scores: dict[str, float] | None = None,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> str:
        """Synthesize a general cross-document answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = self._format_results_with_priority(results, title_scores, question)
        conv_block = self._format_conversation_context(conversation_context)

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}{conv_block}
Question: {question}

Document findings:
{results_text}

Answer:"""

        try:
            return self._llm_fn(prompt)
        except Exception as e:
            return f"Error: {str(e)}\n\n{self._simple_synthesis(question, results)}"


# =============================================================================
# Factory Functions
# =============================================================================


def create_cross_doc_navigator(
    knowledge_graph: KnowledgeGraph,
) -> CrossDocNavigator:
    """
    Create a cross-document navigator.
    
    Args:
        knowledge_graph: Knowledge graph with entities.
        
    Returns:
        Configured CrossDocNavigator.
    """
    navigator = CrossDocNavigator(knowledge_graph)
    
    # Configure LLM
    try:
        from rnsr.llm import get_llm
        llm = get_llm()
        navigator.set_llm_function(lambda p: str(llm.complete(p)))
    except Exception as e:
        logger.warning("llm_config_failed", error=str(e))
    
    return navigator
