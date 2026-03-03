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

Extract:
1. People mentioned (names, roles)
2. Organizations mentioned
3. Documents or sections referenced
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
    
    def query(self, question: str) -> CrossDocAnswer:
        """
        Execute a cross-document query.
        
        Args:
            question: The user's question.
            
        Returns:
            CrossDocAnswer with synthesized result.
        """
        trace = []
        
        # Step 1: Extract entities from query
        trace.append({
            "step": "extract_entities",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        
        query_analysis = self._analyze_query(question)
        
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
        
        retrieval_plan = self._plan_retrieval(question, query_analysis, doc_entities)
        
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
        
        answer = self._synthesize_answer(
            question,
            query_analysis,
            document_results,
            doc_entities,
        )
        
        answer.trace = trace
        
        logger.info(
            "cross_doc_query_complete",
            query=question[:100],
            documents=len(document_results),
            confidence=answer.confidence,
        )
        
        return answer
    
    def _analyze_query(self, question: str) -> CrossDocQuery:
        """
        Analyze the query to extract entities and determine query type.
        
        Args:
            question: The user's question.
            
        Returns:
            CrossDocQuery with extracted information.
        """
        result = CrossDocQuery(original_query=question)
        
        if not self._llm_fn:
            # Basic extraction without LLM
            result.query_type = "general"
            return result
        
        try:
            prompt = QUERY_ENTITY_EXTRACTION_PROMPT.format(query=question)
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
    
    def _plan_retrieval(
        self,
        question: str,
        query: CrossDocQuery,
        doc_entities: dict[str, list[Entity]],
    ) -> list[dict[str, Any]]:
        """
        Plan the retrieval strategy for each document.

        Documents with entity matches get entity-focused sub-queries.
        All other registered documents are still searched with the
        original question so that no document is silently skipped.
        
        Args:
            question: Original question.
            query: Analyzed query.
            doc_entities: Entities by document.
            
        Returns:
            List of retrieval tasks.
        """
        tasks = []
        planned_doc_ids: set[str] = set()
        
        for doc_id, entities in doc_entities.items():
            entity_names = [e.canonical_name for e in entities]
            
            # Always keep the original question so key terms (e.g. "jaws of
            # life") are not dropped.  Add entity context as a hint only.
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
            
            tasks.append({
                "doc_id": doc_id,
                "sub_query": sub_query,
                "entities": entities,
                "target_nodes": list(target_nodes),
            })
            planned_doc_ids.add(doc_id)
        
        # For small document collections (<= 10 docs), query every
        # document — filtering is counterproductive when the collection
        # is small and the cost of querying all docs is low.
        all_registered = set(self.navigators.keys()) | set(self._kv_stores.keys())
        _SMALL_COLLECTION_THRESHOLD = 10
        skip_filtering = len(all_registered) <= _SMALL_COLLECTION_THRESHOLD

        for doc_id in all_registered - planned_doc_ids:
            if not skip_filtering and not self._doc_has_keyword_overlap(doc_id, question):
                logger.debug(
                    "fallback_doc_skipped",
                    doc_id=doc_id,
                    reason="no keyword overlap with query",
                )
                continue

            tasks.append({
                "doc_id": doc_id,
                "sub_query": question,
                "entities": [],
                "target_nodes": [],
            })
            logger.debug(
                "fallback_doc_included",
                doc_id=doc_id,
                reason="small collection — querying all docs" if skip_filtering
                       else "keyword overlap found, searching with original query",
            )
        
        return tasks

    _STOPWORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "it", "this", "that",
        "with", "from", "by", "as", "be", "been", "being", "have", "has",
        "had", "do", "does", "did", "will", "would", "could", "should",
        "what", "when", "where", "who", "how", "which", "there", "about",
    })

    def _doc_has_keyword_overlap(self, doc_id: str, question: str) -> bool:
        """Check if a document's skeleton has any keyword overlap with the query."""
        skeleton = self._skeletons.get(doc_id)
        if not skeleton:
            return True  # no metadata to filter on -- keep it safe

        query_words = {
            w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in self._STOPWORDS and len(w) > 2
        }
        if not query_words:
            return True

        doc_text_parts: list[str] = []
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
        
        for i, task in enumerate(tasks):
            doc_id = task["doc_id"]
            
            # Check if we have a navigator for this document
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
            # KG-first: try resolving before full navigation
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
                # Pass entity guidance to navigator
                nav_metadata: dict[str, Any] | None = None
                if resolution.entity_node_ids:
                    nav_metadata = {
                        "entity_priority_nodes": resolution.entity_node_ids
                    }
            else:
                nav_metadata = None

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

    def _synthesize_answer(
        self,
        question: str,
        query: CrossDocQuery,
        results: list[DocumentResult],
        doc_entities: dict[str, list[Entity]],
    ) -> CrossDocAnswer:
        """
        Synthesize the final cross-document answer.
        
        Args:
            question: Original question.
            query: Analyzed query.
            results: Per-document results.
            doc_entities: Entities by document.
            
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
            )
        elif query.query_type == "timeline":
            answer = self._synthesize_timeline(
                question, useful_results, all_entities, entity_context,
            )
        elif query.query_type == "entity_tracking":
            answer = self._synthesize_entity_tracking(
                question, useful_results, all_entities, entity_context,
            )
        else:
            answer = self._synthesize_general(
                question, useful_results, entity_context,
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
    
    _CROSS_DOC_RULES = """RULES (STRICTLY FOLLOW ALL):
1. Give a DIRECT, CONCISE answer — start with the answer itself, not analysis or preamble.
2. FORBIDDEN: markdown headers (#, ##), section titles, bullet lists, "Overview", "Summary", "Conclusion" sections. Write plain prose only.
3. Do NOT hedge or say "cannot be determined" when the information IS present in the findings.
4. If multiple documents provide the same answer, state it once — do not repeat per document.
5. When documents DISAGREE, use the Knowledge Graph context (entity relationships, document types, entity-document mapping) to determine the MOST CONTEXTUALLY RELEVANT answer. Do NOT pick an answer just because more documents mention it — frequency is NOT correctness. Consider which document type is most likely to contain the authoritative answer for this specific question.
6. Keep the answer under 3 sentences unless the question requires more detail.
7. NEVER wrap the answer in "Entity Tracking", "Timeline", "Analysis", "Comprehensive", or any report-style formatting.
8. Ignore any formatting in the document findings — rewrite in plain, concise prose."""

    def _synthesize_comparison(
        self,
        question: str,
        results: list[DocumentResult],
        entity_context: str = "",
    ) -> str:
        """Synthesize a comparison answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = "\n\n".join([
            f"Document: {r.doc_title}\nFindings: {r.answer}"
            for r in results
        ])

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question by comparing information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}
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
    ) -> str:
        """Synthesize a timeline answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = "\n\n".join([
            f"Document: {r.doc_title}\nFindings: {r.answer}"
            for r in results
        ])

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}
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
    ) -> str:
        """Synthesize an entity tracking answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = "\n\n".join([
            f"Document: {r.doc_title}\nFindings: {r.answer}"
            for r in results
        ])

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}
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
    ) -> str:
        """Synthesize a general cross-document answer."""
        if not self._llm_fn:
            return self._simple_synthesis(question, results)

        results_text = "\n\n".join([
            f"Document: {r.doc_title}\nFindings: {r.answer}"
            for r in results
        ])

        kg_block = ""
        if entity_context:
            kg_block = f"\nKnowledge Graph Context:\n{entity_context}\n"

        prompt = f"""Answer the question using information from multiple documents.

{self._CROSS_DOC_RULES}
{kg_block}
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
