"""
RNSR Entity Linker

Cross-document entity linking with fuzzy matching and LLM disambiguation.
Links entities that represent the same real-world entity across documents.

Features adaptive learning for normalization patterns - learns new titles
and suffixes from user's document workload.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

import structlog

from rnsr.extraction.models import Entity, EntityLink, EntityType
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.indexing.knowledge_graph import KnowledgeGraph

logger = structlog.get_logger(__name__)


# =============================================================================
# Learned Normalization Patterns
# =============================================================================

DEFAULT_NORMALIZATION_PATH = Path.home() / ".rnsr" / "learned_normalization.json"


class LearnedNormalizationPatterns:
    """
    Registry for learning domain-specific normalization patterns.
    
    Learns:
    - Titles/prefixes (Mr., Dr., Esq., Hon., M.D., etc.)
    - Suffixes (Inc., LLC, GmbH, Pty Ltd, etc.)
    - Domain-specific patterns (legal, medical, regional)
    """
    
    # Base patterns (always included)
    BASE_TITLES = [
        "mr.", "mrs.", "ms.", "dr.", "prof.",
        "mr", "mrs", "ms", "dr", "prof",
        "the", "hon.", "hon", "sir", "dame",
    ]
    
    BASE_SUFFIXES = [
        ", inc.", ", inc", ", llc", ", llc.",
        ", corp.", ", corp", ", ltd.", ", ltd",
        "inc.", "inc", "llc", "corp.", "corp", "ltd.", "ltd",
        ", esq.", ", esq", "esq.", "esq",
        ", jr.", ", jr", ", sr.", ", sr",
        "jr.", "jr", "sr.", "sr",
    ]
    
    def __init__(
        self,
        storage_path: Path | str | None = None,
        auto_save: bool = True,
    ):
        """
        Initialize the normalization patterns registry.
        
        Args:
            storage_path: Path to JSON file for persistence.
            auto_save: Whether to save after each new pattern.
        """
        self.storage_path = Path(storage_path) if storage_path else DEFAULT_NORMALIZATION_PATH
        self.auto_save = auto_save
        
        self._lock = Lock()
        self._titles: dict[str, dict[str, Any]] = {}
        self._suffixes: dict[str, dict[str, Any]] = {}
        self._dirty = False
        
        self._load()
    
    def _load(self) -> None:
        """Load learned patterns from storage."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._titles = data.get("titles", {})
            self._suffixes = data.get("suffixes", {})
            
            logger.info(
                "normalization_patterns_loaded",
                titles=len(self._titles),
                suffixes=len(self._suffixes),
            )
            
        except Exception as e:
            logger.warning("failed_to_load_normalization_patterns", error=str(e))
    
    def _save(self) -> None:
        """Save patterns to storage."""
        if not self._dirty:
            return
        
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "titles": self._titles,
                "suffixes": self._suffixes,
            }
            
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            
            self._dirty = False
            
        except Exception as e:
            logger.warning("failed_to_save_normalization_patterns", error=str(e))
    
    def record_title(
        self,
        title: str,
        domain: str = "general",
        entity_example: str = "",
    ) -> None:
        """
        Record a learned title/prefix.
        
        Args:
            title: The title pattern (e.g., "Atty.", "M.D.").
            domain: Domain category (legal, medical, regional, etc.).
            entity_example: Example entity with this title.
        """
        title = title.lower().strip()
        
        if not title or title in self.BASE_TITLES:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if title not in self._titles:
                self._titles[title] = {
                    "count": 0,
                    "domain": domain,
                    "first_seen": now,
                    "last_seen": now,
                    "examples": [],
                }
                logger.info("new_title_pattern_learned", title=title, domain=domain)
            
            self._titles[title]["count"] += 1
            self._titles[title]["last_seen"] = now
            
            if entity_example and len(self._titles[title]["examples"]) < 3:
                self._titles[title]["examples"].append(entity_example)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def record_suffix(
        self,
        suffix: str,
        domain: str = "general",
        entity_example: str = "",
    ) -> None:
        """
        Record a learned suffix.
        
        Args:
            suffix: The suffix pattern (e.g., "GmbH", "Pty Ltd").
            domain: Domain category (legal, corporate, regional, etc.).
            entity_example: Example entity with this suffix.
        """
        suffix = suffix.lower().strip()
        
        if not suffix or suffix in self.BASE_SUFFIXES:
            return
        
        with self._lock:
            now = datetime.utcnow().isoformat()
            
            if suffix not in self._suffixes:
                self._suffixes[suffix] = {
                    "count": 0,
                    "domain": domain,
                    "first_seen": now,
                    "last_seen": now,
                    "examples": [],
                }
                logger.info("new_suffix_pattern_learned", suffix=suffix, domain=domain)
            
            self._suffixes[suffix]["count"] += 1
            self._suffixes[suffix]["last_seen"] = now
            
            if entity_example and len(self._suffixes[suffix]["examples"]) < 3:
                self._suffixes[suffix]["examples"].append(entity_example)
            
            self._dirty = True
            
            if self.auto_save:
                self._save()
    
    def get_all_titles(self, min_count: int = 1) -> list[str]:
        """Get all titles (base + learned)."""
        learned = [
            title for title, data in self._titles.items()
            if data["count"] >= min_count
        ]
        return list(set(self.BASE_TITLES + learned))
    
    def get_all_suffixes(self, min_count: int = 1) -> list[str]:
        """Get all suffixes (base + learned)."""
        learned = [
            suffix for suffix, data in self._suffixes.items()
            if data["count"] >= min_count
        ]
        return list(set(self.BASE_SUFFIXES + learned))
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about learned patterns."""
        return {
            "learned_titles": len(self._titles),
            "learned_suffixes": len(self._suffixes),
            "total_titles": len(self.get_all_titles()),
            "total_suffixes": len(self.get_all_suffixes()),
        }


# Global normalization patterns instance
_global_normalization_patterns: LearnedNormalizationPatterns | None = None


def get_learned_normalization_patterns() -> LearnedNormalizationPatterns:
    """Get the global normalization patterns registry."""
    global _global_normalization_patterns
    
    if _global_normalization_patterns is None:
        custom_path = os.getenv("RNSR_NORMALIZATION_PATH")
        _global_normalization_patterns = LearnedNormalizationPatterns(
            storage_path=custom_path if custom_path else None
        )
    
    return _global_normalization_patterns


# LLM disambiguation prompt
DISAMBIGUATION_PROMPT = """You are an expert at entity resolution. Determine if these two entities refer to the same real-world entity.

Entity 1:
- Name: {name1}
- Type: {type1}
- Aliases: {aliases1}
- Context: {context1}
- Document: {doc1}

Entity 2:
- Name: {name2}
- Type: {type2}
- Aliases: {aliases2}
- Context: {context2}
- Document: {doc2}

Consider:
1. Name similarity (accounting for variations, titles, abbreviations)
2. Context similarity (same role, same events, same relationships)
3. Document context (are these documents related?)

Respond with JSON:
```json
{{
  "same_entity": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of your decision"
}}
```
"""


class EntityLinker:
    """
    Links entities across documents using multiple strategies:
    
    1. Exact match on canonical_name
    2. Fuzzy string matching (Levenshtein-based)
    3. Alias matching
    4. LLM disambiguation for ambiguous cases
    """
    
    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm: Any | None = None,
        exact_match_threshold: float = 1.0,
        fuzzy_match_threshold: float = 0.85,
        llm_confidence_threshold: float = 0.75,
        use_llm_disambiguation: bool = True,
        enable_pattern_learning: bool = True,
    ):
        """
        Initialize the entity linker.
        
        Args:
            knowledge_graph: Knowledge graph for storing/querying entities.
            llm: LLM instance for disambiguation. If None, uses get_llm().
            exact_match_threshold: Confidence for exact matches.
            fuzzy_match_threshold: Minimum similarity for fuzzy matching.
            llm_confidence_threshold: Minimum LLM confidence to accept.
            use_llm_disambiguation: Whether to use LLM for ambiguous cases.
            enable_pattern_learning: Learn new normalization patterns.
        """
        self.kg = knowledge_graph
        self.llm = llm
        self.exact_match_threshold = exact_match_threshold
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.llm_confidence_threshold = llm_confidence_threshold
        self.use_llm_disambiguation = use_llm_disambiguation
        self.enable_pattern_learning = enable_pattern_learning
        
        # Lazy LLM initialization
        self._llm_initialized = False
        
        # Normalization patterns registry
        self._normalization_patterns = get_learned_normalization_patterns() if enable_pattern_learning else None
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM."""
        if self.llm is None and not self._llm_initialized:
            self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def link_entities(
        self,
        new_entities: list[Entity],
        target_doc_id: str | None = None,
    ) -> list[EntityLink]:
        """
        Link new entities to existing entities in the knowledge graph.
        
        Args:
            new_entities: Newly extracted entities to link.
            target_doc_id: If provided, only link to entities in this document.
            
        Returns:
            List of EntityLink objects for confirmed links.
        """
        links = []
        
        for entity in new_entities:
            # Find candidates from knowledge graph
            candidates = self._find_candidates(entity, target_doc_id)
            
            if not candidates:
                continue
            
            # Score each candidate
            scored_candidates = []
            for candidate in candidates:
                score, method = self._score_match(entity, candidate)
                if score >= self.fuzzy_match_threshold:
                    scored_candidates.append((candidate, score, method))
            
            # Sort by score
            scored_candidates.sort(key=lambda x: -x[1])
            
            if not scored_candidates:
                continue
            
            # Handle disambiguation
            best_candidate, best_score, best_method = scored_candidates[0]
            
            if best_score >= self.exact_match_threshold:
                # Exact match - link directly
                link = EntityLink(
                    entity_id_1=entity.id,
                    entity_id_2=best_candidate.id,
                    confidence=best_score,
                    link_method=best_method,
                    evidence=f"Matched on: {entity.canonical_name} = {best_candidate.canonical_name}",
                )
                links.append(link)
                
                # Store in knowledge graph
                self.kg.link_entities(
                    entity.id,
                    best_candidate.id,
                    confidence=best_score,
                    link_method=best_method,
                    evidence=link.evidence,
                )
            
            elif len(scored_candidates) == 1 and best_score >= self.fuzzy_match_threshold:
                # Single fuzzy match - link with lower confidence
                link = EntityLink(
                    entity_id_1=entity.id,
                    entity_id_2=best_candidate.id,
                    confidence=best_score,
                    link_method=best_method,
                    evidence=f"Fuzzy match: {entity.canonical_name} ~ {best_candidate.canonical_name}",
                )
                links.append(link)
                
                self.kg.link_entities(
                    entity.id,
                    best_candidate.id,
                    confidence=best_score,
                    link_method=best_method,
                    evidence=link.evidence,
                )
            
            elif self.use_llm_disambiguation and len(scored_candidates) >= 1:
                # Ambiguous - use LLM
                llm_link = self._disambiguate_with_llm(entity, scored_candidates)
                if llm_link:
                    links.append(llm_link)
                    
                    self.kg.link_entities(
                        llm_link.entity_id_1,
                        llm_link.entity_id_2,
                        confidence=llm_link.confidence,
                        link_method=llm_link.link_method,
                        evidence=llm_link.evidence,
                    )
        
        logger.info(
            "entity_linking_complete",
            new_entities=len(new_entities),
            links_created=len(links),
        )
        
        return links
    
    def _find_candidates(
        self,
        entity: Entity,
        target_doc_id: str | None = None,
    ) -> list[Entity]:
        """
        Find candidate entities that might match the given entity.
        
        Args:
            entity: Entity to find matches for.
            target_doc_id: Optional document ID filter.
            
        Returns:
            List of candidate entities.
        """
        candidates = []
        
        # Search by exact name
        exact_matches = self.kg.find_entities_by_name(
            entity.canonical_name,
            entity_type=entity.type,
            fuzzy=False,
        )
        candidates.extend(exact_matches)
        
        # Search by fuzzy name
        fuzzy_matches = self.kg.find_entities_by_name(
            entity.canonical_name,
            entity_type=entity.type,
            fuzzy=True,
        )
        for match in fuzzy_matches:
            if match.id not in {c.id for c in candidates}:
                candidates.append(match)
        
        # Search by aliases
        for alias in entity.aliases:
            alias_matches = self.kg.find_entities_by_name(
                alias,
                entity_type=entity.type,
                fuzzy=True,
            )
            for match in alias_matches:
                if match.id not in {c.id for c in candidates}:
                    candidates.append(match)
        
        # Filter out the entity itself
        candidates = [c for c in candidates if c.id != entity.id]
        
        # Filter by target document if specified
        if target_doc_id:
            candidates = [
                c for c in candidates
                if target_doc_id in c.document_ids
            ]
        
        return candidates
    
    def _score_match(
        self,
        entity1: Entity,
        entity2: Entity,
    ) -> tuple[float, str]:
        """
        Score the similarity between two entities.
        
        Args:
            entity1: First entity.
            entity2: Second entity.
            
        Returns:
            Tuple of (score, method) where score is 0.0-1.0.
        """
        # Type mismatch is a strong negative signal
        if entity1.type != entity2.type:
            return 0.0, "type_mismatch"
        
        # Exact canonical name match
        if entity1.canonical_name.lower().strip() == entity2.canonical_name.lower().strip():
            return 1.0, "exact"
        
        # Check alias matches
        all_names1 = {n.lower().strip() for n in entity1.all_names}
        all_names2 = {n.lower().strip() for n in entity2.all_names}
        
        if all_names1 & all_names2:
            return 0.95, "alias"
        
        # Fuzzy string matching on all name combinations
        best_similarity = 0.0
        for name1 in entity1.all_names:
            for name2 in entity2.all_names:
                similarity = self._string_similarity(name1, name2)
                best_similarity = max(best_similarity, similarity)
        
        if best_similarity >= self.fuzzy_match_threshold:
            return best_similarity, "fuzzy"
        
        # Name containment (one name contains the other)
        name1_lower = entity1.canonical_name.lower()
        name2_lower = entity2.canonical_name.lower()
        
        if name1_lower in name2_lower or name2_lower in name1_lower:
            # Score based on length ratio
            shorter = min(len(name1_lower), len(name2_lower))
            longer = max(len(name1_lower), len(name2_lower))
            return 0.7 + 0.2 * (shorter / longer), "containment"
        
        return best_similarity, "low_similarity"
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """
        Calculate string similarity using SequenceMatcher.
        
        Args:
            s1: First string.
            s2: Second string.
            
        Returns:
            Similarity score 0.0-1.0.
        """
        # Normalize strings
        s1 = self._normalize_name(s1)
        s2 = self._normalize_name(s2)
        
        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _normalize_name(self, name: str, learn_patterns: bool = True) -> str:
        """
        Normalize a name for comparison.
        
        Uses both base patterns and learned patterns from the registry.
        
        Args:
            name: Name to normalize.
            learn_patterns: Whether to record new patterns found.
            
        Returns:
            Normalized name.
        """
        original_name = name
        
        # Lowercase
        name = name.lower()
        
        # Get titles and suffixes (base + learned)
        if self._normalization_patterns:
            titles = self._normalization_patterns.get_all_titles()
            suffixes = self._normalization_patterns.get_all_suffixes()
        else:
            titles = LearnedNormalizationPatterns.BASE_TITLES
            suffixes = LearnedNormalizationPatterns.BASE_SUFFIXES
        
        # Remove titles/prefixes
        removed_title = None
        for title in sorted(titles, key=len, reverse=True):  # Longest first
            if name.startswith(title + " "):
                removed_title = title
                name = name[len(title) + 1:]
                break
        
        # Remove suffixes
        removed_suffix = None
        for suffix in sorted(suffixes, key=len, reverse=True):  # Longest first
            if name.endswith(suffix):
                removed_suffix = suffix
                name = name[:-len(suffix)]
                break
        
        # Learn new patterns if enabled
        if learn_patterns and self._normalization_patterns and self.enable_pattern_learning:
            # Detect potential new titles (patterns like "X. Name" or "X Name")
            title_match = re.match(r'^([A-Za-z]{1,4}\.?)\s+[A-Z]', original_name)
            if title_match and not removed_title:
                potential_title = title_match.group(1).lower()
                if potential_title not in titles and len(potential_title) >= 2:
                    self._normalization_patterns.record_title(
                        potential_title,
                        domain="detected",
                        entity_example=original_name,
                    )
            
            # Detect potential new suffixes (patterns at end of company names)
            suffix_patterns = [
                r',?\s*(gmbh|ag|s\.a\.|pty\s*ltd|plc|bv|nv|asa|ab|as|oy|a/s|k\.k\.|co\.,?\s*ltd\.?)$',
                r',?\s*([A-Z]{2,4}\.?)$',  # Acronym suffixes
            ]
            for pattern in suffix_patterns:
                suffix_match = re.search(pattern, original_name, re.IGNORECASE)
                if suffix_match and not removed_suffix:
                    potential_suffix = suffix_match.group(1).lower()
                    if potential_suffix not in suffixes:
                        self._normalization_patterns.record_suffix(
                            potential_suffix,
                            domain="detected",
                            entity_example=original_name,
                        )
        
        # Normalize whitespace
        name = " ".join(name.split())
        
        return name.strip()
    
    def _disambiguate_with_llm(
        self,
        entity: Entity,
        candidates: list[tuple[Entity, float, str]],
    ) -> EntityLink | None:
        """
        Use LLM to disambiguate between multiple candidates.
        
        Args:
            entity: Entity to match.
            candidates: List of (candidate, score, method) tuples.
            
        Returns:
            EntityLink if a match is found, None otherwise.
        """
        llm = self._get_llm()
        if llm is None:
            return None
        
        # Get context from first mention
        entity_context = ""
        if entity.mentions:
            entity_context = entity.mentions[0].context
        
        # Try each candidate
        for candidate, score, method in candidates:
            candidate_context = ""
            if candidate.mentions:
                candidate_context = candidate.mentions[0].context
            
            prompt = DISAMBIGUATION_PROMPT.format(
                name1=entity.canonical_name,
                type1=entity.type.value,
                aliases1=", ".join(entity.aliases) or "None",
                context1=entity_context or "No context available",
                doc1=entity.source_doc_id or "Unknown",
                name2=candidate.canonical_name,
                type2=candidate.type.value,
                aliases2=", ".join(candidate.aliases) or "None",
                context2=candidate_context or "No context available",
                doc2=candidate.source_doc_id or "Unknown",
            )
            
            try:
                _complete_fn = getattr(llm, "complete_json", None) or llm.complete
                response = _complete_fn(prompt)
                response_text = str(response) if not isinstance(response, str) else response
                
                result = self._parse_disambiguation_response(response_text)
                
                if result and result.get("same_entity") and result.get("confidence", 0) >= self.llm_confidence_threshold:
                    return EntityLink(
                        entity_id_1=entity.id,
                        entity_id_2=candidate.id,
                        confidence=result["confidence"],
                        link_method="llm",
                        evidence=result.get("reasoning", "LLM disambiguation"),
                    )
            
            except Exception as e:
                logger.warning(
                    "llm_disambiguation_failed",
                    entity=entity.canonical_name,
                    candidate=candidate.canonical_name,
                    error=str(e),
                )
        
        return None
    
    def _parse_disambiguation_response(
        self,
        response_text: str,
    ) -> dict[str, Any] | None:
        """
        Parse LLM disambiguation response.
        
        Args:
            response_text: Raw LLM response.
            
        Returns:
            Parsed result dict or None.
        """
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    
    def link_all_entities_in_document(
        self,
        doc_id: str,
    ) -> list[EntityLink]:
        """
        Link all entities within a document to each other.
        
        Useful for finding co-references within a single document.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            List of EntityLink objects.
        """
        entities = self.kg.find_entities_in_document(doc_id)
        
        if len(entities) < 2:
            return []
        
        links = []
        
        # Group entities by type
        by_type: dict[EntityType, list[Entity]] = defaultdict(list)
        for entity in entities:
            by_type[entity.type].append(entity)
        
        # Link within each type group
        for entity_type, type_entities in by_type.items():
            for i, entity1 in enumerate(type_entities):
                for entity2 in type_entities[i + 1:]:
                    score, method = self._score_match(entity1, entity2)
                    
                    if score >= self.fuzzy_match_threshold:
                        link = EntityLink(
                            entity_id_1=entity1.id,
                            entity_id_2=entity2.id,
                            confidence=score,
                            link_method=method,
                            evidence=f"Same document co-reference: {entity1.canonical_name} = {entity2.canonical_name}",
                        )
                        links.append(link)
                        
                        self.kg.link_entities(
                            entity1.id,
                            entity2.id,
                            confidence=score,
                            link_method=method,
                            evidence=link.evidence,
                        )
        
        return links
    
    def link_across_documents(
        self,
        doc_id_1: str,
        doc_id_2: str,
    ) -> list[EntityLink]:
        """
        Link entities between two specific documents.
        
        Args:
            doc_id_1: First document ID.
            doc_id_2: Second document ID.
            
        Returns:
            List of EntityLink objects.
        """
        entities_1 = self.kg.find_entities_in_document(doc_id_1)
        entities_2 = self.kg.find_entities_in_document(doc_id_2)
        
        if not entities_1 or not entities_2:
            return []
        
        # Use the main linking method
        return self.link_entities(entities_1, target_doc_id=doc_id_2)
