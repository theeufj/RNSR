"""
RNSR Knowledge Graph - SQLite-Backed Graph Storage

Stores entities, relationships, and entity links for ontological
document understanding and cross-document queries.

Usage:
    kg = KnowledgeGraph("./data/knowledge_graph.db")
    kg.add_entity(entity)
    kg.add_relationship(relationship)
    
    # Query entities
    entities = kg.find_entities_by_name("John Smith")
    
    # Get entity relationships
    relationships = kg.get_entity_relationships("ent_abc123")
    
    # Cross-document entity linking
    kg.link_entities("ent_doc1_john", "ent_doc2_john", confidence=0.95)
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import structlog

from rnsr.extraction.models import (
    Entity,
    EntityLink,
    EntityType,
    Mention,
    Relationship,
    RelationType,
)

logger = structlog.get_logger(__name__)


class KnowledgeGraph:
    """
    SQLite-backed knowledge graph for entity and relationship storage.
    
    Supports:
    - Entity storage with mentions and aliases
    - Relationship storage between entities and nodes
    - Cross-document entity linking
    - Efficient querying by name, type, document, and node
    """
    
    def __init__(self, db_path: Path | str):
        """
        Initialize the knowledge graph.
        
        Args:
            db_path: Path to the SQLite database file, or ``:memory:`` for
                     an in-memory graph.
                     Will be created if it doesn't exist.
        """
        self._is_memory = str(db_path) == ":memory:"
        self.db_path = Path(db_path) if not self._is_memory else db_path  # type: ignore[assignment]
        self._persistent_conn: sqlite3.Connection | None = None

        if not self._is_memory:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
        
        logger.info("knowledge_graph_initialized", db_path=str(self.db_path))
    
    def _init_db(self) -> None:
        """Create the database schema if it doesn't exist."""
        with self._connect() as conn:
            # Entities table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    aliases TEXT,
                    metadata TEXT,
                    source_doc_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Mentions table (entity-to-node links)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mentions (
                    id TEXT PRIMARY KEY,
                    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                    node_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    context TEXT,
                    span_start INTEGER,
                    span_end INTEGER,
                    page_num INTEGER,
                    confidence REAL DEFAULT 1.0
                )
            """)
            
            # Relationships table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    source_type TEXT DEFAULT 'entity',
                    target_type TEXT DEFAULT 'entity',
                    doc_id TEXT,
                    confidence REAL DEFAULT 1.0,
                    evidence TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Entity links table (cross-document entity resolution)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entity_links (
                    entity_id_1 TEXT NOT NULL,
                    entity_id_2 TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    link_method TEXT DEFAULT 'exact',
                    evidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (entity_id_1, entity_id_2)
                )
            """)
            
            # Indexes for efficient querying
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_type 
                ON entities(type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_name 
                ON entities(canonical_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_entities_doc 
                ON entities(source_doc_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mentions_entity 
                ON mentions(entity_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mentions_node 
                ON mentions(node_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mentions_doc 
                ON mentions(doc_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_source 
                ON relationships(source_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_target 
                ON relationships(target_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_relationships_type 
                ON relationships(type)
            """)
            
            conn.commit()
    
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections.

        For ``:memory:`` databases a single persistent connection is reused
        so that the schema and data survive across calls.  For file-backed
        databases a fresh connection is opened (and closed) each time.
        """
        if self._is_memory:
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._persistent_conn.row_factory = sqlite3.Row
                self._persistent_conn.execute("PRAGMA foreign_keys = ON")
            yield self._persistent_conn
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
            finally:
                conn.close()
    
    # =========================================================================
    # Entity Operations
    # =========================================================================
    
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: Entity to add.
            
        Returns:
            Entity ID.
        """
        with self._connect() as conn:
            # Insert entity
            conn.execute(
                """
                INSERT INTO entities (id, type, canonical_name, aliases, metadata, source_doc_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    aliases = excluded.aliases,
                    metadata = excluded.metadata
                """,
                (
                    entity.id,
                    entity.type.value,
                    entity.canonical_name,
                    json.dumps(entity.aliases),
                    json.dumps(entity.metadata),
                    entity.source_doc_id,
                    entity.created_at.isoformat(),
                ),
            )
            
            # Insert mentions
            for mention in entity.mentions:
                conn.execute(
                    """
                    INSERT INTO mentions (id, entity_id, node_id, doc_id, context, span_start, span_end, page_num, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        context = excluded.context,
                        confidence = excluded.confidence
                    """,
                    (
                        mention.id,
                        entity.id,
                        mention.node_id,
                        mention.doc_id,
                        mention.context,
                        mention.span_start,
                        mention.span_end,
                        mention.page_num,
                        mention.confidence,
                    ),
                )
            
            conn.commit()
        
        logger.debug(
            "entity_added",
            entity_id=entity.id,
            type=entity.type.value,
            name=entity.canonical_name,
            mentions=len(entity.mentions),
        )
        
        return entity.id
    
    def get_entity(self, entity_id: str) -> Entity | None:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            Entity or None if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM entities WHERE id = ?",
                (entity_id,),
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            # Get mentions
            mentions_cursor = conn.execute(
                "SELECT * FROM mentions WHERE entity_id = ?",
                (entity_id,),
            )
            mentions = [self._row_to_mention(m) for m in mentions_cursor]
        
        return self._row_to_entity(row, mentions)
    
    def find_entities_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
        fuzzy: bool = False,
    ) -> list[Entity]:
        """
        Find entities by name.
        
        Args:
            name: Entity name to search for.
            entity_type: Optional type filter.
            fuzzy: If True, use LIKE matching.
            
        Returns:
            List of matching entities.
        """
        with self._connect() as conn:
            if fuzzy:
                name_pattern = f"%{name}%"
                if entity_type:
                    cursor = conn.execute(
                        """
                        SELECT * FROM entities 
                        WHERE (canonical_name LIKE ? OR aliases LIKE ?)
                        AND type = ?
                        """,
                        (name_pattern, name_pattern, entity_type.value),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM entities 
                        WHERE canonical_name LIKE ? OR aliases LIKE ?
                        """,
                        (name_pattern, name_pattern),
                    )
            else:
                if entity_type:
                    cursor = conn.execute(
                        """
                        SELECT * FROM entities 
                        WHERE canonical_name = ? AND type = ?
                        """,
                        (name, entity_type.value),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM entities WHERE canonical_name = ?",
                        (name,),
                    )
            
            entities = []
            for row in cursor:
                mentions_cursor = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ?",
                    (row["id"],),
                )
                mentions = [self._row_to_mention(m) for m in mentions_cursor]
                entities.append(self._row_to_entity(row, mentions))
        
        return entities
    
    def find_entities_by_type(
        self,
        entity_type: EntityType,
        doc_id: str | None = None,
    ) -> list[Entity]:
        """
        Find entities by type.
        
        Args:
            entity_type: Entity type to filter by.
            doc_id: Optional document ID filter.
            
        Returns:
            List of matching entities.
        """
        with self._connect() as conn:
            if doc_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM entities 
                    WHERE type = ? AND source_doc_id = ?
                    """,
                    (entity_type.value, doc_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM entities WHERE type = ?",
                    (entity_type.value,),
                )
            
            entities = []
            for row in cursor:
                mentions_cursor = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ?",
                    (row["id"],),
                )
                mentions = [self._row_to_mention(m) for m in mentions_cursor]
                entities.append(self._row_to_entity(row, mentions))
        
        return entities
    
    def find_entities_in_node(self, node_id: str) -> list[Entity]:
        """
        Find all entities mentioned in a specific node.
        
        Args:
            node_id: Node ID.
            
        Returns:
            List of entities with mentions in the node.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT e.* FROM entities e
                JOIN mentions m ON e.id = m.entity_id
                WHERE m.node_id = ?
                """,
                (node_id,),
            )
            
            entities = []
            for row in cursor:
                mentions_cursor = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ?",
                    (row["id"],),
                )
                mentions = [self._row_to_mention(m) for m in mentions_cursor]
                entities.append(self._row_to_entity(row, mentions))
        
        return entities
    
    def find_entities_in_document(self, doc_id: str) -> list[Entity]:
        """
        Find all entities in a document.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            List of entities with mentions in the document.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT e.* FROM entities e
                JOIN mentions m ON e.id = m.entity_id
                WHERE m.doc_id = ?
                """,
                (doc_id,),
            )
            
            entities = []
            for row in cursor:
                mentions_cursor = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ? AND doc_id = ?",
                    (row["id"], doc_id),
                )
                mentions = [self._row_to_mention(m) for m in mentions_cursor]
                entities.append(self._row_to_entity(row, mentions))
        
        return entities
    
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and its mentions.
        
        Args:
            entity_id: Entity ID.
            
        Returns:
            True if deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM entities WHERE id = ?",
                (entity_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # Relationship Operations
    # =========================================================================
    
    def add_relationship(self, relationship: Relationship) -> str:
        """
        Add a relationship to the knowledge graph.
        
        Args:
            relationship: Relationship to add.
            
        Returns:
            Relationship ID.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO relationships (id, type, source_id, target_id, source_type, target_type, doc_id, confidence, evidence, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    confidence = excluded.confidence,
                    evidence = excluded.evidence,
                    metadata = excluded.metadata
                """,
                (
                    relationship.id,
                    relationship.type.value,
                    relationship.source_id,
                    relationship.target_id,
                    relationship.source_type,
                    relationship.target_type,
                    relationship.doc_id,
                    relationship.confidence,
                    relationship.evidence,
                    json.dumps(relationship.metadata),
                    relationship.created_at.isoformat(),
                ),
            )
            conn.commit()
        
        logger.debug(
            "relationship_added",
            relationship_id=relationship.id,
            type=relationship.type.value,
            source=relationship.source_id,
            target=relationship.target_id,
        )
        
        return relationship.id
    
    def get_relationship(self, relationship_id: str) -> Relationship | None:
        """
        Get a relationship by ID.
        
        Args:
            relationship_id: Relationship ID.
            
        Returns:
            Relationship or None if not found.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM relationships WHERE id = ?",
                (relationship_id,),
            )
            row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_relationship(row)
    
    def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[Relationship]:
        """
        Get relationships involving an entity.
        
        Args:
            entity_id: Entity ID.
            relationship_type: Optional type filter.
            direction: "outgoing", "incoming", or "both".
            
        Returns:
            List of relationships.
        """
        with self._connect() as conn:
            conditions = []
            params = []
            
            if direction in ("outgoing", "both"):
                conditions.append("source_id = ?")
                params.append(entity_id)
            if direction in ("incoming", "both"):
                conditions.append("target_id = ?")
                params.append(entity_id)
            
            where_clause = " OR ".join(conditions)
            
            if relationship_type:
                query = f"SELECT * FROM relationships WHERE ({where_clause}) AND type = ?"
                params.append(relationship_type.value)
            else:
                query = f"SELECT * FROM relationships WHERE {where_clause}"
            
            cursor = conn.execute(query, params)
            relationships = [self._row_to_relationship(row) for row in cursor]
        
        return relationships
    
    def get_node_relationships(
        self,
        node_id: str,
        relationship_type: RelationType | None = None,
    ) -> list[Relationship]:
        """
        Get relationships involving a node (as source or target).
        
        Args:
            node_id: Node ID.
            relationship_type: Optional type filter.
            
        Returns:
            List of relationships.
        """
        with self._connect() as conn:
            if relationship_type:
                cursor = conn.execute(
                    """
                    SELECT * FROM relationships 
                    WHERE (source_id = ? OR target_id = ?) AND type = ?
                    """,
                    (node_id, node_id, relationship_type.value),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM relationships 
                    WHERE source_id = ? OR target_id = ?
                    """,
                    (node_id, node_id),
                )
            relationships = [self._row_to_relationship(row) for row in cursor]
        
        return relationships
    
    def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship.
        
        Args:
            relationship_id: Relationship ID.
            
        Returns:
            True if deleted.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM relationships WHERE id = ?",
                (relationship_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
    
    # =========================================================================
    # Entity Linking Operations
    # =========================================================================
    
    def link_entities(
        self,
        entity_id_1: str,
        entity_id_2: str,
        confidence: float = 1.0,
        link_method: str = "exact",
        evidence: str = "",
    ) -> None:
        """
        Create a link between two entities (same real-world entity).
        
        Args:
            entity_id_1: First entity ID.
            entity_id_2: Second entity ID.
            confidence: Link confidence (0.0-1.0).
            link_method: How the link was established.
            evidence: Justification for the link.
        """
        # Ensure consistent ordering
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1
        
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO entity_links (entity_id_1, entity_id_2, confidence, link_method, evidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_id_1, entity_id_2) DO UPDATE SET
                    confidence = MAX(excluded.confidence, entity_links.confidence),
                    link_method = excluded.link_method,
                    evidence = excluded.evidence
                """,
                (entity_id_1, entity_id_2, confidence, link_method, evidence),
            )
            conn.commit()
        
        logger.debug(
            "entities_linked",
            entity_1=entity_id_1,
            entity_2=entity_id_2,
            confidence=confidence,
            method=link_method,
        )
    
    def get_linked_entities(
        self,
        entity_id: str,
        min_confidence: float = 0.0,
    ) -> list[EntityLink]:
        """
        Get all entities linked to a given entity.
        
        Args:
            entity_id: Entity ID.
            min_confidence: Minimum link confidence.
            
        Returns:
            List of EntityLink objects.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM entity_links 
                WHERE (entity_id_1 = ? OR entity_id_2 = ?) AND confidence >= ?
                """,
                (entity_id, entity_id, min_confidence),
            )
            
            links = []
            for row in cursor:
                links.append(EntityLink(
                    entity_id_1=row["entity_id_1"],
                    entity_id_2=row["entity_id_2"],
                    confidence=row["confidence"],
                    link_method=row["link_method"],
                    evidence=row["evidence"] or "",
                    created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
                ))
        
        return links
    
    def find_entity_across_documents(
        self,
        entity_id: str,
        min_confidence: float = 0.5,
    ) -> list[Entity]:
        """
        Find the same entity across multiple documents.
        
        Args:
            entity_id: Starting entity ID.
            min_confidence: Minimum link confidence.
            
        Returns:
            List of linked entities (including the original).
        """
        # Get the original entity
        original = self.get_entity(entity_id)
        if not original:
            return []
        
        result = [original]
        
        # Get linked entities
        links = self.get_linked_entities(entity_id, min_confidence)
        
        for link in links:
            linked_id = link.entity_id_2 if link.entity_id_1 == entity_id else link.entity_id_1
            linked_entity = self.get_entity(linked_id)
            if linked_entity:
                result.append(linked_entity)
        
        return result
    
    # =========================================================================
    # Query Operations
    # =========================================================================
    
    def get_entities_mentioned_together(
        self,
        entity_id: str,
    ) -> list[tuple[Entity, int]]:
        """
        Find entities that appear in the same nodes as a given entity.
        
        Args:
            entity_id: Entity ID to find co-occurrences for.
            
        Returns:
            List of (entity, co-occurrence count) tuples.
        """
        with self._connect() as conn:
            # Get nodes where the entity is mentioned
            cursor = conn.execute(
                "SELECT DISTINCT node_id FROM mentions WHERE entity_id = ?",
                (entity_id,),
            )
            node_ids = [row["node_id"] for row in cursor]
            
            if not node_ids:
                return []
            
            # Find other entities in those nodes
            placeholders = ",".join("?" * len(node_ids))
            cursor = conn.execute(
                f"""
                SELECT e.*, COUNT(DISTINCT m.node_id) as co_count
                FROM entities e
                JOIN mentions m ON e.id = m.entity_id
                WHERE m.node_id IN ({placeholders}) AND e.id != ?
                GROUP BY e.id
                ORDER BY co_count DESC
                """,
                node_ids + [entity_id],
            )
            
            results = []
            for row in cursor:
                mentions_cursor = conn.execute(
                    "SELECT * FROM mentions WHERE entity_id = ?",
                    (row["id"],),
                )
                mentions = [self._row_to_mention(m) for m in mentions_cursor]
                entity = self._row_to_entity(row, mentions)
                results.append((entity, row["co_count"]))
        
        return results
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        with self._connect() as conn:
            entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            mention_count = conn.execute("SELECT COUNT(*) FROM mentions").fetchone()[0]
            relationship_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
            link_count = conn.execute("SELECT COUNT(*) FROM entity_links").fetchone()[0]
            
            # Type distribution
            type_cursor = conn.execute(
                "SELECT type, COUNT(*) as count FROM entities GROUP BY type"
            )
            type_distribution = {row["type"]: row["count"] for row in type_cursor}
        
        return {
            "entity_count": entity_count,
            "mention_count": mention_count,
            "relationship_count": relationship_count,
            "entity_link_count": link_count,
            "entity_type_distribution": type_distribution,
        }
    
    def clear(self) -> dict[str, int]:
        """
        Clear all data from the knowledge graph.
        
        Returns:
            Count of deleted items by type.
        """
        with self._connect() as conn:
            entity_count = conn.execute("DELETE FROM entities").rowcount
            mention_count = conn.execute("DELETE FROM mentions").rowcount
            relationship_count = conn.execute("DELETE FROM relationships").rowcount
            link_count = conn.execute("DELETE FROM entity_links").rowcount
            conn.commit()
        
        logger.warning(
            "knowledge_graph_cleared",
            entities=entity_count,
            mentions=mention_count,
            relationships=relationship_count,
            links=link_count,
        )
        
        return {
            "entities": entity_count,
            "mentions": mention_count,
            "relationships": relationship_count,
            "entity_links": link_count,
        }
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _row_to_entity(self, row: sqlite3.Row, mentions: list[Mention]) -> Entity:
        """Convert a database row to an Entity object."""
        return Entity(
            id=row["id"],
            type=EntityType(row["type"]),
            canonical_name=row["canonical_name"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            source_doc_id=row["source_doc_id"],
            mentions=mentions,
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )
    
    def _row_to_mention(self, row: sqlite3.Row) -> Mention:
        """Convert a database row to a Mention object."""
        return Mention(
            id=row["id"],
            node_id=row["node_id"],
            doc_id=row["doc_id"],
            context=row["context"] or "",
            span_start=row["span_start"],
            span_end=row["span_end"],
            page_num=row["page_num"],
            confidence=row["confidence"],
        )
    
    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert a database row to a Relationship object."""
        return Relationship(
            id=row["id"],
            type=RelationType(row["type"]),
            source_id=row["source_id"],
            target_id=row["target_id"],
            source_type=row["source_type"],
            target_type=row["target_type"],
            doc_id=row["doc_id"],
            confidence=row["confidence"],
            evidence=row["evidence"] or "",
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
        )


class InMemoryKnowledgeGraph:
    """
    In-memory knowledge graph for testing and ephemeral usage.
    
    API-compatible with KnowledgeGraph.
    """
    
    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relationships: dict[str, Relationship] = {}
        self._entity_links: dict[tuple[str, str], EntityLink] = {}
    
    def add_entity(self, entity: Entity) -> str:
        self._entities[entity.id] = entity
        return entity.id
    
    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)
    
    def find_entities_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
        fuzzy: bool = False,
    ) -> list[Entity]:
        results = []
        name_lower = name.lower()
        
        for entity in self._entities.values():
            if entity_type and entity.type != entity_type:
                continue
            
            if fuzzy:
                if name_lower in entity.canonical_name.lower() or any(
                    name_lower in alias.lower() for alias in entity.aliases
                ):
                    results.append(entity)
            else:
                if entity.canonical_name == name:
                    results.append(entity)
        
        return results
    
    def find_entities_by_type(
        self,
        entity_type: EntityType,
        doc_id: str | None = None,
    ) -> list[Entity]:
        results = []
        for entity in self._entities.values():
            if entity.type == entity_type:
                if doc_id is None or entity.source_doc_id == doc_id:
                    results.append(entity)
        return results
    
    def find_entities_in_node(self, node_id: str) -> list[Entity]:
        results = []
        for entity in self._entities.values():
            if any(m.node_id == node_id for m in entity.mentions):
                results.append(entity)
        return results
    
    def find_entities_in_document(self, doc_id: str) -> list[Entity]:
        results = []
        for entity in self._entities.values():
            if any(m.doc_id == doc_id for m in entity.mentions):
                results.append(entity)
        return results
    
    def delete_entity(self, entity_id: str) -> bool:
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False
    
    def add_relationship(self, relationship: Relationship) -> str:
        self._relationships[relationship.id] = relationship
        return relationship.id
    
    def get_relationship(self, relationship_id: str) -> Relationship | None:
        return self._relationships.get(relationship_id)
    
    def get_entity_relationships(
        self,
        entity_id: str,
        relationship_type: RelationType | None = None,
        direction: str = "both",
    ) -> list[Relationship]:
        results = []
        for rel in self._relationships.values():
            if relationship_type and rel.type != relationship_type:
                continue
            
            if direction == "outgoing" and rel.source_id == entity_id:
                results.append(rel)
            elif direction == "incoming" and rel.target_id == entity_id:
                results.append(rel)
            elif direction == "both" and (rel.source_id == entity_id or rel.target_id == entity_id):
                results.append(rel)
        
        return results
    
    def get_node_relationships(
        self,
        node_id: str,
        relationship_type: RelationType | None = None,
    ) -> list[Relationship]:
        results = []
        for rel in self._relationships.values():
            if relationship_type and rel.type != relationship_type:
                continue
            if rel.source_id == node_id or rel.target_id == node_id:
                results.append(rel)
        return results
    
    def delete_relationship(self, relationship_id: str) -> bool:
        if relationship_id in self._relationships:
            del self._relationships[relationship_id]
            return True
        return False
    
    def link_entities(
        self,
        entity_id_1: str,
        entity_id_2: str,
        confidence: float = 1.0,
        link_method: str = "exact",
        evidence: str = "",
    ) -> None:
        if entity_id_1 > entity_id_2:
            entity_id_1, entity_id_2 = entity_id_2, entity_id_1
        
        self._entity_links[(entity_id_1, entity_id_2)] = EntityLink(
            entity_id_1=entity_id_1,
            entity_id_2=entity_id_2,
            confidence=confidence,
            link_method=link_method,
            evidence=evidence,
        )
    
    def get_linked_entities(
        self,
        entity_id: str,
        min_confidence: float = 0.0,
    ) -> list[EntityLink]:
        results = []
        for link in self._entity_links.values():
            if (link.entity_id_1 == entity_id or link.entity_id_2 == entity_id) and link.confidence >= min_confidence:
                results.append(link)
        return results
    
    def find_entity_across_documents(
        self,
        entity_id: str,
        min_confidence: float = 0.5,
    ) -> list[Entity]:
        original = self.get_entity(entity_id)
        if not original:
            return []
        
        result = [original]
        links = self.get_linked_entities(entity_id, min_confidence)
        
        for link in links:
            linked_id = link.entity_id_2 if link.entity_id_1 == entity_id else link.entity_id_1
            linked_entity = self.get_entity(linked_id)
            if linked_entity:
                result.append(linked_entity)
        
        return result
    
    def get_entities_mentioned_together(
        self,
        entity_id: str,
    ) -> list[tuple[Entity, int]]:
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        
        node_ids = {m.node_id for m in entity.mentions}
        co_occurrences: dict[str, int] = {}
        
        for other in self._entities.values():
            if other.id == entity_id:
                continue
            count = sum(1 for m in other.mentions if m.node_id in node_ids)
            if count > 0:
                co_occurrences[other.id] = count
        
        results = [
            (self._entities[eid], count)
            for eid, count in sorted(co_occurrences.items(), key=lambda x: -x[1])
        ]
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        type_distribution: dict[str, int] = {}
        for entity in self._entities.values():
            type_distribution[entity.type.value] = type_distribution.get(entity.type.value, 0) + 1
        
        return {
            "entity_count": len(self._entities),
            "mention_count": sum(len(e.mentions) for e in self._entities.values()),
            "relationship_count": len(self._relationships),
            "entity_link_count": len(self._entity_links),
            "entity_type_distribution": type_distribution,
        }
    
    def clear(self) -> dict[str, int]:
        counts = {
            "entities": len(self._entities),
            "mentions": sum(len(e.mentions) for e in self._entities.values()),
            "relationships": len(self._relationships),
            "entity_links": len(self._entity_links),
        }
        self._entities.clear()
        self._relationships.clear()
        self._entity_links.clear()
        return counts
