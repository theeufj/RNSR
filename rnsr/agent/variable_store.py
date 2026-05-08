"""
Variable Store - Pointer-Based Content Management

Implements the Variable Stitching pattern for efficient context management.

CRITICAL DESIGN:
- Agent stores findings as $POINTER_NAME (e.g., "$LIABILITY_CLAUSE")
- Full content stored externally in this VariableStore
- LLM context contains ONLY pointers until synthesis
- Pointers resolved to full text ONLY at final synthesis step

Why This Matters:
- Prevents context pollution during navigation
- Allows comparison of multiple sections efficiently
- Enables true multi-hop reasoning without context overflow

Usage:
    store = VariableStore()
    
    # During navigation - store finding as pointer
    store.assign("$LIABILITY_CLAUSE", content, source_node_id)
    
    # Agent context contains only: "Found: $LIABILITY_CLAUSE"
    # NOT the full 2000-word clause text
    
    # At synthesis - resolve pointers
    full_text = store.resolve("$LIABILITY_CLAUSE")
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import structlog

from rnsr.exceptions import AgentError
from rnsr.models import StoredVariable

logger = structlog.get_logger(__name__)


class VariableStore:
    """
    Pointer-based variable storage for agent context management.
    
    Stores content externally while agent context only holds pointers.
    
    Attributes:
        variables: Dictionary of pointer name -> content.
        metadata: Dictionary of pointer name -> StoredVariable.
    """
    
    # Pattern for valid pointer names: $UPPER_CASE_NAME
    POINTER_PATTERN = re.compile(r"^\$[A-Z][A-Z0-9_]*$")
    
    @staticmethod
    def sanitize_pointer(pointer: str) -> str:
        """
        Sanitize a pointer name to be valid.
        
        Converts invalid characters (periods, spaces, lowercase) to valid format.
        
        Args:
            pointer: Raw pointer name (may be invalid)
            
        Returns:
            Valid pointer name matching $UPPER_CASE_NAME pattern.
            
        Example:
            sanitize_pointer("$SECTION_3.1_CONTENT") -> "$SECTION_3_1_CONTENT"
            sanitize_pointer("section_name") -> "$SECTION_NAME"
        """
        # Ensure it starts with $
        if not pointer.startswith("$"):
            pointer = "$" + pointer
        
        # Convert to uppercase
        pointer = pointer.upper()
        
        # Replace invalid characters with underscores
        # Valid: A-Z, 0-9, _
        sanitized = "$"
        for char in pointer[1:]:  # Skip the $
            if char.isalnum() or char == "_":
                sanitized += char
            else:
                sanitized += "_"  # Replace periods, spaces, etc. with _
        
        # Remove consecutive underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        
        # Remove trailing underscores
        sanitized = sanitized.rstrip("_")
        
        # Ensure it starts with a letter after $
        if len(sanitized) > 1 and not sanitized[1].isalpha():
            sanitized = "$X" + sanitized[1:]
        
        # Ensure minimum length
        if len(sanitized) < 2:
            sanitized = "$VAR"
        
        return sanitized
    
    def __init__(self):
        """Initialize an empty variable store."""
        self._content: dict[str, str] = {}
        self._metadata: dict[str, StoredVariable] = {}
        
        logger.debug("variable_store_initialized")
    
    def assign(
        self,
        pointer: str,
        content: str,
        source_node_id: str = "",
    ) -> StoredVariable:
        """
        Store content under a pointer name.
        
        Args:
            pointer: Variable pointer (e.g., "$LIABILITY_CLAUSE").
                     Must match pattern $UPPER_CASE_NAME.
            content: Full text content to store.
            source_node_id: ID of the source node (for traceability).
            
        Returns:
            StoredVariable with metadata.
            
        Raises:
            AgentError: If pointer format is invalid.
            
        Example:
            store.assign("$PAYMENT_TERMS", "Payment due in 30 days...", "node_123")
        """
        # Sanitize pointer format (instead of raising error)
        if not self.POINTER_PATTERN.match(pointer):
            original = pointer
            pointer = self.sanitize_pointer(pointer)
            logger.debug(
                "pointer_sanitized",
                original=original,
                sanitized=pointer,
            )

        # Avoid silent overwrite on collision. Multiple sections can share a
        # header (e.g. four "Broker Non-Votes" subsections under one Foot Locker
        # 8-K), and previously they all collapsed onto the same pointer with
        # last-write-wins, hiding earlier content from synthesis. We instead
        # disambiguate with a numeric suffix so every section stays addressable.
        # Re-assigning the *same* (source_node_id, content) is a no-op so callers
        # that re-run navigation don't accumulate duplicate copies.
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        existing = self._metadata.get(pointer)
        if existing is not None:
            if existing.source_node_id == source_node_id and existing.content_hash == content_hash:
                logger.debug(
                    "variable_assign_idempotent",
                    pointer=pointer,
                    source=source_node_id,
                )
                return existing

            base = pointer
            n = 2
            candidate = f"{base}_{n}"
            while candidate in self._content:
                if (
                    self._metadata[candidate].source_node_id == source_node_id
                    and self._metadata[candidate].content_hash == content_hash
                ):
                    logger.debug(
                        "variable_assign_idempotent",
                        pointer=candidate,
                        source=source_node_id,
                    )
                    return self._metadata[candidate]
                n += 1
                candidate = f"{base}_{n}"
            logger.info(
                "pointer_collision_disambiguated",
                requested=base,
                assigned=candidate,
                existing_source=existing.source_node_id,
                new_source=source_node_id,
            )
            pointer = candidate

        meta = StoredVariable(
            pointer=pointer,
            source_node_id=source_node_id,
            content_hash=content_hash,
            char_count=len(content),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._content[pointer] = content
        self._metadata[pointer] = meta

        logger.info(
            "variable_assigned",
            pointer=pointer,
            source=source_node_id,
            chars=len(content),
        )

        return meta
    
    def resolve(self, pointer: str) -> str | None:
        """
        Resolve a pointer to its full content.
        
        Args:
            pointer: Variable pointer (e.g., "$LIABILITY_CLAUSE").
            
        Returns:
            Full text content, or None if not found.
            
        Example:
            content = store.resolve("$PAYMENT_TERMS")
        """
        content = self._content.get(pointer)
        
        if content is None:
            logger.warning("variable_not_found", pointer=pointer)
        else:
            logger.debug("variable_resolved", pointer=pointer)
        
        return content
    
    def resolve_many(self, pointers: list[str]) -> dict[str, str | None]:
        """
        Resolve multiple pointers at once.
        
        Args:
            pointers: List of pointer names.
            
        Returns:
            Dictionary mapping pointer -> content (or None).
        """
        return {p: self._content.get(p) for p in pointers}
    
    def resolve_all_in_text(self, text: str) -> str:
        """
        Find and resolve all pointers in a text string.
        
        Args:
            text: Text containing $POINTER references.
            
        Returns:
            Text with pointers replaced by their content.
            
        Example:
            text = "Compare $SECTION_A with $SECTION_B"
            resolved = store.resolve_all_in_text(text)
            # Returns: "Compare [full content A] with [full content B]"
        """
        # Find all pointers in text
        pointers = re.findall(r"\$[A-Z][A-Z0-9_]*", text)
        
        result = text
        for pointer in pointers:
            content = self._content.get(pointer)
            if content is not None:
                result = result.replace(pointer, content)
            else:
                logger.warning("unresolved_pointer", pointer=pointer)
        
        return result
    
    def list_variables(self) -> list[StoredVariable]:
        """
        List all stored variables with metadata.
        
        Returns:
            List of StoredVariable objects.
        """
        return list(self._metadata.values())
    
    def list_pointers(self) -> list[str]:
        """
        Get all pointer names.
        
        Returns:
            List of pointer strings.
        """
        return list(self._content.keys())
    
    def get_metadata(self, pointer: str) -> StoredVariable | None:
        """Get metadata for a stored variable."""
        return self._metadata.get(pointer)
    
    def exists(self, pointer: str) -> bool:
        """Check if a pointer exists."""
        return pointer in self._content
    
    def delete(self, pointer: str) -> bool:
        """
        Delete a stored variable.
        
        Returns:
            True if deleted, False if not found.
        """
        if pointer in self._content:
            del self._content[pointer]
            del self._metadata[pointer]
            logger.debug("variable_deleted", pointer=pointer)
            return True
        return False
    
    def clear(self) -> int:
        """
        Clear all stored variables.
        
        Returns:
            Number of variables cleared.
        """
        count = len(self._content)
        self._content.clear()
        self._metadata.clear()
        logger.info("variable_store_cleared", count=count)
        return count
    
    def count(self) -> int:
        """Get the number of stored variables."""
        return len(self._content)
    
    def total_chars(self) -> int:
        """Get total character count across all stored content."""
        return sum(len(c) for c in self._content.values())
    
    def summary(self) -> dict[str, Any]:
        """
        Get a summary of store contents for agent context.
        
        This summary can be included in agent context to show
        what variables are available without including content.
        
        Returns:
            Summary dict suitable for LLM context.
        """
        variables = []
        for meta in self._metadata.values():
            variables.append({
                "pointer": meta.pointer,
                "source": meta.source_node_id,
                "chars": meta.char_count,
            })
        
        return {
            "stored_variables": variables,
            "count": len(variables),
            "total_chars": self.total_chars(),
            "hint": "Use resolve_variable(pointer) to get content during synthesis",
        }


def generate_pointer_name(header: str, prefix: str = "") -> str:
    """
    Generate a valid pointer name from a section header.
    
    Args:
        header: Section header text.
        prefix: Optional prefix (e.g., "SEC" for sections).
        
    Returns:
        Valid pointer name like $LIABILITY_CLAUSE.
        
    Example:
        generate_pointer_name("Liability Clause") -> "$LIABILITY_CLAUSE"
        generate_pointer_name("Section 3.2", prefix="S") -> "$S_SECTION_3_2"
    """
    # Clean and normalize
    name = header.upper()
    
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^A-Z0-9]+", "_", name)
    
    # Remove leading/trailing underscores
    name = name.strip("_")
    
    # Add prefix
    if prefix:
        name = f"{prefix}_{name}"
    
    # Ensure valid start (letter, not number)
    if name and name[0].isdigit():
        name = "N" + name
    
    # Truncate if too long
    if len(name) > 30:
        name = name[:30].rstrip("_")
    
    # Add $ prefix
    return f"${name}" if name else "$UNNAMED"
