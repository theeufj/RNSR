"""
RNSR RLM Entity Extractor

Implements the TRUE RLM pattern for entity extraction:
1. LLM writes its own regex/Python code based on the document
2. Code executes on DOC_VAR (grounded in actual text)
3. LLM validates and classifies results

This is more powerful than pre-defined patterns because:
- LLM adapts to domain-specific patterns it discovers
- Can write complex extraction logic we didn't anticipate
- Still grounded because code executes on actual text
- Recursive - can use sub_llm for complex validation

From the RLM paper:
"The Neural Network generates code to interact with the document,
rather than having the document in its context window."
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import structlog

from rnsr.extraction.models import Entity, EntityType, ExtractionResult, Mention
from rnsr.extraction.learned_types import get_learned_type_registry
from rnsr.llm import get_llm

if TYPE_CHECKING:
    from rnsr.agent.repl_env import REPLEnvironment
    from rnsr.models import DocumentTree

logger = structlog.get_logger(__name__)


# =============================================================================
# RLM Extraction Prompts
# =============================================================================

RLM_ENTITY_EXTRACTION_SYSTEM = """You are an RLM (Recursive Language Model) extracting entities from a document.

CRITICAL: You do NOT have the full document in context. It is stored in DOC_VAR.
You must write Python code to extract entities from DOC_VAR.

## Available Variables:
- DOC_VAR: The document text (string). Use slicing, regex, etc.
- SECTION_CONTENT: Current section content (string, smaller than DOC_VAR)

## Available Functions:
- search_text(pattern): Search DOC_VAR for regex pattern, returns list of (start, end, match)
- len(DOC_VAR): Get document length
- DOC_VAR[i:j]: Slice document
- re.findall(pattern, text): Standard regex
- re.finditer(pattern, text): Iterate matches with positions
- store_variable(name, content): Store findings for later

## Your Task:
Extract entities (people, organizations, dates, etc.) by writing Python code.

IMPORTANT: Your code should:
1. Write regex patterns tailored to THIS document
2. Execute patterns to find matches (grounded in text)
3. Return structured results with exact positions

## Output Format:
Write Python code that produces a list of entity dictionaries:
```python
entities = []

# Example: Find person names with titles
for match in re.finditer(r'(?:Mr\.|Mrs\.|Dr\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)', SECTION_CONTENT):
    entities.append({{
        "text": match.group(),
        "canonical_name": match.group(1),  # Without title
        "type": "PERSON",
        "start": match.start(),
        "end": match.end(),
        "confidence": 0.9
    }})

# Example: Find dollar amounts
for match in re.finditer(r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion))?', SECTION_CONTENT):
    entities.append({{
        "text": match.group(),
        "type": "MONETARY",
        "start": match.start(),
        "end": match.end(),
        "confidence": 0.95
    }})

# Store results
store_variable("ENTITIES", entities)
```

Write code appropriate for the document type and content shown."""


RLM_EXTRACTION_PROMPT = """Document section to extract entities from:

Section Header: {header}
Section Content (first 2000 chars):
---
{content_preview}
---

Total section length: {content_length} characters

Based on this content, write Python code to extract all significant entities.
Consider:
1. What types of entities appear in this document? (people, companies, dates, money, etc.)
2. What patterns would match them? (titles, suffixes, formats, etc.)
3. Are there domain-specific entities? (legal terms, technical concepts, etc.)

Write Python code that will execute on SECTION_CONTENT to extract entities.
End your code with: store_variable("ENTITIES", entities)"""


RLM_VALIDATION_PROMPT = """You extracted these entity candidates from the document.
Validate each one and determine if it's a real, significant entity.

Candidates:
{candidates_json}

For each candidate, provide:
1. valid: true if significant entity, false if noise
2. type: Entity type (PERSON, ORGANIZATION, DATE, MONETARY, LOCATION, etc.)
3. canonical_name: Cleaned/normalized name
4. confidence: 0.0-1.0

Return JSON array:
```json
[
  {{"id": 0, "valid": true, "type": "PERSON", "canonical_name": "John Smith", "confidence": 0.9}},
  {{"id": 1, "valid": false, "reason": "Generic term, not specific entity"}}
]
```"""


# =============================================================================
# Lightweight REPL for Extraction (if full REPL not available)
# =============================================================================

class LightweightREPL:
    """
    Lightweight REPL for entity extraction.
    
    Provides the core DOC_VAR + code execution pattern
    without the full REPL infrastructure.
    """
    
    def __init__(self, document_text: str, section_content: str = ""):
        """Initialize with document text."""
        self.document_text = document_text
        self.section_content = section_content or document_text
        self.variables: dict[str, Any] = {}
        
        self._namespace = self._build_namespace()
    
    def _build_namespace(self) -> dict[str, Any]:
        """Build Python namespace for code execution."""
        return {
            # Core variables
            "DOC_VAR": self.document_text,
            "SECTION_CONTENT": self.section_content,
            "VARIABLES": self.variables,
            
            # Built-ins
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "range": range,
            "enumerate": enumerate,
            "sorted": sorted,
            "min": min,
            "max": max,
            "re": re,
            
            # Functions
            "search_text": self._search_text,
            "store_variable": self._store_variable,
            "get_variable": self._get_variable,
        }
    
    def _search_text(self, pattern: str) -> list[tuple[int, int, str]]:
        """Search document for regex pattern."""
        results = []
        try:
            for match in re.finditer(pattern, self.section_content, re.IGNORECASE):
                results.append((match.start(), match.end(), match.group()))
        except re.error as e:
            logger.warning("regex_error", pattern=pattern, error=str(e))
        return results
    
    def _store_variable(self, name: str, content: Any) -> str:
        """Store a variable."""
        self.variables[name] = content
        return f"Stored ${name}"
    
    def _get_variable(self, name: str) -> Any:
        """Retrieve a variable."""
        return self.variables.get(name)
    
    def execute(self, code: str) -> dict[str, Any]:
        """Execute Python code."""
        result = {
            "success": False,
            "output": None,
            "error": None,
            "variables": list(self.variables.keys()),
        }
        
        # Clean code
        code = self._clean_code(code)
        
        try:
            # Compile and execute
            compiled = compile(code, "<rlm_extraction>", "exec")
            exec(compiled, self._namespace)
            
            result["success"] = True
            result["variables"] = list(self.variables.keys())
            result["output"] = self.variables.get("ENTITIES", [])
            
        except Exception as e:
            result["error"] = str(e)
            logger.warning("rlm_execution_error", error=str(e), code=code[:200])
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Remove markdown code blocks."""
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*$', '', code, flags=re.MULTILINE)
        return code.strip()


# =============================================================================
# RLM Entity Extractor
# =============================================================================

@dataclass
class RLMExtractionResult:
    """Result of RLM-based extraction."""
    
    entities: list[Entity] = field(default_factory=list)
    code_generated: str = ""
    code_executed: bool = False
    execution_output: Any = None
    raw_candidates: list[dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class RLMEntityExtractor:
    """
    RLM-based entity extractor.
    
    The LLM writes its own extraction code based on the document,
    then the code executes on DOC_VAR (grounded).
    
    Flow:
    1. Show LLM a preview of the document
    2. LLM writes Python code to extract entities
    3. Code executes on actual document (grounded)
    4. LLM validates/classifies the extracted candidates
    """
    
    def __init__(
        self,
        llm: Any | None = None,
        repl_environment: "REPLEnvironment | None" = None,
        enable_type_learning: bool = True,
        max_code_attempts: int = 3,
        validate_with_llm: bool = True,
    ):
        """
        Initialize the RLM extractor.
        
        Args:
            llm: LLM instance.
            repl_environment: Optional full REPL environment.
            enable_type_learning: Learn new entity types.
            max_code_attempts: Max attempts if code fails.
            validate_with_llm: Validate candidates with LLM.
        """
        self.llm = llm
        self.repl_environment = repl_environment
        self.enable_type_learning = enable_type_learning
        self.max_code_attempts = max_code_attempts
        self.validate_with_llm = validate_with_llm
        
        self._llm_initialized = False
        self._type_registry = get_learned_type_registry() if enable_type_learning else None
    
    def _get_llm(self) -> Any:
        """Get or initialize LLM."""
        if self.llm is None and not self._llm_initialized:
            self.llm = get_llm()
            self._llm_initialized = True
        return self.llm
    
    def extract_from_node(
        self,
        node_id: str,
        doc_id: str,
        header: str,
        content: str,
        page_num: int | None = None,
        document_text: str | None = None,
    ) -> ExtractionResult:
        """
        Extract entities using RLM approach.
        
        The LLM writes code to extract entities, which is then
        executed on the actual document text.
        
        Args:
            node_id: Section node ID.
            doc_id: Document ID.
            header: Section header.
            content: Section content.
            page_num: Page number.
            document_text: Full document text for DOC_VAR.
            
        Returns:
            ExtractionResult with extracted entities.
        """
        start_time = time.time()
        
        result = ExtractionResult(
            node_id=node_id,
            doc_id=doc_id,
            extraction_method="rlm",
        )
        
        if len(content.strip()) < 50:
            return result
        
        llm = self._get_llm()
        if llm is None:
            result.warnings.append("No LLM available for RLM extraction")
            return result
        
        # STEP 1: LLM generates extraction code
        rlm_result = self._generate_and_execute_code(
            header=header,
            content=content,
            document_text=document_text or content,
        )
        
        if not rlm_result.code_executed:
            result.warnings.append(f"Code execution failed: {rlm_result.warnings}")
            return result
        
        # STEP 2: Validate candidates with LLM
        if self.validate_with_llm and rlm_result.raw_candidates:
            validated = self._validate_candidates(rlm_result.raw_candidates)
        else:
            validated = rlm_result.raw_candidates
        
        # STEP 3: Convert to Entity objects
        entities = self._candidates_to_entities(
            candidates=validated,
            node_id=node_id,
            doc_id=doc_id,
            content=content,
            page_num=page_num,
        )
        
        result.entities = entities
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            "rlm_extraction_complete",
            node_id=node_id,
            candidates=len(rlm_result.raw_candidates),
            validated=len(entities),
            time_ms=result.processing_time_ms,
        )
        
        return result
    
    def _generate_and_execute_code(
        self,
        header: str,
        content: str,
        document_text: str,
    ) -> RLMExtractionResult:
        """Generate extraction code and execute it."""
        result = RLMExtractionResult()
        
        llm = self._get_llm()
        
        # Create REPL environment
        repl = self.repl_environment or LightweightREPL(
            document_text=document_text,
            section_content=content,
        )
        
        # Generate code prompt
        extraction_prompt = RLM_EXTRACTION_PROMPT.format(
            header=header,
            content_preview=content,
            content_length=len(content),
        )
        prompt = f"{RLM_ENTITY_EXTRACTION_SYSTEM}\n\n{extraction_prompt}"
        
        for attempt in range(self.max_code_attempts):
            try:
                # LLM generates extraction code
                response = llm.complete(prompt)
                code = str(response) if not isinstance(response, str) else response
                result.code_generated = code
                
                # Execute the code
                exec_result = repl.execute(code)
                
                if exec_result["success"]:
                    result.code_executed = True
                    result.execution_output = exec_result["output"]
                    result.raw_candidates = exec_result.get("output", [])
                    
                    # Get ENTITIES from variables if not in output
                    if not result.raw_candidates and hasattr(repl, 'variables'):
                        result.raw_candidates = repl.variables.get("ENTITIES", [])
                    
                    break
                else:
                    result.warnings.append(f"Attempt {attempt + 1}: {exec_result['error']}")
                    # Add error to prompt for retry
                    prompt += f"\n\nPrevious code had error: {exec_result['error']}\nPlease fix and try again."
                    
            except Exception as e:
                result.warnings.append(f"Attempt {attempt + 1}: {str(e)}")
        
        return result
    
    def _validate_candidates(
        self,
        candidates: list[dict],
    ) -> list[dict]:
        """Validate extracted candidates with LLM."""
        if not candidates:
            return []
        
        llm = self._get_llm()
        
        # Format candidates for validation
        candidates_json = json.dumps([
            {
                "id": i,
                "text": c.get("text", ""),
                "type": c.get("type", "UNKNOWN"),
                "context": c.get("context", "") if c.get("context") else "",
            }
            for i, c in enumerate(candidates)
        ], indent=2)
        
        prompt = RLM_VALIDATION_PROMPT.format(candidates_json=candidates_json)
        
        try:
            _complete_fn = getattr(llm, "complete_json", None) or llm.complete
            response = _complete_fn(prompt)
            response_text = str(response) if not isinstance(response, str) else response
            
            # Parse validation response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if not json_match:
                return candidates
            
            validations = json.loads(json_match.group())
            
            # Merge validations with candidates
            validated = []
            validation_by_id = {v.get("id"): v for v in validations}
            
            for i, candidate in enumerate(candidates):
                validation = validation_by_id.get(i, {})
                
                if validation.get("valid", True):
                    candidate["type"] = validation.get("type", candidate.get("type", "OTHER"))
                    candidate["canonical_name"] = validation.get("canonical_name", candidate.get("text", ""))
                    candidate["confidence"] = validation.get("confidence", candidate.get("confidence", 0.5))
                    validated.append(candidate)
            
            return validated
            
        except Exception as e:
            logger.warning("rlm_validation_failed", error=str(e))
            return candidates
    
    def _candidates_to_entities(
        self,
        candidates: list[dict],
        node_id: str,
        doc_id: str,
        content: str,
        page_num: int | None,
    ) -> list[Entity]:
        """Convert validated candidates to Entity objects."""
        entities = []
        
        for candidate in candidates:
            if not candidate.get("text"):
                continue
            
            # Map entity type
            entity_type = self._map_entity_type(candidate.get("type", "OTHER"))
            
            # Learn new types
            if entity_type == EntityType.OTHER and self._type_registry:
                self._type_registry.record_type(
                    type_name=candidate.get("type", "unknown").lower(),
                    context=candidate.get("context", content[:100]),
                    entity_name=candidate.get("text", ""),
                )
            
            # Create mention
            mention = Mention(
                node_id=node_id,
                doc_id=doc_id,
                span_start=candidate.get("start"),
                span_end=candidate.get("end"),
                context=candidate.get("context", content[
                    max(0, (candidate.get("start") or 0) - 50):
                    (candidate.get("end") or 0) + 50
                ]),
                page_num=page_num,
                confidence=candidate.get("confidence", 0.5),
            )
            
            # Build metadata
            metadata = {
                "rlm_extracted": True,
                "grounded": candidate.get("start") is not None,
            }
            
            if entity_type == EntityType.OTHER:
                metadata["original_type"] = candidate.get("type", "").lower()
            
            entity = Entity(
                type=entity_type,
                canonical_name=candidate.get("canonical_name", candidate.get("text", "")),
                aliases=[candidate.get("text")] if candidate.get("canonical_name") != candidate.get("text") else [],
                mentions=[mention],
                metadata=metadata,
                source_doc_id=doc_id,
            )
            entities.append(entity)
        
        return entities
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """Map type string to EntityType enum."""
        type_str = type_str.upper()
        
        mapping = {
            "PERSON": EntityType.PERSON,
            "PEOPLE": EntityType.PERSON,
            "NAME": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "ORG": EntityType.ORGANIZATION,
            "COMPANY": EntityType.ORGANIZATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "MONETARY": EntityType.MONETARY,
            "MONEY": EntityType.MONETARY,
            "AMOUNT": EntityType.MONETARY,
            "LOCATION": EntityType.LOCATION,
            "PLACE": EntityType.LOCATION,
            "ADDRESS": EntityType.LOCATION,
            "REFERENCE": EntityType.REFERENCE,
            "CITATION": EntityType.REFERENCE,
            "DOCUMENT": EntityType.DOCUMENT,
            "EVENT": EntityType.EVENT,
            "LEGAL_CONCEPT": EntityType.LEGAL_CONCEPT,
            "LEGAL": EntityType.LEGAL_CONCEPT,
        }
        
        try:
            return EntityType(type_str.lower())
        except ValueError:
            return mapping.get(type_str, EntityType.OTHER)
