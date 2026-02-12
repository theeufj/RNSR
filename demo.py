#!/usr/bin/env python3
"""
RNSR Interactive Demo

A Gradio-based web interface for testing RNSR with your own documents.

Usage:
    python demo.py

Then open http://localhost:7860 in your browser.
"""
from __future__ import annotations  # Enable Python 3.10+ type hints on Python 3.9

import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import gradio as gr

# =============================================================================
# Monkeypatch Gradio to fix the Pydantic v2 schema parsing bug
# Only needed for Gradio < 6.0 -- Gradio 6 handles Pydantic v2 natively.
# =============================================================================
_gradio_major = int(gr.__version__.split(".")[0]) if hasattr(gr, "__version__") else 0

if _gradio_major < 6:
    import gradio_client.utils as client_utils

    _original_json_schema_to_python_type = client_utils._json_schema_to_python_type
    _original_get_type = client_utils.get_type

    def _patched_json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_json_schema_to_python_type(schema, defs)

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_get_type(schema)

    client_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
    client_utils.get_type = _patched_get_type

    _original_json_schema_to_python_type_public = client_utils.json_schema_to_python_type

    def _patched_json_schema_to_python_type_public(schema):
        if isinstance(schema, bool):
            return "any" if schema else "never"
        if schema is None:
            return "any"
        return _original_json_schema_to_python_type_public(schema)

    client_utils.json_schema_to_python_type = _patched_json_schema_to_python_type_public

# =============================================================================
# RNSR imports
# =============================================================================
from rnsr import RNSRClient, ingest_document, build_skeleton_index
from rnsr.models import DocumentTree, DocumentNode
from rnsr.indexing import KVStore
from rnsr.indexing.knowledge_graph import KnowledgeGraph, InMemoryKnowledgeGraph
from rnsr.extraction.models import Entity, EntityType


# =============================================================================
# Session state
# =============================================================================
class SessionState:
    def __init__(self):
        self.tree: DocumentTree | None = None
        self.skeleton: dict[str, Any] | None = None
        self.kv_store: KVStore | None = None
        self.knowledge_graph: KnowledgeGraph | InMemoryKnowledgeGraph | None = None
        self.document_name: str = ""
        self.document_path: str | None = None
        self.chat_history: list = []
        self.entities: list[Entity] = []
        self.extraction_stats: dict[str, Any] = {}


state = SessionState()
client = RNSRClient(cache_dir=".rnsr_demo_cache")


# =============================================================================
# Helper: format chat messages for the correct Gradio version
# =============================================================================
def _chat_pair(user_msg: str, bot_msg: str) -> list:
    """Return a single exchange formatted for the installed Gradio."""
    if _gradio_major >= 6:
        return [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": bot_msg},
        ]
    return [(user_msg, bot_msg)]


# =============================================================================
# Core logic
# =============================================================================
def process_document(file, extract_entities: bool = True, progress=gr.Progress()):
    """Process an uploaded PDF document."""
    if file is None:
        yield "Upload a PDF to get started.", [], [], ""
        return

    try:
        file_path = file if isinstance(file, str) else file
        file_name = Path(file_path).name
        state.document_path = file_path
        state.document_name = file_name
        state.chat_history = []

        # Step 1 -- ingest
        yield (
            f"**Processing** `{file_name}` — reading PDF…",
            [], [], "",
        )
        progress(0.1, desc="Reading PDF…")
        result = ingest_document(file_path)
        state.tree = result.tree
        node_count = state.tree.total_nodes if state.tree else 0

        # Step 2 -- skeleton
        yield (
            f"**Processing** `{file_name}` — building hierarchy ({node_count} nodes)…",
            [], [], "",
        )
        progress(0.3, desc="Building hierarchy…")
        state.skeleton, state.kv_store = build_skeleton_index(state.tree)
        state.knowledge_graph = InMemoryKnowledgeGraph()
        state.entities = []
        state.extraction_stats = {}

        # Step 3 -- entity extraction (optional)
        if extract_entities and state.tree:
            est_min = max(1, (node_count * 45) // 60)
            yield (
                f"**Processing** `{file_name}` — extracting entities in parallel "
                f"({node_count} nodes, ~{est_min}-{est_min + 2} min)…",
                [], [], "",
            )
            progress(0.4, desc="Extracting entities…")

            try:
                start = time.time()
                cache_key = client._get_cache_key(Path(file_path))
                kg = client._get_or_create_knowledge_graph(
                    cache_key=cache_key,
                    skeleton=state.skeleton,
                    kv_store=state.kv_store,
                    doc_id=state.tree.id if state.tree else "document",
                )
                kg_stats = kg.get_stats()
                all_entities = kg.find_entities_in_document(
                    state.tree.id if state.tree else "document"
                )
                state.entities = all_entities
                state.knowledge_graph = kg
                elapsed = int(time.time() - start)
                state.extraction_stats = {
                    "nodes_processed": node_count,
                    "entities_extracted": kg_stats.get("entity_count", 0),
                    "relationships_extracted": kg_stats.get("relationship_count", 0),
                    "entity_types": kg_stats.get("entity_type_distribution", {}),
                    "elapsed_seconds": elapsed,
                }
            except Exception as e:
                state.extraction_stats = {"error": str(e)}

        # Step 4 -- warm client cache
        progress(0.9, desc="Preparing for questions…")
        try:
            client._get_or_create_index(Path(file_path), force_reindex=False)
        except Exception:
            pass

        progress(1.0, desc="Ready!")

        # Build final outputs
        status_md = _build_status_card()
        tables_data = _build_tables_dataframe()
        table_choices = _get_table_choices()
        tree_text = _build_tree_text()

        yield status_md, tables_data, table_choices, tree_text

    except Exception as e:
        import traceback
        yield f"**Error:** {e}\n```\n{traceback.format_exc()}\n```", [], [], ""


def chat(message: str, history: list, progress=gr.Progress()):
    """Handle a chat message."""
    if state.document_path is None:
        return "", history + _chat_pair(
            message,
            "Please upload and process a document first.",
        )

    if not message.strip():
        return "", history

    try:
        progress(0.3, desc="Searching document…")
        result = client.ask_advanced(
            document=state.document_path,
            question=message,
            use_knowledge_graph=True,
            enable_verification=False,
        )
        progress(0.8, desc="Generating answer…")

        answer = result.get("answer", str(result))
        confidence = result.get("confidence", None)

        if confidence is not None:
            response = f"{answer}\n\n*Confidence: {confidence:.0%}*"
        else:
            response = str(answer)

        # Append entity context when relevant
        if state.knowledge_graph and state.entities:
            mentioned = []
            q_lower = message.lower()
            for entity in state.entities:
                if entity.canonical_name.lower() in q_lower:
                    mentioned.append(entity)
                else:
                    for alias in entity.aliases:
                        if alias.lower() in q_lower:
                            mentioned.append(entity)
                            break
            if mentioned:
                parts = []
                for ent in mentioned[:5]:
                    rels = state.knowledge_graph.get_entity_relationships(ent.id)
                    rel_lines = []
                    for rel in rels[:8]:
                        rel_lines.append(
                            f"  - {rel.type.value} → {rel.target_id}"
                        )
                    co = state.knowledge_graph.get_entities_mentioned_together(ent.id)
                    related = [e.canonical_name for e, _ in co[:5]]
                    line = f"- **{ent.canonical_name}** ({ent.type.value})"
                    if related:
                        line += f" — related to: {', '.join(related)}"
                    if rel_lines:
                        line += "\n" + "\n".join(rel_lines)
                    parts.append(line)
                response += "\n\n*Entity context:*\n" + "\n".join(parts)

        progress(1.0, desc="Done")
        new_history = history + _chat_pair(message, response)
        state.chat_history = new_history
        return "", new_history

    except Exception as e:
        import traceback
        err = f"Error: {e}\n```\n{traceback.format_exc()}\n```"
        return "", history + _chat_pair(message, err)


# =============================================================================
# Status card builder
# =============================================================================
def _build_status_card() -> str:
    if not state.document_path:
        return "Upload a PDF above to get started."

    node_count = state.tree.total_nodes if state.tree else 0
    root_sections = len(state.tree.root.children) if state.tree and state.tree.root else 0

    parts = [
        f"### {state.document_name}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sections | {node_count} |",
        f"| Root sections | {root_sections} |",
    ]

    stats = state.extraction_stats
    if stats.get("entities_extracted"):
        parts.append(f"| Entities | {stats['entities_extracted']} |")
        parts.append(f"| Relationships | {stats.get('relationships_extracted', 0)} |")
        elapsed = stats.get("elapsed_seconds", "—")
        parts.append(f"| Extraction time | {elapsed}s |")

        type_dist = stats.get("entity_types", {})
        if type_dist:
            parts.append("")
            parts.append("**Entity breakdown:** " + ", ".join(
                f"{v} {k}" for k, v in sorted(type_dist.items(), key=lambda x: -x[1])
            ))
    elif stats.get("error"):
        parts.append(f"| Extraction | Skipped ({stats['error'][:40]}…) |")

    return "\n".join(parts)


# =============================================================================
# Tree helpers
# =============================================================================
def _build_tree_text() -> str:
    if state.tree is None:
        return "No document loaded."
    lines: list[str] = []
    _walk_tree(state.tree.root, lines, "", True)
    return "\n".join(lines)


def _walk_tree(node: DocumentNode, lines: list, prefix: str, is_last: bool):
    connector = "└─ " if is_last else "├─ "
    title = node.header or "(untitled)"
    if len(title) > 80:
        title = title[:77] + "…"
    lines.append(f"{prefix}{connector}{title}")
    if node.children:
        ext = "   " if is_last else "│  "
        for i, child in enumerate(node.children):
            _walk_tree(child, lines, prefix + ext, i == len(node.children) - 1)


# =============================================================================
# Table helpers
# =============================================================================
def _get_table_choices() -> list[str]:
    if not state.document_path:
        return []
    try:
        tables = client.list_tables(state.document_path)
        return [
            f"{t['id']} — {t.get('title') or 'Untitled'} ({t['num_rows']}×{t['num_cols']})"
            for t in tables
        ]
    except Exception:
        return []


def _build_tables_dataframe() -> list[list[str]]:
    """Build a summary table of all detected tables as a list-of-lists (for gr.Dataframe)."""
    if not state.document_path:
        return []
    try:
        tables = client.list_tables(state.document_path)
        if not tables:
            return []
        rows = []
        for t in tables:
            headers_str = ", ".join(t["headers"][:6])
            if len(t["headers"]) > 6:
                headers_str += f" (+{len(t['headers']) - 6})"
            rows.append([
                t["id"],
                t.get("title") or "—",
                str(t["num_rows"]),
                str(t["num_cols"]),
                headers_str,
                str(t.get("page_num") or "—"),
            ])
        return rows
    except Exception:
        return []


def preview_table(table_selection: str) -> list[list[str]]:
    """Return the full contents of a table as a list-of-lists (header row + data)."""
    if not state.document_path or not table_selection:
        return []
    try:
        table_id = table_selection.split(" — ")[0].strip()
        rows = client.query_table(
            document=state.document_path,
            table_id=table_id,
        )
        if not rows:
            return []
        headers = list(rows[0].keys())
        data = [headers] + [[str(r.get(h, "")) for h in headers] for r in rows]
        return data
    except Exception:
        return []


def query_table_ui(
    table_selection: str,
    columns: str,
    where_column: str,
    where_op: str,
    where_value: str,
    order_by: str,
    limit: int,
) -> list[list[str]]:
    """Run a SQL-like query on a table and return results as list-of-lists for Dataframe."""
    if not state.document_path or not table_selection:
        return []

    try:
        table_id = table_selection.split(" — ")[0].strip()

        cols = None
        if columns.strip():
            cols = [c.strip() for c in columns.split(",") if c.strip()]

        where = None
        if where_column.strip() and where_value.strip():
            if where_op in ("==", "!=", ">", ">=", "<", "<="):
                try:
                    val = float(where_value.replace(",", "").replace("$", "").strip())
                    where = {where_column.strip(): {"op": where_op, "value": val}}
                except ValueError:
                    where = {where_column.strip(): {"op": where_op, "value": where_value}}
            elif where_op == "contains":
                where = {where_column.strip(): {"op": "contains", "value": where_value}}
            else:
                where = {where_column.strip(): where_value}

        order = order_by.strip() or None
        lim = int(limit) if limit and limit > 0 else None

        results = client.query_table(
            document=state.document_path,
            table_id=table_id,
            columns=cols,
            where=where,
            order_by=order,
            limit=lim,
        )
        if not results:
            return []

        headers = list(results[0].keys())
        return [headers] + [[str(r.get(h, "")) for h in headers] for r in results]

    except Exception as e:
        return [[f"Error: {e}"]]


def run_aggregation(table_selection: str, column: str, operation: str) -> str:
    if not state.document_path or not table_selection or not column.strip():
        return ""
    try:
        table_id = table_selection.split(" — ")[0].strip()
        result = client.aggregate_table(
            document=state.document_path,
            table_id=table_id,
            column=column.strip(),
            operation=operation,
        )
        op_label = {"sum": "Sum", "avg": "Average", "count": "Count", "min": "Minimum", "max": "Maximum"}
        if isinstance(result, float):
            formatted = f"{result:,.2f}" if result != int(result) else f"{int(result):,}"
        else:
            formatted = str(result)
        return f"### {op_label.get(operation, operation)} of `{column.strip()}`\n\n# {formatted}"
    except Exception as e:
        return f"**Error:** {e}"


# =============================================================================
# Entity / KG helpers
# =============================================================================
def get_entities_visualization() -> str:
    if not state.entities:
        if state.document_path:
            return (
                "**No entities extracted yet.**\n\n"
                'Re-process the document with "Extract entities" checked on the **Chat** tab.\n\n'
                "*Q&A still works — the AI extracts entities on-the-fly when answering.*"
            )
        return "No document loaded."

    lines = [f"## Extracted Entities ({len(state.entities)} total)\n"]

    by_type: dict[str, list[Entity]] = {}
    for entity in state.entities:
        by_type.setdefault(entity.type.value, []).append(entity)

    icons = {
        "person": "👤", "organization": "🏢", "date": "📅", "event": "📌",
        "legal_concept": "⚖️", "location": "📍", "reference": "📎",
        "monetary": "💰", "document": "📄", "other": "🏷️",
    }

    for type_name, entities in sorted(by_type.items(), key=lambda x: -len(x[1])):
        icon = icons.get(type_name, "•")
        display = type_name.replace("_", " ").title()
        lines.append(f"\n### {icon} {display} ({len(entities)})\n")
        for ent in entities[:20]:
            mentions = len(ent.mentions)
            aliases = f" *(aka {', '.join(ent.aliases[:3])})*" if ent.aliases else ""
            orig = ""
            if type_name == "other" and ent.metadata.get("original_type"):
                orig = f" [{ent.metadata['original_type']}]"
            lines.append(f"- **{ent.canonical_name}**{orig}{aliases} — {mentions} mention(s)")
        if len(entities) > 20:
            lines.append(f"  *… and {len(entities) - 20} more*")

    return "\n".join(lines)


def _rel_label(rel) -> str:
    """Return a human-readable relationship label, preferring the original LLM type."""
    # If the LLM assigned a custom type that mapped to OTHER, use the original
    original = (rel.metadata or {}).get("original_type", "")
    if original and rel.type.value == "other":
        return original.replace("_", " ").title()
    return rel.type.value.replace("_", " ").title()


def get_relationships_visualization() -> str:
    if not state.knowledge_graph or not state.entities:
        if state.document_path:
            return (
                "**No relationships extracted yet.**\n\n"
                'Re-process with "Extract entities" checked to see relationships.'
            )
        return "No document loaded."

    stats = state.knowledge_graph.get_stats()
    rel_count = stats.get("relationship_count", 0)
    if rel_count == 0:
        return "No relationships found in this document."

    # Build an entity-id → name lookup
    entity_name: dict[str, str] = {}
    for ent in state.entities:
        entity_name[ent.id] = ent.canonical_name

    # Collect ALL relationships, grouped by source entity
    grouped: dict[str, list] = defaultdict(list)   # source_name -> [(label, target_name, evidence)]
    seen_ids: set[str] = set()

    for entity in state.entities:
        rels = state.knowledge_graph.get_entity_relationships(entity.id, direction="outgoing")
        for rel in rels:
            if rel.id in seen_ids:
                continue
            seen_ids.add(rel.id)

            source_name = entity_name.get(rel.source_id, rel.source_id)
            target = state.knowledge_graph.get_entity(rel.target_id)
            target_name = target.canonical_name if target else entity_name.get(rel.target_id, rel.target_id)
            label = _rel_label(rel)
            evidence = (rel.evidence or "").strip()
            grouped[source_name].append((label, target_name, evidence))

    # Also pick up any relationships whose source wasn't in our outgoing sweep
    for entity in state.entities:
        rels = state.knowledge_graph.get_entity_relationships(entity.id, direction="incoming")
        for rel in rels:
            if rel.id in seen_ids:
                continue
            seen_ids.add(rel.id)

            source_name = entity_name.get(rel.source_id, rel.source_id)
            target = state.knowledge_graph.get_entity(rel.target_id)
            target_name = target.canonical_name if target else entity_name.get(rel.target_id, rel.target_id)
            label = _rel_label(rel)
            evidence = (rel.evidence or "").strip()
            grouped[source_name].append((label, target_name, evidence))

    lines = [f"## Entity Relationships ({rel_count} total — {len(seen_ids)} shown)\n"]

    if not grouped:
        lines.append("*(No entity-to-entity relationships found)*")
        return "\n".join(lines)

    # Sort groups: entities with most relationships first
    for source_name, rels_list in sorted(grouped.items(), key=lambda x: -len(x[1])):
        lines.append(f"\n### {source_name} ({len(rels_list)} rel{'s' if len(rels_list) != 1 else ''})\n")
        for label, target_name, evidence in rels_list:
            line = f"- **{label}** → {target_name}"
            if evidence:
                # Show a short snippet of evidence for context
                snippet = evidence[:120].replace("\n", " ")
                if len(evidence) > 120:
                    snippet += "…"
                line += f"  \n  *\"{snippet}\"*"
            lines.append(line)

    return "\n".join(lines)


def search_entities(query: str) -> str:
    if not state.knowledge_graph:
        return "Process a document with entity extraction first."
    if not query.strip():
        return "Type a name or keyword to search."
    results = state.knowledge_graph.find_entities_by_name(query.strip(), fuzzy=True)
    if not results:
        return f"No entities matching **{query}**."
    lines = [f"### Results for \"{query}\" ({len(results)} found)\n"]
    for ent in results[:10]:
        t = ent.type.value.replace("_", " ").title()
        m = len(ent.mentions)
        lines.append(f"**{ent.canonical_name}** ({t}) — {m} mention(s)")
        if ent.aliases:
            lines.append(f"  Also known as: {', '.join(ent.aliases[:3])}")
        co = state.knowledge_graph.get_entities_mentioned_together(ent.id)
        if co:
            lines.append(f"  Related to: {', '.join(e.canonical_name for e, _ in co[:3])}")
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# Build the Gradio app
# =============================================================================
EXAMPLE_QUESTIONS = [
    "What are the main sections of this document?",
    "Who are the key people mentioned?",
    "Summarize the key findings",
    "What dates are mentioned in the document?",
]


def create_demo():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        block_title_text_weight="600",
        block_border_width="1px",
        block_shadow="0 1px 3px rgba(0,0,0,0.08)",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )

    css = """
    .header-row { text-align: center; padding: 0.5rem 0; }
    .stat-box { background: var(--block-background-fill); border-radius: 8px;
                padding: 0.75rem 1rem; text-align: center; }
    .stat-box h2 { margin: 0; font-size: 1.6rem; }
    .stat-box p  { margin: 0; font-size: 0.8rem; opacity: 0.7; }
    footer { text-align: center; opacity: 0.6; font-size: 0.82rem; margin-top: 2rem; }
    """

    blocks_kw: dict[str, Any] = {"title": "RNSR — Document Intelligence"}
    if _gradio_major < 6:
        blocks_kw["theme"] = theme
        blocks_kw["css"] = css

    with gr.Blocks(**blocks_kw) as demo:

        # ---- Header ----
        gr.Markdown(
            "# RNSR — Document Intelligence\n"
            "Upload a PDF, explore its structure, tables and entities, then ask questions.",
            elem_classes=["header-row"],
        )

        # ============================================================
        # TOP-LEVEL TABS
        # ============================================================
        with gr.Tabs():

            # ============================
            # TAB 1 — Chat & Upload
            # ============================
            with gr.TabItem("💬 Chat", id="tab-chat"):
                with gr.Row():
                    # ---- left: upload + status ----
                    with gr.Column(scale=1, min_width=300):
                        gr.Markdown("#### Upload & Process")
                        file_input = gr.File(
                            label="PDF file",
                            file_types=[".pdf"],
                            type="filepath",
                            file_count="single",
                        )
                        with gr.Row():
                            extract_cb = gr.Checkbox(
                                label="Extract entities",
                                value=True,
                                info="Builds knowledge graph (~5-10 min). Uncheck to skip.",
                            )
                        process_btn = gr.Button("Process Document", variant="primary", size="lg")

                        status_md = gr.Markdown("Upload a PDF above to get started.")

                    # ---- right: chatbot ----
                    with gr.Column(scale=3, min_width=450):
                        chatbot_kw: dict[str, Any] = dict(
                            height=420,
                            show_label=False,
                            placeholder="Upload and process a document, then ask questions here…",
                            avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=rnsr"),
                        )
                        if _gradio_major >= 6:
                            chatbot_kw["buttons"] = ["copy"]
                        else:
                            chatbot_kw["show_copy_button"] = True
                        chatbot = gr.Chatbot(**chatbot_kw)

                        gr.Markdown("**Try:**", visible=True)
                        with gr.Row():
                            example_btns = []
                            for q in EXAMPLE_QUESTIONS:
                                label = q if len(q) <= 38 else q[:35] + "…"
                                btn = gr.Button(label, size="sm", variant="secondary")
                                example_btns.append((btn, q))

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question…",
                                show_label=False,
                                container=False,
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=90)
                        with gr.Row():
                            clear_btn = gr.Button("Clear chat", size="sm", variant="secondary")

            # ============================
            # TAB 2 — Document Structure
            # ============================
            with gr.TabItem("📄 Document", id="tab-doc"):
                gr.Markdown("#### Document Hierarchy")
                tree_output = gr.Textbox(
                    show_label=False,
                    lines=20,
                    max_lines=40,
                    interactive=False,
                    placeholder="Process a document to see its structure…",
                )
                refresh_tree_btn = gr.Button("Refresh", size="sm")

            # ============================
            # TAB 3 — Tables
            # ============================
            with gr.TabItem("📊 Tables", id="tab-tables"):
                gr.Markdown("#### Detected Tables")
                gr.Markdown(
                    "*Tables are automatically extracted during processing. "
                    "Select a table below to preview its contents with headers.*"
                )

                # Summary of all tables
                tables_summary = gr.Dataframe(
                    headers=["ID", "Title", "Rows", "Cols", "Headers", "Page"],
                    interactive=False,
                    wrap=True,
                    label="All tables",
                )

                gr.Markdown("---")
                gr.Markdown("#### Table Preview")
                with gr.Row():
                    table_dropdown = gr.Dropdown(
                        label="Select table",
                        choices=[],
                        interactive=True,
                        scale=3,
                    )
                    preview_btn = gr.Button("Preview", variant="primary", scale=1)

                table_preview = gr.Dataframe(
                    interactive=False,
                    wrap=True,
                    label="Table contents",
                )

                gr.Markdown("---")
                gr.Markdown("#### Query Builder")
                with gr.Row():
                    q_cols = gr.Textbox(label="Columns (comma-sep, blank = all)", placeholder="Name, Amount", scale=2)
                with gr.Row():
                    q_where_col = gr.Textbox(label="Where column", placeholder="Amount", scale=1)
                    q_where_op = gr.Dropdown(
                        label="Op",
                        choices=["==", "!=", ">", ">=", "<", "<=", "contains"],
                        value=">=",
                        scale=1,
                    )
                    q_where_val = gr.Textbox(label="Value", placeholder="1000", scale=1)
                with gr.Row():
                    q_order = gr.Textbox(label="Order by (prefix - for DESC)", placeholder="-Amount", scale=2)
                    q_limit = gr.Number(label="Limit", value=10, precision=0, scale=1)
                query_btn = gr.Button("Run Query", variant="primary")
                query_result = gr.Dataframe(interactive=False, wrap=True, label="Query results")

                gr.Markdown("---")
                gr.Markdown("#### Aggregation")
                with gr.Row():
                    agg_col = gr.Textbox(label="Column", placeholder="Revenue", scale=2)
                    agg_op = gr.Dropdown(
                        label="Operation",
                        choices=["sum", "avg", "count", "min", "max"],
                        value="sum",
                        scale=1,
                    )
                    agg_btn = gr.Button("Calculate", variant="secondary", scale=1)
                agg_result = gr.Markdown("")

            # ============================
            # TAB 4 — Knowledge Graph
            # ============================
            with gr.TabItem("🧠 Knowledge Graph", id="tab-kg"):
                with gr.Tabs():
                    with gr.TabItem("Entities"):
                        entities_md = gr.Markdown(
                            "*Process a document with entity extraction to see entities*"
                        )
                        refresh_ent_btn = gr.Button("Refresh", size="sm")

                    with gr.TabItem("Relationships"):
                        rels_md = gr.Markdown(
                            "*Process a document with entity extraction to see relationships*"
                        )
                        refresh_rel_btn = gr.Button("Refresh", size="sm")

                    with gr.TabItem("Search"):
                        search_input = gr.Textbox(
                            placeholder="Search entities by name…",
                            show_label=False,
                        )
                        search_btn = gr.Button("Search", size="sm")
                        search_md = gr.Markdown("")

        # ---- Footer ----
        gr.Markdown(
            "<footer>"
            "**RNSR** — Recursive Neural-Symbolic Retriever · "
            "[GitHub](https://github.com/theeufj/RNSR)"
            "</footer>"
        )

        # ============================================================
        # EVENT WIRING
        # ============================================================

        # --- Process document ---
        def _process_wrapper(file, extract_ent):
            for status, tbl_data, tbl_choices, tree_text in process_document(file, extract_ent):
                if _gradio_major >= 6:
                    dd = gr.Dropdown(choices=tbl_choices, value=tbl_choices[0] if tbl_choices else None)
                else:
                    dd = gr.update(choices=tbl_choices, value=tbl_choices[0] if tbl_choices else None)
                yield status, tbl_data, dd, tree_text

        process_btn.click(
            fn=_process_wrapper,
            inputs=[file_input, extract_cb],
            outputs=[status_md, tables_summary, table_dropdown, tree_output],
        )

        # --- Chat ---
        msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        clear_btn.click(fn=lambda: ("", []), inputs=[], outputs=[msg_input, chatbot])

        for btn, q in example_btns:
            btn.click(fn=lambda question=q: question, inputs=[], outputs=[msg_input])

        # --- Tree ---
        refresh_tree_btn.click(fn=_build_tree_text, inputs=[], outputs=[tree_output])

        # --- Tables ---
        preview_btn.click(fn=preview_table, inputs=[table_dropdown], outputs=[table_preview])
        table_dropdown.change(fn=preview_table, inputs=[table_dropdown], outputs=[table_preview])
        query_btn.click(
            fn=query_table_ui,
            inputs=[table_dropdown, q_cols, q_where_col, q_where_op, q_where_val, q_order, q_limit],
            outputs=[query_result],
        )
        agg_btn.click(
            fn=run_aggregation,
            inputs=[table_dropdown, agg_col, agg_op],
            outputs=[agg_result],
        )

        # --- Knowledge Graph ---
        refresh_ent_btn.click(fn=get_entities_visualization, inputs=[], outputs=[entities_md])
        refresh_rel_btn.click(fn=get_relationships_visualization, inputs=[], outputs=[rels_md])
        search_btn.click(fn=search_entities, inputs=[search_input], outputs=[search_md])
        search_input.submit(fn=search_entities, inputs=[search_input], outputs=[search_md])

    demo._rnsr_theme = theme
    demo._rnsr_css = css
    return demo


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # Check for API key (skip warning if Ollama is configured)
    has_cloud_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    has_ollama = os.getenv("OLLAMA_BASE_URL") or os.getenv("USE_OLLAMA") or os.getenv("LLM_PROVIDER", "").lower() == "ollama"
    if not has_cloud_key and not has_ollama:
        print("⚠️  Warning: No LLM provider configured.")
        print("   Set a cloud API key (GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY)")
        print("   or configure Ollama (OLLAMA_BASE_URL or LLM_PROVIDER=ollama)")
        print()

    print("🚀 Starting RNSR Demo...")
    print("   Open http://localhost:7860 in your browser")
    print()

    demo = create_demo()
    launch_kw: dict[str, Any] = dict(server_name="0.0.0.0", server_port=7860, share=False)
    if _gradio_major >= 6:
        launch_kw["theme"] = demo._rnsr_theme
        launch_kw["css"] = demo._rnsr_css
    demo.launch(**launch_kw)
