#!/usr/bin/env python3
"""Run all test PDFs through RNSR's timeline, contradiction, and multi-doc features.

Usage:
    python scripts/run_test_pdfs.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnsr import RNSRClient, DocumentStore, ingest_document, build_skeleton_index
from rnsr.indexing.knowledge_graph import InMemoryKnowledgeGraph
from rnsr.extraction.timeline_extractor import extract_timeline, format_timeline
from rnsr.analysis.contradiction_detector import (
    detect_document_contradictions,
    detect_cross_document_contradictions,
)

TEST_DIR = Path(__file__).resolve().parent.parent / "test-documents"
CACHE_DIR = ".rnsr_test_cache"

client = RNSRClient(cache_dir=CACHE_DIR)

DIVIDER = "=" * 70


def section(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# =========================================================================
# 1. Timeline: Meridian Project History
# =========================================================================
def test_timeline_project():
    section("TIMELINE: Meridian Infrastructure Project")
    pdf = str(TEST_DIR / "Timeline - Meridian Project History.pdf")

    print("\n[1/3] Ingesting document...")
    t0 = time.time()
    result = ingest_document(pdf)
    skeleton, kv_store = build_skeleton_index(result.tree)
    print(f"      Done in {time.time()-t0:.1f}s -- {len(skeleton)} nodes")

    print("[2/3] Extracting entities...")
    t0 = time.time()
    cache_key = client._get_cache_key(Path(pdf))
    kg = client._get_or_create_knowledge_graph(
        cache_key=cache_key,
        skeleton=skeleton,
        kv_store=kv_store,
        doc_id=result.tree.id,
    )
    stats = kg.get_stats()
    print(f"      Done in {time.time()-t0:.1f}s -- {stats.get('entity_count',0)} entities, {stats.get('relationship_count',0)} relationships")

    print("[3/3] Extracting timeline...")
    events = extract_timeline(kg)
    dated = [e for e in events if e.date_parsed is not None]
    undated = [e for e in events if e.date_parsed is None]
    print(f"      Found {len(events)} events ({len(dated)} dated, {len(undated)} undated)")
    if dated:
        print("\n      DATED EVENTS:")
        for ev in dated[:15]:
            d = ev.date_parsed.strftime("%d %b %Y") if ev.date_parsed else ev.date_str
            print(f"        {d}  --  {ev.description[:80]}")
    if undated:
        print(f"\n      UNDATED EVENTS ({len(undated)}):")
        for ev in undated[:5]:
            print(f"        {ev.date_str[:40]}  --  {ev.description[:60]}")


# =========================================================================
# 2. Timeline: Legal Case
# =========================================================================
def test_timeline_legal():
    section("TIMELINE: Baxter v Thornton")
    pdf = str(TEST_DIR / "Timeline - Baxter v Thornton.pdf")

    print("\n[1/3] Ingesting document...")
    t0 = time.time()
    result = ingest_document(pdf)
    skeleton, kv_store = build_skeleton_index(result.tree)
    print(f"      Done in {time.time()-t0:.1f}s -- {len(skeleton)} nodes")

    print("[2/3] Extracting entities...")
    t0 = time.time()
    cache_key = client._get_cache_key(Path(pdf))
    kg = client._get_or_create_knowledge_graph(
        cache_key=cache_key,
        skeleton=skeleton,
        kv_store=kv_store,
        doc_id=result.tree.id,
    )
    stats = kg.get_stats()
    print(f"      Done in {time.time()-t0:.1f}s -- {stats.get('entity_count',0)} entities, {stats.get('relationship_count',0)} relationships")

    print("[3/3] Extracting timeline...")
    events = extract_timeline(kg)
    dated = [e for e in events if e.date_parsed is not None]
    print(f"      Found {len(events)} events ({len(dated)} dated)")
    if dated:
        print("\n      DATED EVENTS:")
        for ev in dated[:15]:
            d = ev.date_parsed.strftime("%d %b %Y") if ev.date_parsed else ev.date_str
            print(f"        {d}  --  {ev.description[:80]}")


# =========================================================================
# 3. Single-Doc Contradictions: Greenfield Annual Report
# =========================================================================
def test_contradictions_single():
    section("CONTRADICTIONS (single doc): Greenfield Annual Report")
    pdf = str(TEST_DIR / "Contradictions - Greenfield Annual Report.pdf")

    print("\n[1/3] Ingesting document...")
    t0 = time.time()
    result = ingest_document(pdf)
    skeleton, kv_store = build_skeleton_index(result.tree)
    print(f"      Done in {time.time()-t0:.1f}s -- {len(skeleton)} nodes")

    print("[2/3] Extracting entities...")
    t0 = time.time()
    cache_key = client._get_cache_key(Path(pdf))
    kg = client._get_or_create_knowledge_graph(
        cache_key=cache_key,
        skeleton=skeleton,
        kv_store=kv_store,
        doc_id=result.tree.id,
    )
    stats = kg.get_stats()
    print(f"      Done in {time.time()-t0:.1f}s -- {stats.get('entity_count',0)} entities, {stats.get('relationship_count',0)} relationships")

    print("[3/3] Detecting contradictions...")
    contradictions = detect_document_contradictions(
        kg=kg, skeleton=skeleton, kv_store=kv_store,
    )
    print(f"      Found {len(contradictions)} contradiction(s)")
    for i, c in enumerate(contradictions, 1):
        print(f"\n      [{i}] {c.type.upper()} ({c.confidence:.0%} confidence)")
        print(f"          Claim 1 ({c.source_1}): {c.claim_1[:100]}")
        print(f"          Claim 2 ({c.source_2}): {c.claim_2[:100]}")
        if c.explanation:
            print(f"          Reason: {c.explanation[:100]}")


# =========================================================================
# 4. Cross-Doc Contradictions: Expert Reports + Incident Report
# =========================================================================
def test_crossdoc_contradictions():
    section("CROSS-DOC CONTRADICTIONS: Expert Reports + Incident Report")

    pdfs = [
        ("expert_a", str(TEST_DIR / "CrossDoc - Expert Report A (Dr Hartley).pdf")),
        ("expert_b", str(TEST_DIR / "CrossDoc - Expert Report B (Dr Webb).pdf")),
        ("incident", str(TEST_DIR / "CrossDoc - Employer Incident Report.pdf")),
    ]

    doc_tuples = []
    kg = InMemoryKnowledgeGraph()

    for doc_id, pdf in pdfs:
        print(f"\n  [{doc_id}] Ingesting {Path(pdf).name}...")
        t0 = time.time()
        result = ingest_document(pdf)
        skeleton, kv_store = build_skeleton_index(result.tree)
        print(f"          {len(skeleton)} nodes in {time.time()-t0:.1f}s")

        print(f"          Extracting entities...")
        t0 = time.time()
        cache_key = client._get_cache_key(Path(pdf))
        doc_kg = client._get_or_create_knowledge_graph(
            cache_key=cache_key,
            skeleton=skeleton,
            kv_store=kv_store,
            doc_id=doc_id,
        )
        stats = doc_kg.get_stats()
        print(f"          {stats.get('entity_count',0)} entities, {stats.get('relationship_count',0)} rels in {time.time()-t0:.1f}s")

        doc_tuples.append((doc_id, skeleton, kv_store))

        # Merge entities into the shared KG
        all_entities = doc_kg.find_entities_in_document(doc_id)
        for ent in all_entities:
            kg.add_entity(ent)
            for rel in doc_kg.get_entity_relationships(ent.id):
                kg.add_relationship(rel)

    print(f"\n  Workspace KG: {kg.get_stats()}")

    print("\n  Detecting cross-document contradictions...")
    contradictions = detect_cross_document_contradictions(
        kg=kg, documents=doc_tuples,
    )
    print(f"  Found {len(contradictions)} contradiction(s)")
    for i, c in enumerate(contradictions, 1):
        print(f"\n    [{i}] {c.type.upper()} ({c.confidence:.0%} confidence)")
        print(f"        Claim 1 ({c.source_1}): {c.claim_1[:120]}")
        print(f"        Claim 2 ({c.source_2}): {c.claim_2[:120]}")
        if c.explanation:
            print(f"        Reason: {c.explanation[:120]}")


# =========================================================================
# 5. Multi-Document Workspace (DocumentStore)
# =========================================================================
def test_multidoc_workspace():
    section("MULTI-DOC WORKSPACE: All CrossDoc PDFs via DocumentStore")

    import shutil
    store_path = Path(".rnsr_test_store")
    if store_path.exists():
        shutil.rmtree(store_path)

    store = DocumentStore(str(store_path))
    pdfs = [
        str(TEST_DIR / "CrossDoc - Expert Report A (Dr Hartley).pdf"),
        str(TEST_DIR / "CrossDoc - Expert Report B (Dr Webb).pdf"),
        str(TEST_DIR / "CrossDoc - Employer Incident Report.pdf"),
    ]

    for pdf in pdfs:
        print(f"\n  Adding: {Path(pdf).name}")
        t0 = time.time()
        doc_id = store.add_document(pdf)
        print(f"    -> doc_id={doc_id} in {time.time()-t0:.1f}s")

    print(f"\n  Store has {len(store)} documents")
    for d in store.list_documents():
        print(f"    - {d['title']} ({d['id']}) -- {d['node_count']} nodes")

    print("\n  Building workspace KG...")
    t0 = time.time()
    workspace_kg = store.build_workspace_kg()
    stats = workspace_kg.get_stats()
    print(f"    Done in {time.time()-t0:.1f}s -- {stats.get('entity_count',0)} entities, {stats.get('relationship_count',0)} rels")

    print("\n  Extracting cross-doc timeline...")
    events = extract_timeline(workspace_kg)
    dated = [e for e in events if e.date_parsed is not None]
    print(f"    Found {len(events)} events ({len(dated)} dated)")
    for ev in dated[:10]:
        d = ev.date_parsed.strftime("%d %b %Y") if ev.date_parsed else ev.date_str
        print(f"      {d}  --  {ev.description[:70]}")

    # Cleanup
    shutil.rmtree(store_path, ignore_errors=True)


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print("RNSR Feature Test Suite")
    print(f"Test documents: {TEST_DIR}")
    print(f"Cache: {CACHE_DIR}")

    t_total = time.time()

    test_timeline_project()
    test_timeline_legal()
    test_contradictions_single()
    test_crossdoc_contradictions()
    test_multidoc_workspace()

    section("ALL TESTS COMPLETE")
    print(f"\n  Total time: {time.time()-t_total:.1f}s\n")
