"""
Hierarchical Clustering for Semantic Segmentation (H-SBM)

Implements Section 4.2.2 of the research paper:
"For more advanced segmentation, we can employ unsupervised hierarchical 
clustering techniques. By clustering the sentence embeddings, we can 
discover 'Latent Topics' at various resolutions."

Features:
- Micro-Clusters: Groups of 5-10 sentences forming paragraph-level thoughts
- Macro-Clusters: Groups of micro-clusters forming chapter-level themes
- Synthetic Header Generation: LLM-generated titles for each cluster

This provides multi-resolution topic discovery when:
- Font histogram detects no variance (flat text)
- Semantic splitter produces too many fine-grained chunks
- Need hierarchical structure beyond simple breakpoints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from rnsr.models import DocumentNode, DocumentTree

logger = structlog.get_logger(__name__)


@dataclass
class TextCluster:
    """A cluster of semantically related text segments."""
    
    id: str
    texts: list[str]
    embeddings: np.ndarray | None = None
    centroid: np.ndarray | None = None
    children: list["TextCluster"] = field(default_factory=list)
    synthetic_header: str = ""
    level: int = 0  # 0 = leaf, 1 = micro, 2 = macro
    
    @property
    def full_text(self) -> str:
        """Concatenate all texts in this cluster."""
        return "\n\n".join(self.texts)
    
    @property
    def text_preview(self) -> str:
        """First 200 chars for summary."""
        full = self.full_text
        return full[:200] + "..." if len(full) > 200 else full


class HierarchicalSemanticClusterer:
    """
    Multi-resolution topic discovery via hierarchical clustering.
    
    Creates a two-level hierarchy:
    1. Micro-clusters (5-10 sentences) - paragraph-level thoughts
    2. Macro-clusters (groups of micro-clusters) - chapter-level themes
    """
    
    def __init__(
        self,
        micro_cluster_size: int = 7,  # Target sentences per micro-cluster
        macro_cluster_ratio: float = 0.3,  # Macro = 30% of micro count
        embed_provider: str | None = None,
        generate_headers: bool = True,
    ):
        """
        Initialize the clusterer.
        
        Args:
            micro_cluster_size: Target number of sentences per micro-cluster.
            macro_cluster_ratio: Ratio of macro to micro clusters.
            embed_provider: "gemini", "openai", or None for auto-detect.
            generate_headers: Whether to generate LLM headers for clusters.
        """
        self.micro_cluster_size = micro_cluster_size
        self.macro_cluster_ratio = macro_cluster_ratio
        self.embed_provider = embed_provider
        self.generate_headers = generate_headers
        self._embed_model = None
    
    def cluster_text(self, text: str) -> list[TextCluster]:
        """
        Perform hierarchical clustering on text.
        
        Args:
            text: Full document text.
            
        Returns:
            List of macro-clusters, each containing micro-clusters.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) < 3:
            # Too short for clustering
            return [TextCluster(
                id="cluster_0",
                texts=sentences,
                synthetic_header="Document Content",
            )]
        
        logger.info("clustering_text", sentences=len(sentences))
        
        # Get embeddings
        embeddings = self._get_embeddings(sentences)
        
        if embeddings is None:
            # Fallback to simple chunking
            return self._simple_cluster_fallback(sentences)
        
        # Step 1: Create micro-clusters (sentence-level → paragraph-level)
        micro_clusters = self._create_micro_clusters(sentences, embeddings)
        
        # Step 2: Create macro-clusters (micro-level → chapter-level)
        macro_clusters = self._create_macro_clusters(micro_clusters)
        
        # Step 3: Generate synthetic headers
        if self.generate_headers:
            self._generate_headers_for_clusters(macro_clusters)
        
        logger.info(
            "clustering_complete",
            micro_clusters=len(micro_clusters),
            macro_clusters=len(macro_clusters),
        )
        
        return macro_clusters
    
    def cluster_to_tree(self, text: str, title: str) -> DocumentTree:
        """
        Cluster text and convert to DocumentTree.
        
        Args:
            text: Full document text.
            title: Document title.
            
        Returns:
            DocumentTree with hierarchical structure.
        """
        macro_clusters = self.cluster_text(text)
        
        # Build tree
        root = DocumentNode(
            id="root",
            level=0,
            header=title,
        )
        
        for i, macro in enumerate(macro_clusters):
            # Macro cluster becomes H2
            macro_node = DocumentNode(
                id=f"macro_{i:03d}",
                level=1,
                header=macro.synthetic_header or f"Section {i + 1}",
            )
            
            if macro.children:
                # Add micro-clusters as H3
                for j, micro in enumerate(macro.children):
                    micro_node = DocumentNode(
                        id=f"micro_{i:03d}_{j:03d}",
                        level=2,
                        header=micro.synthetic_header or f"Subsection {i + 1}.{j + 1}",
                        content=micro.full_text,
                    )
                    macro_node.children.append(micro_node)
            else:
                # Macro is a leaf (no sub-clusters)
                macro_node.content = macro.full_text
            
            root.children.append(macro_node)
        
        return DocumentTree(
            title=title,
            root=root,
            total_nodes=self._count_nodes(root),
            ingestion_tier=2,
            ingestion_method="hierarchical_clustering",
        )
    
    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting
        # Handle common abbreviations
        text = text.replace("Dr.", "Dr")
        text = text.replace("Mr.", "Mr")
        text = text.replace("Mrs.", "Mrs")
        text = text.replace("Ms.", "Ms")
        text = text.replace("vs.", "vs")
        text = text.replace("etc.", "etc")
        text = text.replace("i.e.", "ie")
        text = text.replace("e.g.", "eg")
        
        # Split on sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty and very short
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return sentences
    
    def _get_embeddings(self, texts: list[str]) -> np.ndarray | None:
        """Get embeddings for a list of texts."""
        import os
        
        # Auto-detect provider
        provider = self.embed_provider
        if provider is None:
            if os.getenv("GOOGLE_API_KEY"):
                provider = "gemini"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                logger.warning("no_embedding_api_key")
                return None
        
        try:
            if provider == "gemini":
                return self._get_gemini_embeddings(texts)
            elif provider == "openai":
                return self._get_openai_embeddings(texts)
            else:
                logger.warning("unknown_embed_provider", provider=provider)
                return None
        except Exception as e:
            logger.warning("embedding_failed", error=str(e))
            return None
    
    def _get_gemini_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings using Gemini."""
        from google import genai
        
        client = genai.Client()
        
        # Embed texts individually
        embeddings = []
        
        for text in texts:
            result = client.models.embed_content(
                model="models/text-embedding-005",
                contents=text,
            )
            
            if result.embeddings is not None and len(result.embeddings) > 0:
                embeddings.append(result.embeddings[0].values)
        
        return np.array(embeddings)
    
    def _get_openai_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings using OpenAI."""
        from openai import OpenAI
        
        client = OpenAI()
        
        # Batch texts
        embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            
            for item in response.data:
                embeddings.append(item.embedding)
        
        return np.array(embeddings)
    
    def _create_micro_clusters(
        self,
        sentences: list[str],
        embeddings: np.ndarray,
    ) -> list[TextCluster]:
        """
        Create micro-clusters (paragraph-level) via agglomerative clustering.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        
        n_sentences = len(sentences)
        target_clusters = max(3, n_sentences // self.micro_cluster_size)
        
        # Compute linkage
        distances = pdist(embeddings, metric='cosine')
        Z = linkage(distances, method='ward')
        
        # Cut tree to get target number of clusters
        labels = fcluster(Z, t=target_clusters, criterion='maxclust')
        
        # Group sentences by cluster
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = {
                    'texts': [],
                    'embeddings': [],
                    'indices': [],
                }
            clusters[label]['texts'].append(sentences[i])
            clusters[label]['embeddings'].append(embeddings[i])
            clusters[label]['indices'].append(i)
        
        # Convert to TextCluster objects, sorted by first sentence index
        micro_clusters = []
        for label, data in sorted(clusters.items(), key=lambda x: min(x[1]['indices'])):
            emb_array = np.array(data['embeddings'])
            cluster = TextCluster(
                id=f"micro_{label}",
                texts=data['texts'],
                embeddings=emb_array,
                centroid=np.mean(emb_array, axis=0),
                level=1,
            )
            micro_clusters.append(cluster)
        
        logger.debug("micro_clusters_created", count=len(micro_clusters))
        return micro_clusters
    
    def _create_macro_clusters(
        self,
        micro_clusters: list[TextCluster],
    ) -> list[TextCluster]:
        """
        Create macro-clusters (chapter-level) from micro-clusters.
        """
        if len(micro_clusters) <= 3:
            # Too few for macro clustering
            for i, micro in enumerate(micro_clusters):
                micro.level = 2  # Treat as macro
                micro.id = f"macro_{i}"
            return micro_clusters
        
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist
        
        # Get centroids
        centroids = np.array([c.centroid for c in micro_clusters if c.centroid is not None])
        
        if len(centroids) < 3:
            return micro_clusters
        
        # Target macro clusters
        target_macros = max(2, int(len(micro_clusters) * self.macro_cluster_ratio))
        
        # Cluster centroids
        distances = pdist(centroids, metric='cosine')
        Z = linkage(distances, method='ward')
        labels = fcluster(Z, t=target_macros, criterion='maxclust')
        
        # Group micro-clusters by macro label
        macro_groups = {}
        for i, label in enumerate(labels):
            if label not in macro_groups:
                macro_groups[label] = []
            macro_groups[label].append(micro_clusters[i])
        
        # Create macro clusters
        macro_clusters = []
        for label, micros in sorted(macro_groups.items()):
            # Combine all texts
            all_texts = []
            for micro in micros:
                all_texts.extend(micro.texts)
            
            macro = TextCluster(
                id=f"macro_{label}",
                texts=all_texts,
                children=micros,
                level=2,
            )
            macro_clusters.append(macro)
        
        logger.debug("macro_clusters_created", count=len(macro_clusters))
        return macro_clusters
    
    def _generate_headers_for_clusters(
        self,
        clusters: list[TextCluster],
    ) -> None:
        """Generate synthetic headers for all clusters using LLM."""
        for cluster in clusters:
            # Generate header for macro cluster
            cluster.synthetic_header = self._generate_single_header(cluster.text_preview)
            
            # Generate headers for children
            for child in cluster.children:
                child.synthetic_header = self._generate_single_header(child.text_preview)
    
    def _generate_single_header(self, text_preview: str) -> str:
        """Generate a single header via LLM."""
        import os
        
        prompt = f"""Generate a concise 3-7 word descriptive title for this text section:

{text_preview}

Return ONLY the title, nothing else."""
        
        # Try Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from google import genai
                from google.genai import types as _gentypes
                
                client = genai.Client()
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                    config=_gentypes.GenerateContentConfig(temperature=0.0),
                )
                if response.text is not None:
                    header = response.text.strip().strip('"').strip("'")
                    if 3 <= len(header) <= 100:
                        return header
            except Exception:
                pass
        
        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=30,
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                if content is not None:
                    header = content.strip().strip('"').strip("'")
                    if 3 <= len(header) <= 100:
                        return header
            except Exception:
                pass
        
        # Fallback: first few words
        words = text_preview.split()[:5]
        return " ".join(words) + "..." if words else "Section"
    
    def _simple_cluster_fallback(
        self,
        sentences: list[str],
    ) -> list[TextCluster]:
        """Simple clustering when embeddings unavailable."""
        # Group sentences into chunks
        chunk_size = self.micro_cluster_size
        clusters = []
        
        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            cluster = TextCluster(
                id=f"cluster_{i // chunk_size}",
                texts=chunk,
                synthetic_header=f"Section {i // chunk_size + 1}",
                level=2,
            )
            clusters.append(cluster)
        
        return clusters
    
    def _count_nodes(self, node: DocumentNode) -> int:
        """Count total nodes in tree."""
        return 1 + sum(self._count_nodes(c) for c in node.children)


def cluster_document_hierarchically(
    pdf_path: Path | str,
    title: str | None = None,
) -> DocumentTree:
    """
    Convenience function for hierarchical clustering of a PDF.
    
    Args:
        pdf_path: Path to the PDF file.
        title: Document title (defaults to filename).
        
    Returns:
        DocumentTree with hierarchical structure.
        
    Example:
        tree = cluster_document_hierarchically("flat_document.pdf")
        for section in tree.root.children:
            print(f"{section.header}: {len(section.children)} subsections")
    """
    import fitz
    
    pdf_path = Path(pdf_path)
    title = title or pdf_path.stem
    
    # Extract text
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pages_text.append(page.get_text("text"))
    doc.close()
    
    full_text = "\n\n".join(pages_text)
    
    # Cluster
    clusterer = HierarchicalSemanticClusterer()
    return clusterer.cluster_to_tree(full_text, title)
