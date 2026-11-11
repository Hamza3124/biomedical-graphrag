"""
MedQuAD XML Parsing and Vector Indexing Script
Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
PFA 5IADA - EMSI

Parses MedQuAD XML files for targeted pathologies (Alzheimer's Disease, Dementias,
related therapeutics), chunks text contextually, generates biomedical embeddings
(PubMedBERT / BioBERT), and indexes them into Qdrant and/or ChromaDB.
"""

import os
import sys

# Configure offline mode so huggingface loads directly from local cache without timeouts
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import uuid
import logging
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from sentence_transformers import SentenceTransformer

# Optional dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Vector.IndexMedQuAD")

DEFAULT_EMBEDDING_MODEL = "NeuML/pubmedbert-base-embeddings"
COLLECTION_NAME = "medquad_disease_qa"

# Relevant subdirectories in MedQuAD containing target diseases and therapeutics
RELEVANT_SUBDIRS = [
    "3_GHR_QA",
    "4_MPlus_Health_Topics_QA",
    "5_NIDDK_QA",
    "6_NINDS_QA",
    "7_SeniorHealth_QA",
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA"
]


class MedQuADParser:
    """Parses MedQuAD XML files and extracts structured Q&A records with metadata."""

    TARGET_KEYWORDS = [
        "alzheimer", "dementia", "cognitive",
        "donepezil", "galantamine", "memantine", "rivastigmine",
        "diabetes"
    ]

    def __init__(self, medquad_dir: Path):
        self.medquad_dir = medquad_dir
        if not self.medquad_dir.exists():
            raise FileNotFoundError(f"MedQuAD root directory not found: {self.medquad_dir}")

    def find_target_xml_files(self, keywords: Optional[List[str]] = None) -> List[Path]:
        """Scans relevant MedQuAD directories and identifies XML files matching target pathology/drugs."""
        kw_list = [k.lower() for k in (keywords or self.TARGET_KEYWORDS)]
        matched_files = []

        target_dirs = []
        for subdir in RELEVANT_SUBDIRS:
            candidate = self.medquad_dir / subdir
            if candidate.is_dir():
                target_dirs.append(candidate)

        if not target_dirs:
            target_dirs = [self.medquad_dir]

        logger.info(f"Scanning MedQuAD directories ({len(target_dirs)} folders) for keywords: {kw_list}...")
        for tdir in target_dirs:
            for root, _, files in os.walk(tdir):
                for f in files:
                    if f.endswith(".xml"):
                        p = Path(root) / f
                        try:
                            with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                                header = fp.read(1000).lower()
                                if any(k in header for k in kw_list):
                                    matched_files.append(p)
                        except Exception:
                            continue

        logger.info(f"Identified {len(matched_files):,} matching MedQuAD XML files.")
        return matched_files

    def parse_file(self, xml_path: Path) -> List[Dict[str, Any]]:
        """Parses a single MedQuAD XML file and extracts all Q/A pairs with rich metadata."""
        records = []
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            doc_id = root.get("id", xml_path.stem)
            source = root.get("source", xml_path.parent.name)
            url = root.get("url", "")

            focus_el = root.find("Focus")
            focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""

            cui = ""
            cui_el = root.find(".//UMLS/CUIs/CUI")
            if cui_el is not None and cui_el.text:
                cui = cui_el.text.strip()

            qa_pairs = root.findall(".//QAPair")
            for qa in qa_pairs:
                pid = qa.get("pid", "1")
                q_el = qa.find("Question")
                a_el = qa.find("Answer")

                if q_el is None or a_el is None or not a_el.text:
                    continue

                question = q_el.text.strip()
                answer = a_el.text.strip()
                qid = q_el.get("qid", f"{doc_id}-{pid}")
                qtype = q_el.get("qtype", "general")

                records.append({
                    "doc_id": doc_id,
                    "source": source,
                    "url": url,
                    "disease_name": focus,
                    "cui": cui,
                    "pid": pid,
                    "qid": qid,
                    "question_type": qtype,
                    "question": question,
                    "answer": answer,
                    "file_path": str(xml_path)
                })

        except Exception as e:
            logger.debug(f"Error parsing XML file {xml_path}: {e}")

        return records


class ContextualChunker:
    """Chunks biomedical Q&A records into retrieval-optimized chunks with headers."""

    def __init__(self, max_words: int = 350, overlap_words: int = 50):
        self.max_words = max_words
        self.overlap_words = overlap_words

    def chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        question = record["question"]
        answer = record["answer"]
        disease = record["disease_name"]
        qtype = record["question_type"]

        words = answer.split()
        chunks = []

        source_clean = record.get('source', '').replace(' ', '_')
        if len(words) <= self.max_words:
            content = f"Pathology: {disease}\nTopic: {qtype}\nQuestion: {question}\nAnswer: {answer}"
            chunk = dict(record)
            chunk["chunk_id"] = f"{source_clean}_{record['doc_id']}_{record['pid']}_0"
            chunk["chunk_index"] = 0
            chunk["text"] = content
            chunk["word_count"] = len(words)
            chunks.append(chunk)
        else:
            start = 0
            chunk_idx = 0
            step = self.max_words - self.overlap_words
            while start < len(words):
                end = min(start + self.max_words, len(words))
                sub_answer = " ".join(words[start:end])
                content = (
                    f"Pathology: {disease}\nTopic: {qtype} (Part {chunk_idx + 1})\n"
                    f"Question: {question}\nAnswer: {sub_answer}"
                )
                chunk = dict(record)
                chunk["chunk_id"] = f"{source_clean}_{record['doc_id']}_{record['pid']}_{chunk_idx}"
                chunk["chunk_index"] = chunk_idx
                chunk["text"] = content
                chunk["word_count"] = end - start
                chunks.append(chunk)
                chunk_idx += 1
                start += step

        return chunks


class BiomedicalVectorIndexer:
    """Generates embeddings and indexes chunks into ChromaDB and/or Qdrant."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        chroma_persist_dir: Optional[Path] = None,
        qdrant_url: Optional[str] = None
    ):
        self.model_name = model_name
        self.chroma_persist_dir = chroma_persist_dir or (Path.cwd() / "data" / "vector_store" / "chroma")
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")

        logger.info(f"Loading embedding model '{self.model_name}' (offline mode)...")
        self.model = SentenceTransformer(self.model_name, local_files_only=True)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Encodes a list of texts into dense vectors."""
        logger.info(f"Encoding {len(texts):,} text chunks (batch size: {batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def index_chroma(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """Stores chunks and embeddings in persistent ChromaDB."""
        import chromadb
        logger.info(f"Indexing into local ChromaDB at {self.chroma_persist_dir}...")
        client = chromadb.PersistentClient(path=str(self.chroma_persist_dir))

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "model": self.model_name}
        )

        seen_ids = set()
        clean_ids = []
        for cid in [c["chunk_id"] for c in chunks]:
            final_id = cid
            idx = 1
            while final_id in seen_ids:
                final_id = f"{cid}_{idx}"
                idx += 1
            seen_ids.add(final_id)
            clean_ids.append(final_id)

        ids = clean_ids
        documents = [c["text"] for c in chunks]
        metadatas = [{
            "doc_id": str(c["doc_id"]),
            "source": str(c["source"]),
            "disease_name": str(c["disease_name"]),
            "question_type": str(c["question_type"]),
            "question": str(c["question"])[:250],
            "cui": str(c["cui"]),
            "url": str(c["url"])
        } for c in chunks]

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size]
            )

        count = collection.count()
        logger.info(f"ChromaDB indexing complete. Total collection count: {count:,}")
        return count

    def index_qdrant(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> bool:
        """Stores chunks and embeddings in Qdrant instance via REST/gRPC."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels

            logger.info(f"Connecting to Qdrant at {self.qdrant_url}...")
            client = QdrantClient(url=self.qdrant_url, timeout=5.0)

            collections = client.get_collections().collections
            col_names = [c.name for c in collections]

            if COLLECTION_NAME not in col_names:
                logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim={self.embedding_dim}, metric=Cosine)...")
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_dim,
                        distance=qmodels.Distance.COSINE
                    )
                )

            points = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                pt_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "source": chunk["source"],
                    "disease_name": chunk["disease_name"],
                    "question_type": chunk["question_type"],
                    "question": chunk["question"],
                    "answer": chunk["answer"][:1000],
                    "cui": chunk["cui"],
                    "url": chunk["url"],
                    "text": chunk["text"]
                }
                points.append(qmodels.PointStruct(id=pt_id, vector=emb, payload=payload))

            batch_size = 200
            for i in range(0, len(points), batch_size):
                client.upsert(collection_name=COLLECTION_NAME, points=points[i:i + batch_size])

            logger.info(f"Successfully uploaded {len(points):,} points to Qdrant collection '{COLLECTION_NAME}'.")
            return True

        except Exception as e:
            logger.warning(f"Qdrant indexing skipped / not reachable at {self.qdrant_url}: {e}")
            logger.info("Tip: Start Qdrant with 'docker-compose up -d qdrant' to enable Docker-based indexing.")
            return False


def locate_medquad_dir(root: Path) -> Path:
    candidates = [
        root / "data" / "raw" / "MedQuAD-master",
        root / "MedQuAD-master" / "MedQuAD-master",
        root / "MedQuAD-master"
    ]
    for c in candidates:
        if c.is_dir() and any((c / d).is_dir() for d in ["3_GHR_QA", "4_MPlus_Health_Topics_QA", "6_NINDS_QA"]):
            return c
    raise FileNotFoundError(f"Could not locate valid MedQuAD directory in {[str(p) for p in candidates]}")


def main():
    parser = argparse.ArgumentParser(description="Parse MedQuAD XML files and index vectors.")
    parser.add_argument("--backend", choices=["chroma", "qdrant", "both"], default="both", help="Vector store target")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant REST URL")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--max-words", type=int, default=350, help="Max words per chunk")
    parser.add_argument("--output-csv", type=str, default="data/processed/medquad_chunks.csv", help="CSV export for chunks")

    args = parser.parse_args()
    root = Path.cwd()

    medquad_dir = locate_medquad_dir(root)
    logger.info(f"Found MedQuAD directory: {medquad_dir}")

    parser_obj = MedQuADParser(medquad_dir=medquad_dir)
    target_files = parser_obj.find_target_xml_files()

    all_records = []
    for xml_file in target_files:
        all_records.extend(parser_obj.parse_file(xml_file))

    logger.info(f"Extracted {len(all_records):,} raw Q/A pairs from {len(target_files):,} XML documents.")

    chunker = ContextualChunker(max_words=args.max_words)
    all_chunks = []
    for r in all_records:
        all_chunks.extend(chunker.chunk_record(r))

    logger.info(f"Generated {len(all_chunks):,} contextual chunks.")

    # Save chunks table to CSV
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_chunks = pd.DataFrame(all_chunks)
    df_chunks.to_csv(out_csv, index=False)
    logger.info(f"Saved chunk metadata table to {out_csv}")

    # Generate Embeddings
    indexer = BiomedicalVectorIndexer(qdrant_url=args.qdrant_url)
    texts = [c["text"] for c in all_chunks]
    embeddings = indexer.embed_texts(texts, batch_size=args.batch_size)

    # Ingest into vector stores
    if args.backend in ("chroma", "both"):
        indexer.index_chroma(all_chunks, embeddings)

    if args.backend in ("qdrant", "both"):
        indexer.index_qdrant(all_chunks, embeddings)

    print("\n" + "=" * 60)
    print("MEDQUAD VECTOR INDEXING SUCCESSFULLY COMPLETED")
    print("=" * 60)
    print(f"Target XML files parsed  : {len(target_files):,}")
    print(f"Q&A records extracted    : {len(all_records):,}")
    print(f"Contextual chunks indexed: {len(all_chunks):,}")
    print(f"Embedding model          : {indexer.model_name} (dim={indexer.embedding_dim})")
    print(f"Local ChromaDB location  : {indexer.chroma_persist_dir}")
    print(f"Chunks CSV table         : {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
