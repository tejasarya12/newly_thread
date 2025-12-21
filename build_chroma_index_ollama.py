"""
Build Chroma index using Ollama embeddings (no Hugging Face)
Replaces any previous HF-based indexing with Ollama-only vectors.
"""
import os
import glob
import logging
from typing import List, Dict

import chromadb
from ollama import Client

from config_threaded import RETRIEVAL, CHUNKING, INGESTION, OLLAMA

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_text_files(root: str, exts: List[str]) -> List[Dict]:
    """
    Simple loader: reads .txt/.md files from a folder.
    Returns list of {id, text, metadata}.
    """
    docs = []
    patterns = [os.path.join(root, f"**/*{ext}") for ext in exts]

    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                    if not text:
                        continue

                    docs.append(
                        {
                            "id": os.path.abspath(path),
                            "text": text,
                            "metadata": {
                                "source": os.path.basename(path),
                                "source_file": os.path.abspath(path),
                                "source_category": "general",
                            },
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}", exc_info=True)

    logger.info(f"Loaded {len(docs)} raw documents from {root}")
    return docs


def chunk_text(text: str, doc_meta: Dict, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """
    Simple fixed-size chunker based on characters.
    Adapts to CHUNKING['chunk_size'] and ['chunk_overlap'].
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "content": chunk,
                    "metadata": {
                        **doc_meta,
                        "chunk_start": start,
                        "chunk_end": end,
                    },
                }
            )
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if start >= length:
            break

    return chunks


def embed_texts_ollama(texts: List[str], model: str, base_url: str) -> List[List[float]]:
    """Compute embeddings for a list of texts using Ollama."""
    client = Client(host=base_url)
    embeddings: List[List[float]] = []

    for i, t in enumerate(texts):
        try:
            resp = client.embeddings(model=model, prompt=t)
            emb = resp.get("embedding")
            if emb is None:
                raise RuntimeError("No 'embedding' field in Ollama response")
            embeddings.append(emb)
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{len(texts)} texts...")
        except Exception as e:
            logger.error(f"Embedding generation failed for text {i}: {t[:80]!r}: {e}")
            # Fallback to zero-vector of appropriate size if we have at least one
            if embeddings:
                embeddings.append([0.0] * len(embeddings[0]))
            else:
                raise

    return embeddings


def main():
    """Main indexing function"""
    # Config
    vector_cfg = RETRIEVAL["vector_db"]
    chunk_cfg = CHUNKING
    ingestion_cfg = INGESTION

    persist_dir = vector_cfg["persist_directory"]
    collection_name = vector_cfg["collection_name"]
    watch_folder = ingestion_cfg["watch_folder"]

    chunk_size = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]

    embedding_model = RETRIEVAL.get("embedding_model", "nomic-embed-text")
    ollama_base = OLLAMA["base_url"]

    logger.info("=" * 60)
    logger.info("Building Chroma index with Ollama embeddings")
    logger.info("=" * 60)
    logger.info(f"Chroma path: {persist_dir}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Source folder: {watch_folder}")
    logger.info(f"Ollama embeddings model: {embedding_model}")
    logger.info(f"Ollama base URL: {ollama_base}")
    logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")

    # 1. Load documents
    if not os.path.exists(watch_folder):
        logger.warning(f"Watch folder '{watch_folder}' does not exist. Creating it...")
        os.makedirs(watch_folder, exist_ok=True)
        logger.info(f"Please add .txt or .md files to '{watch_folder}' and run again.")
        return

    docs = load_text_files(root=watch_folder, exts=[".txt", ".md"])

    if not docs:
        logger.warning(f"No documents found in {watch_folder}. Add .txt or .md files and try again.")
        return

    # 2. Chunk documents
    logger.info("Chunking documents...")
    all_chunks: List[Dict] = []
    for doc in docs:
        chunks = chunk_text(
            text=doc["text"],
            doc_meta=doc["metadata"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(
            {
                "id": f"{doc['id']}::chunk-{i}",
                "content": c["content"],
                "metadata": c["metadata"],
            }
            for i, c in enumerate(chunks)
        )

    logger.info(f"Created {len(all_chunks)} chunks from {len(docs)} documents")

    if not all_chunks:
        logger.warning("No chunks to index; exiting.")
        return

    # 3. Initialize Chroma
    logger.info("Initializing Chroma client...")
    chroma_client = chromadb.PersistentClient(path=persist_dir)

    # Optional: wipe existing collection (so you don't mix HF + Ollama embeddings)
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"Deleted existing collection '{collection_name}' (to avoid mixing HF + Ollama embeddings)")
    except Exception:
        logger.info(f"Collection '{collection_name}' does not exist yet (will be created)")

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 4. Embed and add in batches
    batch_size = 32  # Smaller batches to avoid overwhelming Ollama
    logger.info(f"Starting embedding + indexing into Chroma (batch size: {batch_size})...")

    total_batches = (len(all_chunks) + batch_size - 1) // batch_size

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        ids = [c["id"] for c in batch]
        texts = [c["content"] for c in batch]
        metas = [c["metadata"] for c in batch]

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            vectors = embed_texts_ollama(texts, model=embedding_model, base_url=ollama_base)

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metas,
                embeddings=vectors,
            )

            logger.info(f"✅ Indexed batch {batch_num}/{total_batches}")

        except Exception as e:
            logger.error(f"Failed to index batch {batch_num}: {e}", exc_info=True)
            continue

    # 5. Verify
    count = collection.count()
    logger.info("=" * 60)
    logger.info(f"✅ Finished building Chroma index with Ollama embeddings")
    logger.info(f"Total chunks indexed: {count}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
