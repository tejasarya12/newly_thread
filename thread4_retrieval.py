"""
Thread 4: RAG Retrieval
Fetches relevant context from Vector DB.
"""
import logging
import queue
import chromadb
from config_threaded import SYSTEM, THREADS

logger = logging.getLogger("Thread-4-Retrieval")

def retrieval_thread_main(manager, input_queue, output_queue, shutdown_event):
    # Initialize ChromaDB
    try:
        client = chromadb.PersistentClient(path=SYSTEM['vector_db_path'])
        collection = client.get_or_create_collection("company_documents")
        logger.info("ChromaDB Connected")
    except Exception as e:
        logger.error(f"ChromaDB Init Failed: {e}")
        return

    while not shutdown_event.is_set():
        try:
            data = input_queue.get(timeout=0.1)
            query = data["cleaned"]
            
            # Perform Search
            results = collection.query(
                query_texts=[query],
                n_results=THREADS['retrieval']['top_k']
            )
            
            # Extract Documents
            docs = results.get('documents', [[]])[0]
            context = "\n\n".join(docs) if docs else ""
            
            data["context"] = context
            data["retrieved_count"] = len(docs)
            
            if docs:
                logger.info(f"📚 Retrieved {len(docs)} chunks")
            else:
                logger.info("⚠️ No relevant docs found")
            
            output_queue.put(data)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Retrieval Error: {e}")
            # Pass through even if retrieval fails
            data["context"] = ""
            output_queue.put(data)











"""
Thread 4: Retrieval (Vector + BM25)
Hybrid retrieval using Ollama embeddings (no Hugging Face)

import logging
import numpy as np
from queue import Queue, Empty
from threading import Event
from typing import List, Dict, Tuple

import chromadb
from rank_bm25 import BM25Okapi
from ollama import Client

from config_threaded import THREADS, RETRIEVAL, OLLAMA


logger = logging.getLogger(__name__)


class RetrievalThread:


    def __init__(self, input_queue: Queue, output_queue: Queue, shutdown_event: Event):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event

        # Configuration
        self.config = RETRIEVAL
        # Use previous batch_size as "top_k_initial"
        self.top_k_initial = self.config["reranking"]["batch_size"]
        self.top_k_final = THREADS["retrieval"]["top_k_final"]

        # Components
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.bm25_corpus: List[str] = []

        # Ollama embeddings
        self.ollama_client = None
        self.embedding_model = self.config.get("embedding_model", "nomic-embed-text")

        logger.info("RetrievalThread initialized (Ollama embeddings, no HF)")

    # --- Initialization helpers ---

    def _init_ollama(self):
        try:
            self.ollama_client = Client(host=OLLAMA["base_url"])
            logger.info(f"Ollama client initialized for embeddings using model {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client for embeddings: {e}")
            raise

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
       
        if self.ollama_client is None:
            self._init_ollama()

        embeddings: List[List[float]] = []
        for t in texts:
            try:
                resp = self.ollama_client.embeddings(model=self.embedding_model, prompt=t)
                emb = resp.get("embedding")
                if emb is None:
                    raise RuntimeError("No embedding field in Ollama response")
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Embedding generation failed for text: {t[:80]!r}: {e}")
                # Fallback to zero-vector of appropriate size if we have at least one
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    raise
        return embeddings

    def load_components(self):
        
        try:
            logger.info("Loading retrieval components (Chroma + BM25, Ollama embeddings)...")

            # 1. Initialize Chroma client / collection
            self.chroma_client = chromadb.PersistentClient(
                path=self.config["vector_db"]["persist_directory"]
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config["vector_db"]["collection_name"],
                metadata={"hnsw:space": "cosine"},
            )

            # 2. Initialize BM25 using documents from collection
            logger.info("Initializing BM25 from Chroma documents...")
            self._initialize_bm25()

            # 3. Initialize Ollama client for embeddings
            self._init_ollama()

            logger.info("✅ Retrieval components loaded (no Hugging Face)")

        except Exception as e:
            logger.error(f"Failed to load retrieval components: {e}")
            raise

    def _initialize_bm25(self):
     
        try:
            # Pull all docs from the collection (may need pagination for very large corpora)
            docs = self.collection.get()

            documents = docs.get("documents", [])
            if documents:
                self.bm25_corpus = documents

                # Tokenize corpus
                tokenized_corpus = [doc.lower().split() for doc in self.bm25_corpus]

                # Create BM25 index
                self.bm25 = BM25Okapi(
                    tokenized_corpus,
                    k1=self.config["bm25"]["k1"],
                    b=self.config["bm25"]["b"],
                )

                logger.info(f"BM25 initialized with {len(self.bm25_corpus)} documents")
            else:
                logger.warning("No documents found in Chroma collection for BM25 initialization")

        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}")

    # --- Retrieval primitives ---

    def vector_search(
        self, query: str, source_category: str, k: int = 20
    ) -> List[Tuple[str, float, Dict]]:
        
        if self.collection is None:
            logger.warning("Chroma collection not initialized")
            return []

        try:
            # Apply source filtering
            where_filter = None
            if self.config["source_filtering"]["enabled"] and source_category != "general":
                where_filter = {"source_category": source_category}

            query_emb = self._embed_texts([query])[0]

            res = self.collection.query(
                query_embeddings=[query_emb],
                n_results=k,
                where=where_filter,
            )

            docs = res.get("documents", [[]])[0]
            scores = res.get("distances", [[]])[0]
            metas = res.get("metadatas", [[]])[0]

            # Convert distances to similarity scores (1 / (1 + distance))
            formatted: List[Tuple[str, float, Dict]] = []
            for content, dist, meta in zip(docs, scores, metas):
                score = 1.0 / (1.0 + float(dist)) if dist is not None else 0.0
                formatted.append((content, score, meta or {}))

            logger.info(f"Vector search returned {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def bm25_search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        
        if not self.bm25 or not self.bm25_corpus:
            logger.warning("BM25 not initialized")
            return []

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = np.argsort(scores)[-k:][::-1]

            # Format results
            results: List[Tuple[str, float]] = [
                (self.bm25_corpus[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]

            logger.info(f"BM25 search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float, Dict]],
        bm25_results: List[Tuple[str, float]],
        k: int = 60,
    ) -> List[Tuple[str, float, Dict]]:
        
        # Create maps for scoring
        doc_scores: Dict[str, float] = {}
        doc_metadata: Dict[str, Dict] = {}

        # Add vector results
        for rank, (content, _score, metadata) in enumerate(vector_results):
            rrf_score = 1 / (k + rank + 1)
            doc_scores[content] = doc_scores.get(content, 0.0) + rrf_score * self.config["fusion"]["vector_weight"]
            doc_metadata[content] = metadata

        # Add BM25 results
        for rank, (content, _score) in enumerate(bm25_results):
            rrf_score = 1 / (k + rank + 1)
            doc_scores[content] = doc_scores.get(content, 0.0) + rrf_score * self.config["fusion"]["bm25_weight"]
            if content not in doc_metadata:
                doc_metadata[content] = {}

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Format results
        fused_results: List[Tuple[str, float, Dict]] = [
            (content, float(score), doc_metadata.get(content, {}))
            for content, score in sorted_docs
        ]

        logger.info(f"Fusion produced {len(fused_results)} results")
        return fused_results

    # --- End-to-end pipeline ---

    def retrieve(self, processed_query: dict) -> dict:
        
        query = processed_query["optimized_query"]
        source_category = processed_query["source_category"]

        logger.info(f"Retrieving for query: {query}")
        logger.info(f"Source category: {source_category}")

        # Step 1 & 2: Parallel search
        vector_results = self.vector_search(query, source_category, k=self.top_k_initial)
        bm25_results = self.bm25_search(query, k=self.top_k_initial)

        # Step 3: Fusion
        fused_results = self.reciprocal_rank_fusion(vector_results, bm25_results)

        # Step 4: Return top-K final results (no HF reranker)
        final_results = fused_results[:self.top_k_final]

        logger.info(f"Retrieved {len(final_results)} final results")

        return {
            "query": processed_query,
            "results": [
                {
                    "content": content,
                    "score": float(score),
                    "metadata": metadata,
                }
                for content, score, metadata in final_results
            ],
        }

    def run(self):
       
        logger.info("Retrieval thread started")

        try:
            # Load components
            self.load_components()

            # Process queries
            while not self.shutdown_event.is_set():
                try:
                    # Get processed query
                    processed_query = self.input_queue.get(timeout=0.1)

                    # Perform retrieval
                    retrieval_results = self.retrieve(processed_query)

                    # Push to LLM queue
                    self.output_queue.put(retrieval_results, block=False)
                    logger.info("Retrieval results sent to LLM")

                except Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error during retrieval: {e}", exc_info=True)

            logger.info("Retrieval thread stopping...")

        except Exception as e:
            logger.error(f"Retrieval thread error: {e}", exc_info=True)

        finally:
            self.cleanup()

    def cleanup(self):
     
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.ollama_client = None

        logger.info("Retrieval thread cleaned up")


def retrieval_thread_main(manager):
   
    thread = RetrievalThread(
        input_queue=manager.get_queue("query_processed"),
        output_queue=manager.get_queue("retrieval_results"),
        shutdown_event=manager.shutdown_event,
    )
    thread.run()

"""