import hashlib
import json
import os

from .chunking import chunk_document
from .retrieval import DenseRetriever, HybridRetriever, Reranker, SparseRetriever

INDEX_DIR = "indexes"
CORPUS = os.path.join("data", "scraped_data.json")
DEFAULT_EMBEDDER = "BAAI/bge-m3"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def corpus_version(corpus=CORPUS):
    """Short hash of the corpus contents. Every result is tagged with it, so results from
    different scrapes are never mixed in RESULTS.md."""
    h = hashlib.sha1()
    with open(corpus, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:10]


def index_dir(strategy, size):
    return os.path.join(INDEX_DIR, f"{strategy}_{size}")


def load_chunks(strategy="section", size=200, overlap=40, corpus=CORPUS, rebuild=False):
    """Chunk the corpus once and cache the chunks as jsonl."""
    d = index_dir(strategy, size)
    path = os.path.join(d, "chunks.jsonl")
    meta_path = os.path.join(d, "meta.json")
    # re-chunk whenever the corpus file or chunking params change (e.g. after re-scraping)
    stat = os.stat(corpus)
    meta = {"corpus": corpus, "corpus_size": stat.st_size, "corpus_mtime": stat.st_mtime, "overlap": overlap}
    if os.path.exists(path) and os.path.exists(meta_path) and not rebuild:
        with open(meta_path) as f:
            if json.load(f) == meta:
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f]
    with open(corpus, encoding="utf-8") as f:
        docs = json.load(f)
    chunks = []
    for doc in docs.values():
        chunks.extend(chunk_document(doc, strategy, size, overlap))
    os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    # chunks changed, so any cached dense index is stale
    for name in os.listdir(d):
        if name.endswith(".faiss"):
            os.remove(os.path.join(d, name))
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return chunks


def build_retriever(chunks, strategy, size, retriever="hybrid", fusion="rrf", alpha=0.5,
                    embedder=DEFAULT_EMBEDDER, rerank=False, candidates=50, cache=None,
                    reranker=DEFAULT_RERANKER):
    """`cache` (a dict) lets one process evaluate many configs without reloading models or
    rebuilding the BM25/FAISS indexes for every config."""
    cache = {} if cache is None else cache

    def cached(key, make):
        if key not in cache:
            cache[key] = make()
        return cache[key]

    sparse = dense = None
    if retriever in ("sparse", "hybrid"):
        sparse = cached(("bm25", strategy, size), lambda: SparseRetriever(chunks))
    if retriever in ("dense", "hybrid"):
        faiss_path = os.path.join(index_dir(strategy, size), embedder.replace("/", "__") + ".faiss")
        dense = cached(("dense", strategy, size, embedder), lambda: DenseRetriever(
            chunks, embedder, faiss_path, model=cache.get(("embedder", embedder))))
        cache[("embedder", embedder)] = dense.model
    if retriever == "sparse":
        base = sparse
    elif retriever == "dense":
        base = dense
    elif retriever == "hybrid":
        base = HybridRetriever(sparse, dense, fusion=fusion, alpha=alpha, candidates=candidates)
    else:
        raise ValueError(f"unknown retriever {retriever}")
    if not rerank:
        return base
    model = Reranker(base, chunks, reranker, model=cache.get(("cross-encoder", reranker)))
    cache[("cross-encoder", reranker)] = model.model
    return model


def add_common_args(p):
    p.add_argument("--chunking", default="section", choices=["fixed", "sentence", "section"])
    p.add_argument("--chunk-size", type=int, default=200, help="max words per chunk")
    p.add_argument("--overlap", type=int, default=40, help="word overlap (fixed chunking only)")
    p.add_argument("--retriever", default="hybrid", choices=["sparse", "dense", "hybrid"])
    p.add_argument("--fusion", default="rrf", choices=["rrf", "weighted"])
    p.add_argument("--alpha", type=float, default=0.5, help="hybrid weight on dense (1-alpha on BM25)")
    p.add_argument("--embedder", default=DEFAULT_EMBEDDER)
    p.add_argument("--rerank", action="store_true", help="re-rank with a cross-encoder")
    p.add_argument("--reranker", default=DEFAULT_RERANKER,
                   help="cross-encoder used by --rerank (v2: BAAI/bge-reranker-v2-m3)")
    p.add_argument("--top-k", type=int, default=5, help="chunks given to the reader")
