import re

import faiss
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

# Sparse (BM25), dense (sentence-transformers + FAISS) and hybrid retrieval.
# Every retriever returns a list of (chunk_index, score), best first.

STOPWORDS = set("""a an and are as at be by did do does for from has have how i in is it its of on or
that the their this to was were what when where which who whom whose why will with""".split())

# some embedding models are trained with instruction prefixes
QUERY_PREFIX = {
    "intfloat/e5-base-v2": "query: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}
DOC_PREFIX = {
    "intfloat/e5-base-v2": "passage: ",
}

_stemmer = PorterStemmer()


def tokenize(text):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return [_stemmer.stem(t) for t in toks if t not in STOPWORDS]


class SparseRetriever:
    def __init__(self, chunks):
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in top]


class DenseRetriever:
    def __init__(self, chunks, model_name, index_path=None, device=None, batch_size=32, model=None):
        """Pass an already-loaded `model` to share one embedder across several indexes."""
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = model or SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)
        if index_path and _exists(index_path):
            self.index = faiss.read_index(index_path)
            assert self.index.ntotal == len(chunks), "stale FAISS index, rebuild it"
        else:
            emb = self.encode([DOC_PREFIX.get(model_name, "") + c["text"] for c in chunks], batch_size)
            # vectors are L2-normalised, so inner product == cosine similarity
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)
            if index_path:
                faiss.write_index(self.index, index_path)

    def encode(self, texts, batch_size=32):
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 100, convert_to_numpy=True)
        return np.ascontiguousarray(emb, dtype=np.float32)

    def search(self, query, k):
        q = self.encode([QUERY_PREFIX.get(self.model_name, "") + query])
        scores, ids = self.index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def _exists(path):
    import os
    return os.path.exists(path)


def rrf_fusion(result_lists, k=60, weights=None):
    """Reciprocal Rank Fusion (Cormack et al., 2009): score = sum_r w_r / (k + rank_r)."""
    weights = weights or [1.0] * len(result_lists)
    fused = {}
    for w, results in zip(weights, result_lists):
        for rank, (idx, _) in enumerate(results, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def weighted_fusion(result_lists, weights):
    """Min-max normalise each retriever's scores over its candidates, then take a
    weighted sum. A document missing from a list gets 0 from that list."""
    fused = {}
    for w, results in zip(weights, result_lists):
        if not results:
            continue
        scores = np.array([s for _, s in results])
        lo, hi = scores.min(), scores.max()
        for idx, s in results:
            norm = (s - lo) / (hi - lo) if hi > lo else 1.0
            fused[idx] = fused.get(idx, 0.0) + w * norm
    return sorted(fused.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, sparse, dense, fusion="rrf", alpha=0.5, candidates=50, rrf_k=60):
        """alpha = weight on dense, (1 - alpha) on sparse."""
        self.sparse, self.dense = sparse, dense
        self.fusion, self.alpha, self.candidates, self.rrf_k = fusion, alpha, candidates, rrf_k

    def search(self, query, k):
        lists = [self.dense.search(query, self.candidates), self.sparse.search(query, self.candidates)]
        weights = [self.alpha, 1 - self.alpha]
        if self.fusion == "rrf":
            fused = rrf_fusion(lists, self.rrf_k, [2 * w for w in weights])
        elif self.fusion == "weighted":
            fused = weighted_fusion(lists, weights)
        else:
            raise ValueError(f"unknown fusion {self.fusion}")
        return fused[:k]


class Reranker:
    """Re-scores the first-stage candidates with a cross-encoder that reads query and chunk together."""

    def __init__(self, base, chunks, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", candidates=30, model=None):
        from sentence_transformers import CrossEncoder
        self.base, self.chunks, self.candidates = base, chunks, candidates
        self.model = model or CrossEncoder(model_name, max_length=512)

    def search(self, query, k):
        cands = self.base.search(query, self.candidates)
        if not cands:
            return []
        scores = self.model.predict([(query, self.chunks[i]["text"]) for i, _ in cands], show_progress_bar=False)
        order = np.argsort(-scores)[:k]
        return [(cands[j][0], float(scores[j])) for j in order]
