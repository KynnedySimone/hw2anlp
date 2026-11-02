"""Fast retrieval-only evaluation (no LLM): for each question, does any reference answer
string appear in the top-k retrieved chunks? Use it to compare chunking, retrievers and
fusion strategies before paying for full end-to-end runs.

    # one config
    python eval_retrieval.py --retriever hybrid --fusion rrf
    # many configs in one process (models, chunks and indexes are loaded once and shared)
    python eval_retrieval.py --config "--chunking fixed --retriever sparse" --config "--chunking fixed --retriever dense"
    # ...and skip configs already evaluated on the current corpus
    python eval_retrieval.py --skip-done --config "..." --config "..."
"""
import argparse
import json
import os
import shlex
import sys
import time

import make_results_table
from rag.metrics import answer_in_context, normalize
from rag.pipeline import DEFAULT_RERANKER, add_common_args, build_retriever, corpus_version, load_chunks

RESULTS_LOG = "outputs/retrieval_results.jsonl"


def config_parser():
    p = argparse.ArgumentParser(add_help=False)
    add_common_args(p)
    return p


def config_key(r):
    """The settings that identify a retrieval run (same flags spelled differently = same run)."""
    hybrid = r["retriever"] == "hybrid"
    return (r["chunking"], int(r["chunk_size"]), r["retriever"],
            r["fusion"] if hybrid else None, float(r["alpha"]) if hybrid else None,
            r["embedder"] if r["retriever"] != "sparse" else None, bool(r["rerank"]),
            (r.get("reranker") or DEFAULT_RERANKER) if r["rerank"] else None)


def done_keys(version):
    """Configs already evaluated on this corpus version."""
    if not os.path.exists(RESULTS_LOG):
        return set()
    with open(RESULTS_LOG) as f:
        return {config_key(r) for r in map(json.loads, f) if r.get("corpus") == version}


def evaluate(args, data, ks, version, cache):
    ckey = ("chunks", args.chunking, args.chunk_size, args.overlap)
    if ckey not in cache:
        chunks = load_chunks(args.chunking, args.chunk_size, args.overlap)
        # many answers are missing from the corpus entirely, which caps every retriever equally;
        # also report hit@k over the "answerable" questions whose answer string is somewhere in the corpus
        corpus = normalize(" ".join(c["text"] for c in chunks))
        answerable = [any(normalize(a) and normalize(a) in corpus for a in ex["answers"]) for ex in data]
        cache[ckey] = (chunks, answerable)
    chunks, answerable = cache[ckey]
    retriever = build_retriever(chunks, args.chunking, args.chunk_size, args.retriever, args.fusion,
                                args.alpha, args.embedder, args.rerank, cache=cache, reranker=args.reranker)
    hits = {k: 0.0 for k in ks}
    for ex in data:
        results = retriever.search(ex["question"], max(ks))
        for k in ks:
            hits[k] += answer_in_context(ex["answers"], [chunks[i]["text"] for i, _ in results[:k]])
    n, n_ans = len(data), int(sum(answerable))
    name = f"{args.chunking}{args.chunk_size} {args.retriever}" + \
        (f"/{args.fusion}(a={args.alpha})" if args.retriever == "hybrid" else "") + (" +rerank" if args.rerank else "") + \
        (f"({args.reranker.split('/')[-1]})" if args.rerank and args.reranker != DEFAULT_RERANKER else "")
    print(f"{name}: {len(chunks)} chunks | all n={n}: " + " ".join(f"hit@{k}={hits[k] / n:.3f}" for k in ks)
          + f" | answerable n={n_ans}: " + " ".join(f"hit@{k}={hits[k] / n_ans:.3f}" for k in ks), flush=True)
    return {
        "name": name, "chunking": args.chunking, "chunk_size": args.chunk_size,
        "retriever": args.retriever, "fusion": args.fusion if args.retriever == "hybrid" else None,
        "alpha": args.alpha if args.retriever == "hybrid" else None,
        "embedder": args.embedder if args.retriever != "sparse" else None, "rerank": args.rerank,
        "reranker": args.reranker if args.rerank else None,
        "n_chunks": len(chunks), "n": n, "n_answerable": n_ans,
        "hit_all": {k: hits[k] / n for k in ks}, "hit_answerable": {k: hits[k] / n_ans for k in ks},
        "corpus": version, "time": time.strftime("%Y-%m-%d %H:%M"),
    }


def main():
    p = argparse.ArgumentParser(parents=[config_parser()])
    p.add_argument("--config", action="append", default=[],
                   help="a quoted set of retrieval flags; repeat to evaluate several configs in one process")
    p.add_argument("--skip-done", action="store_true", help="skip configs already evaluated on the current corpus")
    p.add_argument("--data", default="public_evaluation_data.json")
    p.add_argument("--ks", default="1,3,5,10")
    args, rest = p.parse_known_args()
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")

    if args.config:
        configs = [(shlex.split(c), config_parser().parse_args(shlex.split(c))) for c in args.config]
    else:  # single config given directly on the command line
        configs = [(sys.argv[1:], args)]

    version = corpus_version()
    print(f"corpus version {version}", flush=True)
    if args.skip_done:
        done = done_keys(version)
        for flags, c in configs:
            if config_key(vars(c)) in done:
                print("already done on this corpus:", shlex.join(flags))
        configs = [(f, c) for f, c in configs if config_key(vars(c)) not in done]

    unique = {}  # the same config spelled two ways only runs once
    for f, c in configs:
        unique.setdefault(config_key(vars(c)), (f, c))
    configs = list(unique.values())

    ks = [int(k) for k in args.ks.split(",")]
    data = json.load(open(args.data))
    cache = {}
    os.makedirs("outputs", exist_ok=True)
    # group configs by chunking so each chunk set / index is loaded once
    configs.sort(key=lambda fc: (fc[1].chunking, fc[1].chunk_size, fc[1].embedder))
    for flags, c in configs:
        record = evaluate(c, data, ks, version, cache)
        record["command"] = "python eval_retrieval.py " + shlex.join(flags)
        # log every run; make_results_table.py turns this into the retrieval table in RESULTS.md
        with open(RESULTS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        make_results_table.main()  # keep RESULTS.md up to date


if __name__ == "__main__":
    main()
