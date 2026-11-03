"""Answer a question file with the RAG system (or closed-book) and optionally score it.

    # dev set, scored locally
    python run_rag.py --questions public_evaluation_data.json --output outputs/dev_hybrid.json
    # closed-book baseline
    python run_rag.py --questions public_evaluation_data.json --closed-book --output outputs/dev_closed.json
    # leaderboard submission
    python run_rag.py --questions leaderboard_queries.json --output outputs/leaderboard.json --andrewid ANDREWID
    # test set
    python run_rag.py --questions test_set_day_3.json --output system_outputs/system_output_1.json
"""
import argparse
import json
import os
import shlex
import sys
import time

import make_results_table
from rag.metrics import answer_in_context, score
from rag.pipeline import add_common_args, build_retriever, corpus_version, load_chunks
from rag.reader import ANSWER_STYLES, Reader


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--questions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--closed-book", action="store_true", help="no retrieval, LLM only")
    p.add_argument("--reader-model", default=None, help="v2_full: mlx-community/Qwen2.5-14B-Instruct-4bit")
    p.add_argument("--answer-style", default="short", choices=ANSWER_STYLES,
                   help="short = always a short phrase (v1); adaptive = full sentence for descriptive questions (v2)")
    p.add_argument("--backend", default=None, choices=["mlx", "hf"])
    p.add_argument("--andrewid", default=None, help="adds the andrewid field for leaderboard submissions")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    data = json.load(open(args.questions))[:args.limit]
    if not args.closed_book:
        chunks = load_chunks(args.chunking, args.chunk_size, args.overlap)
        retriever = build_retriever(chunks, args.chunking, args.chunk_size, args.retriever, args.fusion,
                                    args.alpha, args.embedder, args.rerank, reranker=args.reranker)
    reader = Reader(args.reader_model, args.backend, answer_style=args.answer_style)

    answers, log = {}, []
    start = time.time()
    for n, ex in enumerate(data, 1):
        contexts, sources = None, []
        if not args.closed_book:
            results = retriever.search(ex["question"], args.top_k)
            contexts = [chunks[i]["text"] for i, _ in results]
            sources = [chunks[i]["id"] for i, _ in results]
        pred = reader.answer(ex["question"], contexts)
        answers[ex["id"]] = pred
        entry = {"id": ex["id"], "question": ex["question"], "prediction": pred, "sources": sources,
                 "sentence_mode": reader.sentence_mode(ex["question"], contexts),
                 "retried": getattr(reader, "retried", False), "fallback": getattr(reader, "fallback", False)}
        if "answers" in ex:
            entry["answers"] = ex["answers"]
            entry.update(score(pred, ex["answers"]))
            if contexts is not None:
                entry["answer_in_context"] = answer_in_context(ex["answers"], contexts)
        log.append(entry)
        if n % 10 == 0:
            print(f"{n}/{len(data)} ({(time.time() - start) / n:.1f}s/q)", flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = ({"andrewid": args.andrewid} if args.andrewid else {}) | answers
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    # per-question details (predictions, retrieved sources, scores) for error analysis
    with open(os.path.splitext(args.output)[0] + ".details.jsonl", "w", encoding="utf-8") as f:
        for e in log:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    scored = [e for e in log if "f1" in e]
    if scored:
        n = len(scored)
        summary = {m: sum(e[m] for e in scored) / n for m in ("em", "f1", "recall", "rougeL")}
        if "answer_in_context" in scored[0]:
            summary["answer_in_context"] = sum(e["answer_in_context"] for e in scored) / n
        summary["n"] = n
        # everything needed to reproduce this variant; make_results_table.py reads these files
        record = {"command": "python " + shlex.join(sys.argv), "config": vars(args), "scores": summary,
                  "corpus": None if args.closed_book else corpus_version(), "time": time.strftime("%Y-%m-%d %H:%M")}
        print(json.dumps(record, indent=2))
        with open(os.path.splitext(args.output)[0] + ".scores.json", "w") as f:
            json.dump(record, f, indent=2)
        make_results_table.main()  # keep RESULTS.md up to date


if __name__ == "__main__":
    main()
