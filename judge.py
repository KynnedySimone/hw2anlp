"""LLM-as-judge (1-5) for a scored run. Reads outputs/<run>.details.jsonl and adds a
"judge" score to outputs/<run>.scores.json.

    python judge.py outputs/dev_hybrid_rrf

The judge is an open model (closed models aren't allowed). By default it is the same
Qwen2.5-7B-Instruct used as the reader, so its scores may be biased toward its own
answers; pass --model to use a different open model.
"""
import argparse
import json
import re

import make_results_table
from rag.reader import Reader

RUBRIC = """You grade answers to factual questions about Pittsburgh and CMU.
Compare the system answer to the reference answer(s) and give a score from 1 to 5:
5 = fully correct, same meaning as a reference (wording may differ)
4 = correct but has minor extra/missing detail
3 = partially correct
2 = mostly wrong but related to the right answer
1 = wrong, irrelevant, or no answer
Reply with only the number."""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="output path without extension, e.g. outputs/dev_hybrid_rrf")
    p.add_argument("--model", default=None)
    p.add_argument("--backend", default=None, choices=["mlx", "hf"])
    args = p.parse_args()
    run = args.run.removesuffix(".json")

    with open(run + ".details.jsonl") as f:
        entries = [json.loads(line) for line in f]
    judge = Reader(args.model, args.backend, max_new_tokens=4)
    scores = []
    for e in entries:
        msgs = [{"role": "system", "content": RUBRIC},
                {"role": "user", "content": f"Question: {e['question']}\nReference answer(s): "
                 f"{' | '.join(e['answers'])}\nSystem answer: {e['prediction']}\nScore:"}]
        m = re.search(r"[1-5]", judge.generate(msgs))
        e["judge"] = int(m.group()) if m else 1
        scores.append(e["judge"])

    with open(run + ".details.jsonl", "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(run + ".scores.json") as f:
        record = json.load(f)
    record["scores"]["judge"] = sum(scores) / len(scores)
    record["judge_model"] = args.model or "reader default (Qwen2.5-7B-Instruct)"
    with open(run + ".scores.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"{run}: judge = {record['scores']['judge']:.2f} (n={len(scores)})")
    make_results_table.main()  # keep RESULTS.md up to date


if __name__ == "__main__":
    main()
