import re
import string
from collections import Counter

# SQuAD-style answer normalisation and token metrics (Rajpurkar et al., 2016, sec 6.1).
# Each metric takes the max over all reference answers.


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(pred_toks, ref_toks):
    common = Counter(pred_toks) & Counter(ref_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    p, r = same / len(pred_toks), same / len(ref_toks)
    return 2 * p * r / (p + r)


def _recall(pred_toks, ref_toks):
    if not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    return sum(common.values()) / len(ref_toks)


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def _rouge_l(pred_toks, ref_toks):
    lcs = _lcs(pred_toks, ref_toks)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(pred_toks), lcs / len(ref_toks)
    return 2 * p * r / (p + r)


def score(pred, refs):
    p = normalize(pred).split()
    out = {"em": 0.0, "f1": 0.0, "recall": 0.0, "rougeL": 0.0}
    for ref in refs:
        r = normalize(ref).split()
        out["em"] = max(out["em"], float(p == r))
        if p and r:
            out["f1"] = max(out["f1"], _f1(p, r))
            out["recall"] = max(out["recall"], _recall(p, r))
            out["rougeL"] = max(out["rougeL"], _rouge_l(p, r))
    return out


def answer_in_context(refs, texts):
    """Retrieval metric: does any reference answer appear (normalised) in the retrieved text?"""
    ctx = normalize(" ".join(texts))
    return float(any(normalize(r) and normalize(r) in ctx for r in refs))
