# HW2 - Pittsburgh Trivia Rag System

A RAG system that coveres local arts, culture, food, city info, sports, etc, in the Pittsburgh area. The answers are sourced from 2,370 documents (see data/scraped_data.json, and the scraper code lives in scraper.py). The system uses sentence-level chunking, weighted hybrid fusion, Qwen2.5-14B-Instruct as the reader, and an adaptive prompting outline.

**(I submitted the wrong README on gradescope, this is the correct one!!)**

## Results

Scores for `public_evaluation_data.json`, where the top 5 chunks for an answer were retrieved:

| System | Recall | F1 | ROUGE-L | EM |
|---|---|---|---|---|
| Closed-book (no retrieval) | 9.7 | 9.9 | 9.9 | 2.5 |
| Sparse only (BM25) | 56.6 | 56.1 | 56.0 | 43.9 |
| Dense only (bge-m3) | 58.5 | 58.7 | 58.6 | 48.4 |
| Hybrid weighted (7B reader) | 59.9 | 59.8 | 59.8 | 47.8 |
| **v3 weighted (this repo)** | **69.8** | **67.6** | **66.8** | **52.2** |
| v4 | 69.6 | 67.4 | 66.8 | 52.2 |

## System outputs

| File | System |
|---|---|
| `system_output_1.json` | v3 with re-ranking (hybrid RRF + cross-encoder) |
| `system_output_2.json` | v3 weighted |
| `system_output_3.json` | v4 |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run this from the repo root:

```bash
# run on public_eval_data
python run_rag.py --questions public_evaluation_data.json --output outputs/dev_v3.json \
  --chunking sentence --chunk-size 200 --top-k 5 \
  --retriever hybrid --fusion weighted --alpha 0.5 \
  --answer-style v3 --reader-model mlx-community/Qwen2.5-14B-Instruct-4bit

# run on the test set: v3 with re-ranking
python run_rag.py --questions test_set_day_5.json --output system_outputs/system_output_1.json \
  --chunking sentence --chunk-size 200 --top-k 5 \
  --retriever hybrid --fusion rrf --alpha 0.5 --rerank \
  --answer-style v3 --reader-model mlx-community/Qwen2.5-14B-Instruct-4bit

# run on the test set: v3 weighted
python run_rag.py --questions test_set_day_5.json --output system_outputs/system_output_2.json \
  --chunking sentence --chunk-size 200 --top-k 5 \
  --retriever hybrid --fusion weighted --alpha 0.5 \
  --answer-style v3 --reader-model mlx-community/Qwen2.5-14B-Instruct-4bit

# run on the test set: v4
python run_rag.py --questions test_set_day_5.json --output system_outputs/system_output_3.json \
  --chunking sentence --chunk-size 200 --top-k 5 \
  --retriever hybrid --fusion weighted --alpha 0.5 \
  --answer-style v4 --reader-model mlx-community/Qwen2.5-14B-Instruct-4bit
```
If you're not running on a mac, use this:

```bash
python run_rag.py ... --answer-style v3 --backend hf --reader-model Qwen/Qwen2.5-14B-Instruct
```
