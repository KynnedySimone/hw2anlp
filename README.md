# HW2 - Pittsburgh Trivia Rag System

A RAG system that coveres local arts, culture, food, city info, sports, etc, in the Pittsburgh area. The answers are sourced from 2,257 documents (see data/scraped_data.json, and the scraper code lives in scraper.py). The system uses sentence-level chunking, weighted hybrid fusion, Qwen2.5-14B-Instruct as the reader, and an adaptive prompting outline.

**(I submitted the wrong README on gradescope, this is the correct one!!)**

## Results

Scores for `public_evaluation_data.json`, where the top 5 chunks for an answer were retrieved:

| System | Recall | F1 | ROUGE-L | EM |
|---|---|---|---|---|
| Closed-book (no retrieval) | 9.7 | 9.9 | 9.9 | 2.5 |
| Sparse only (BM25) | 55.6 | 55.4 | 55.3 | 43.9 |
| Dense only (bge-m3) | 57.4 | 57.6 | 57.5 | 47.1 |
| Hybrid weighted (7B reader) | 59.5 | 59.7 | 59.7 | 47.8 |
| **v3 (this repo)** | **68.8** | **66.7** | **65.9** | **52.2** |

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

# run on the test set
python run_rag.py --questions test_set_day_5.json --output system_outputs/system_output_1.json \
  --chunking sentence --chunk-size 200 --top-k 5 \
  --retriever hybrid --fusion weighted --alpha 0.5 \
  --answer-style v3 --reader-model mlx-community/Qwen2.5-14B-Instruct-4bit
```
If you're not running on a mac, use this:

```bash
python run_rag.py ... --answer-style v3 --backend hf --reader-model Qwen/Qwen2.5-14B-Instruct
```



