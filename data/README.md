# Data Directory

## Structure

```
data/
├── raw/
│   └── sample_twcs.csv          # 50-tweet sample from Kaggle Customer Support on Twitter
├── processed/
│   ├── conversations.json       # Reconstructed multi-turn conversations (AmazonHelp)
│   ├── dev_conversations.json   # 70% Development/Retrieval split (8 conversations)
│   ├── val_conversations.json   # 15% Validation split (2 conversations)
│   └── eval_conversations.json  # 15% Evaluation split (2 conversations)
├── golden/
│   └── golden_set.jsonl         # 200 human-labeled golden evaluation records
├── cache/
│   └── *.json                   # LLM response disk cache (auto-generated)
└── README.md                    # This file
```

## Dataset Source

**Kaggle Customer Support on Twitter** ([link](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter))
- ~3 million tweets across 108 brands
- Selected brand: **AmazonHelp** (150K+ tweets, highest density of multi-turn resolution threads)
- `sample_twcs.csv` contains a 50-tweet subset for offline demo/test purposes

## Leak-Free Splitting

All splits are performed at the **conversation ID level** to prevent data leakage:
- No customer message or agent reply appears in both the retrieval index and the evaluation set
- Verified by `scripts/check_leakage.py` and `tests/test_leakage.py`

## Golden Set

The golden set (`golden_set.jsonl`) contains 200 records with:
- `message`: Customer message text
- `gold_intent`: Human-labeled intent from 12-intent taxonomy
- `gold_escalation`: Human-labeled AUTO_HANDLE/ESCALATE decision
- `gold_reason`: Diagnostic rationale for the label
- `reference_resolution`: Expected agent resolution approach
- `difficulty`: easy / medium / hard
- `annotator_notes`: Stratified sampling notes

## How to Obtain Full Dataset

1. Download `twcs.csv` from [Kaggle](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
2. Place it in `data/raw/twcs.csv`
3. Run `python scripts/preprocess.py` to rebuild all processed files
