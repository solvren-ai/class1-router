---
license: other
license_name: mit-0
license_link: https://github.com/aws/mit-0
language:
  - en
task_categories:
  - text-classification
tags:
  - synthetic
  - request-routing
  - llm-routing
pretty_name: Class 1 Decider router (synthetic)
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: valid.jsonl
      - split: test
        path: test.jsonl
      - split: handwritten_test
        path: handwritten_test.jsonl
---

# Class 1 Decider router dataset (synthetic)

This dataset holds request-routing examples for Solvren AI's "Class 1 Decider", a lightweight router
that sits in front of LLM calls and decides which path handles each request. Each row is one user
request labelled with one of four routes:

| Route | Meaning |
| --- | --- |
| `small-model` | simple, well-scoped, low-risk text work (reformat, short summary, extract, translate, classify) |
| `frontier-model` | complex multi-step reasoning, long-form drafting, nuanced analysis |
| `retrieval` | needs specific internal documents, records, or current external sources |
| `refuse-or-escalate` | high-stakes individualized advice, another person's private records, or anything that needs a qualified human |

## Splits

| Split | Rows | How it was made |
| --- | --- | --- |
| `train` | 1,024 | templates |
| `validation` | 256 | templates, unseen core requests |
| `test` | 256 | templates, unseen core requests |
| `handwritten_test` | 50 | written by hand, different style |

The three template splits are balanced: 25% of rows per route. `handwritten_test` has 12 or 13
rows per route.

## How it was built

- **Template splits:** generated offline by
  [`local/gen_router_data.py`](https://github.com/solvren-ai/class1-router/blob/cb839a10a578adf6ee5a9f81403229dbe508de02/local/gen_router_data.py),
  with no model calls. The grid is 4 domains (healthcare, legal, fintech, biotech) x 4 routes x
  12 core requests, and each core request gets 8 phrasing variants (a greeting, an urgency marker, a
  team prefix and so on) plus fictional slot values (drug name, contract type, assay, record id).
- **Leakage control:** splitting is by **core request**. A request and all of its phrasings land in
  exactly one split, with 8 core requests per domain and route in train, 2 in validation and 2 in
  test, so the test requests never appear in training in any phrasing.
- **`handwritten_test`:** 50 requests written by hand to check generalisation beyond the template
  family. They have different phrasing, varied length, typos and lowercase, and a mix of domains
  that includes general business requests outside the four training domains. None of them appears
  in train.

## Format

Each row uses the platform's canonical chat format:

```json
{"messages": [
  {"role": "system", "content": "<routing policy prompt>"},
  {"role": "user", "content": "Extract the follow-up date from this lab report."},
  {"role": "assistant", "content": "small-model"}],
 "domain": "healthcare"}
```

Every row carries the same system prompt, which defines the four routes. Rows in
`handwritten_test` also carry `"source": "handwritten"`.

## Privacy and safety

- **No real PII or PHI.** Every name, record id, company, lot number and amount is a fictional
  placeholder. The company names (Acme, Northwind, Contoso, Globex, Initech) are the usual sample
  names.
- **`refuse-or-escalate` examples are benign escalation triggers:** urgent symptoms, a request for an
  individualized medical, legal or financial decision, a request for another person's records, a
  skipped human approval, or someone who needs a person rather than a model. The dataset contains no
  harmful instructions.

## Limitations

- Template splits are easy. A model can score near 100% on `test` by learning the template family,
  so read `handwritten_test` for a more honest signal.
- English only. The labels reflect one opinionated routing policy, and some requests sit near a
  boundary (for example, a short drafting task could reasonably be `small-model` or
  `frontier-model`).
- Not a safety benchmark. Do not use it to evaluate content-safety classifiers.

## License

MIT-0, the same as the source repository.
