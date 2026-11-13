# Eval results

Model: `wb-qwen:latest` (qwen2.5:7b, num_ctx 16384, temperature 0) via Ollama on an Apple M5.
Each of the 6 TASK.md questions was run 3 times and graded by `scripts/check_citations.py`.

Reproduce:

```bash
python scripts/run_eval.py              # 6 questions x 3 runs -> eval/runs/ (gitignored)
python scripts/check_citations.py       # replay + grade
python scripts/check_citations.py --self-test
```

A run passes only if every cited title, date and link exists in the tool results the model actually
saw (replayed, with the client's own truncation applied) **and** the per-question check holds:

| Q | Per-question check |
|---|---|
| 1 | ≥3 real documents, ≥1 about climate/resilience, all Kenya |
| 2 | ≥3 real documents, all dated 2019–2022, Sub-Saharan Africa was searched |
| 3 | Top 5 document types in the right order; any counts given must be correct |
| 4 | Cites a document carrying the newest Economic & Sector Work date for Indonesia |
| 5 | Lists ≥95% of the returned languages, none invented; any counts given must be correct |
| 6 | ≥3 real documents, all Brazil, ≥half with water/sanitation/sewerage in the title |

## Final: 18/18 (current prompt and server)

| # | Pass | Tools called (most common run) | Varied between runs |
|---|---|---|---|
| 1 Climate resilience, Kenya | 3/3 | `filter_documents(country="Kenya", topic="climate resilience")` → 0 results + hint → `search_documents("climate resilience Kenya", rows=10)` | One of 10 cited documents (API tie order, below). Runs 1 and 2 pass as ties at the page boundary (below); before the checker handled ties, each passed 4 of 6 manual regrades. |
| 2 Education financing, Sub-Saharan Africa, 2019–22 | 3/3 | `filter_documents(region="Sub-Saharan Africa", topic="Education", start_date, end_date, rows=10)` | Nothing (answers identical, 10 of 10 links). |
| 3 Most frequent document types | 3/3 | `get_facets(document_type)` | Nothing. Correct top 10 in order, no counts. |
| 4 Most recent ESW, Indonesia | 3/3 | `filter_documents(country="Indonesia", major_document_type="Economic & Sector Work", newest_first=true, rows=1)` | Nothing |
| 5 All languages | 3/3 | `get_facets(language)` | Nothing; 100/100 values listed, no counts |
| 6 Water and sanitation, Brazil | 3/3 | `filter_documents(country="Brazil", topic="Water Resources", major_document_type="Project Documents", query="sanitation", rows=10)` | Run 1 omitted all PDF links; runs 2 and 3 had 10 |

No run had a malformed tool call, an unknown tool, a nudge, or hit the 8-iteration cap. No result
needed truncation (every call used `rows` ≤ 10).

This set was run after two server output changes: search/filter results call the country field
`country` instead of the API's `count`, and `get_document` keeps the first 20 `keywd` entries.

Two earlier sets of 18 runs with the same prompt also scored 18/18, with the same variations
(Q1 tie order, one Q6 run without links). In those, Q2 cited 7 of 10 links per run.
`examples/q1.txt`–`q6.txt` are run 1 of the final set above. `python scripts/check_citations.py examples`,
run 5 times: Q2–Q6 PASS every time; Q1 "pass (tie at page boundary)" every time (the tie below).

## Experiment: where the `country` field goes

The first version of the rename appended `country` after `pdfurl` instead of renaming it in place.
Q2 then included **0 PDF links in 3/3 runs** (it still passed: titles and dates were right).

| Q2, 3 runs each | Links per run |
|---|---|
| `count`, original position | 10, 10, 10 |
| `country`, moved after `pdfurl` | 0, 0, 0 |
| `country`, original position (kept) | 10, 10, 10 |

Field order, not the name, decided it. A test now pins the order of the summary fields.

## Experiment: "include counts" prompt line (reverted)

Added: *"When a tool returns counts, include them in the answer."* 18 runs, with counts required for Q3
and Q5 in that round's grading:

| | Effect |
|---|---|
| **Fixed** | Q3 gave counts in 3/3 runs, all correct at run time (e.g. "Procurement Plan - 144,387 documents"). |
| **Did not fix** | Q5 still listed all 100 languages with no counts, 0/3. |
| **Broke** | Q6 included **0 PDF links in 3/3 runs**. Rerunning Q6 with the line removed: links per run [10, 10, 0]; with it: [0, 0, 0]. |

**Why it was reverted:** citing real documents with links is the point of the system. Counts on Q3 are
nice to have and the ranking is already correct without them. One added instruction appears to push
"cite pdf links" out of a 7B model's attention, so the line cost more than it gave.

## Notes on the replay

**API tie order is non-deterministic.** For `search_documents("climate resilience Kenya", rows=10)`,
12 identical calls returned 2 different result sets: two near-identical Kenya procurement plans
(ids 33331272 and 33187992) swap in the 10th slot, 4 vs 8 times. A Q1 run that cited one of them
fails a replay that happens to draw the other. The checker now handles this itself: when a cited
document or link is missing, it replays each search with `rows` + 5 and reports a document found there
as "pass (tie at page boundary)". The two calls can swap the twins again, so it accepts the document
anywhere in the wider result, not only in the extra 5 rows (a first version that looked only at the
extra rows still failed 1 of 5 gradings). Anything not found there fails.

**Facet counts drift.** The API adds documents during the day (Procurement Plan: 144,387 at run time,
144,389 at replay). The checker accepts a count within 0.1% (min 5) of the replayed value. The
self-test plants an error of +5,000 to make sure real mistakes still fail.
