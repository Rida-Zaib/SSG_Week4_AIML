# Task 4.3 — LLM Observability with LangSmith

## Overview
Sets up LangSmith tracing for a small set of LLM calls (reasoning, extraction,
summarization, and one deliberate failure case) and evaluates response quality,
latency, and failure handling.

## How tracing works
LangSmith tracing turns on just by setting two environment variables —
`LANGCHAIN_TRACING_V2` and `LANGCHAIN_PROJECT` — no changes needed to the LLM
call code itself. Every call automatically gets logged to the LangSmith
project dashboard.

## Files
- `run_traces.py` — runs the 4 test cases and prints a latency/status summary
- `langsmith_evaluation_report.pdf` — the write-up: test case table, findings
- `Screenshot task4.3.png` — the LangSmith dashboard showing the
  `skillsetgo-week4-observability` project with the traced runs

## Requirements
Needs a free Groq API key and a free LangSmith API key, plus internet access.

```bash
pip install -r ../requirements.txt
# add GROQ_API_KEY and LANGCHAIN_API_KEY to your .env file
python3 run_traces.py
```

## Result
Verified working — the dashboard screenshot confirms 4 traces logged under
`skillsetgo-week4-observability`, with a 0% error rate and 0.59s median (P50)
latency.

## Deliverable
Evaluation report with trace screenshots, as required by the Skill Set Go
EduTech AI/ML track, Week 4, Task 4.3.
