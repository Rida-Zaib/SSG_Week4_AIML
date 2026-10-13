# Task 4.2 — Multi-Step Workflow in LangGraph

## Overview
A LangGraph workflow that classifies a question, drafts an answer, reviews the
draft, and then either revises it or finalizes it — a genuine multi-step
decision flow, not just a straight-line pipeline.

## Workflow
```
classify_question -> draft_node -> review_answer -+-> revise_answer -> END
                                                     +-> finalize_answer -> END
```
The branch after `review_answer` is a **conditional edge** — the graph picks
`revise_answer` or `finalize_answer` based on the LLM's own judgment of whether
the draft needs work.

## Files
- `multi_step_workflow.py` — the full workflow, runnable end to end
- `workflow_diagram.md` — the state diagram (Mermaid) and node descriptions

## Requirements
Needs a free Groq API key and internet access.

```bash
pip install -r ../requirements.txt
# add GROQ_API_KEY to your .env file
python3 multi_step_workflow.py
```

## Sample result
Verified working — correctly classifies a factual question ("What is the
boiling point of water at sea level?") vs. an opinion question ("What's the
best programming language?"), and produces a sensible final answer for both.

## Deliverable
LangGraph workflow diagram + code, as required by the Skill Set Go EduTech
AI/ML track, Week 4, Task 4.2.
