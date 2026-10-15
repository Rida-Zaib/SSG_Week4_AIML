# Skill Set Go EduTech: AI/ML Track, Week 4

This repository holds all four Week 4 tasks of the Skill Set Go EduTech AI/ML track. The tasks build on each other, from a basic RAG app to a full AI agent with a chat interface.

| Task | Folder | What it is |
|---|---|---|
| 4.1 | [`4.1_rag-application`](./4.1_rag-application) | A retrieval-augmented generation (RAG) application |
| 4.2 | [`4.2_langgraph-workflow`](./4.2_langgraph-workflow) | A multi-step LangGraph workflow with conditional branching |
| 4.3 | [`4.3_langsmith-observability`](./4.3_langsmith-observability) | Tracing and observability with LangSmith |
| 4.4 | [`4.4_capstone-ai-agent`](./4.4_capstone-ai-agent) | **Capstone:** Sage, a LangGraph AI agent with 17 tools and a Streamlit chat app |

Each folder has its own `README.md` with details and run instructions for that task.

## How the tasks connect

```
4.1 RAG  ->  4.2 LangGraph workflow  ->  4.3 LangSmith tracing  ->  4.4 Capstone agent
retrieve     branch and loop in a        see what each step         combines all of it:
and answer   graph (classify, draft,     did, and debug it          RAG + tools + fallback
             review, revise)                                        + chat interface
```

- **4.1** introduces retrieval: documents are embedded and searched so answers are grounded in real content.
- **4.2** adds structure: a workflow graph where the next step depends on the result of the previous one (classify, draft, review, then revise or finalize).
- **4.3** adds visibility: LangSmith traces show every step of a run, so problems can be found and fixed.
- **4.4** brings it together in one agent that decides which tool to use, answers from a knowledge base, retries on a second model if one fails, and runs as a Streamlit chat app.

## Capstone highlights (Task 4.4)

- 17 tools across three groups: internship (RAG), math and money, live and general
- RAG over the internship FAQ using FAISS and `all-MiniLM-L6-v2` embeddings, run locally
- Two-model fallback on Groq (`openai/gpt-oss-120b`, then `openai/gpt-oss-20b`)
- Streamlit chat UI with chat history, optional "Show agent steps", and light/dark mode
- Completely free to run: Groq free API, local embeddings, free public data sources

Live demo: _add your deployed Streamlit link here_

## Repository structure

```
SSG_Week4_AIML/
├── 4.1_rag-application/
├── 4.2_langgraph-workflow/
├── 4.3_langsmith-observability/
├── 4.4_capstone-ai-agent/
├── env.example          # copy to .env and fill in your keys
├── requirements.txt     # shared Python dependencies
└── README.md            # this file
```

## Setup

Python 3.12 is recommended. Python 3.13 had compatibility problems with some compiled packages (numpy, torch, ormsgpack).

```powershell
cd SSG_Week4_AIML
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Create your `.env` file in the repository root from the example, then add your own keys:

```powershell
copy env.example .env
```

At minimum you need a free Groq key from <https://console.groq.com/keys>:

```
GROQ_API_KEY=your-key-here
```

Tasks that use LangSmith also need the LangSmith keys listed in `env.example`.

**Never commit your `.env` file.** It is listed in `.gitignore`. If a key is ever pushed by mistake, delete it in the provider's console and create a new one.

## Running each task

Run the task you want from its own folder, and follow the instructions in that folder's `README.md`.

The capstone app:

```powershell
cd 4.4_capstone-ai-agent
streamlit run app.py
```

Or the command-line version:

```powershell
python agent.py
```

## Tech stack

Python, LangChain, LangGraph, LangSmith, Groq, FAISS, sentence-transformers, Streamlit, SymPy.

## Troubleshooting

| Problem | Fix |
|---|---|
| `numpy` conflict during install | On Python 3.12, pin `numpy==1.26.4` (`langchain==0.3.7` needs `numpy<2`). |
| `torch` DLL error on Windows (WinError 126) | Reinstall the CPU build: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu` |
| `403 Access denied` from Groq | Your network or VPN is blocked. Turn off the VPN, or try another network or a mobile hotspot. |
| Push rejected: file larger than 100 MB | A virtual environment was committed. Add `venv/` and `venv312/` to `.gitignore` and remove them from Git. |
| "GROQ_API_KEY is not set" | Add it to `.env` locally, or to Secrets on Streamlit Cloud (value in quotes). |

## Author

Rida ([@Rida-Zaib](https://github.com/Rida-Zaib))
