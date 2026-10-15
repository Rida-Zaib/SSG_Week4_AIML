# Sage: AI Agent System

Capstone project for the **Skill Set Go EduTech AI/ML Track (Week 4)**.

Sage is a general-purpose AI agent built on **LangGraph**. It answers internship questions from a local knowledge base (RAG), calls 15 other tools for math, conversions and live information, and decides on its own which tool (or tools) to use before answering. It runs as a command-line script (`agent.py`) and as a Streamlit chat app (`app.py`).

Everything it uses is free: Groq's free API, local embeddings, and free public data sources.

## Features

- **17 tools in 3 groups**: internship, math and money, live and general
- **RAG** over the internship FAQ using FAISS and `all-MiniLM-L6-v2` embeddings (runs locally)
- **Several tools per turn**: e.g. a unit conversion and a currency conversion in one answer
- **Two-model fallback**: if `openai/gpt-oss-120b` fails, the app retries with `openai/gpt-oss-20b`
- **Streamlit chat UI** with chat history, New chat, a model picker, a creativity slider, an optional "Show agent steps" view, 13 example prompts, and light/dark mode

## How it works

```
User -> Agent node (LLM + 17 tools) -> Tools node -> back to Agent node -> Final answer
```

The agent node decides whether it can answer or needs a tool. The tools node runs the chosen tool(s) and returns the results. The loop repeats until the agent has enough to answer.

## The 17 tools

| Group | Tools |
|---|---|
| Internship (2) | `search_knowledge_base`, `count_remaining_tasks` |
| Math and money (10) | `calculator`, `percentage_calculator`, `unit_converter`, `currency_convert`, `stats_calculator`, `symbolic_math`, `finance_calculator`, `health_calculator`, `date_calculator`, `number_base_convert` |
| Live and general (5) | `get_current_datetime`, `weather`, `wikipedia_summary`, `web_search`, `text_tools` |

## Project structure

```
4.4_capstone-ai-agent/
├── agent.py            # LangGraph agent, tools, system prompt (also runs in the terminal)
├── app.py              # Streamlit chat interface
├── requirements.txt
├── knowledge_base/     # .txt files for the internship FAQ (RAG source)
├── assets/             # favicon / avatar (optional)
├── .streamlit/
│   └── config.toml     # theme color
└── README.md
```

## Run it locally

Python 3.12 is recommended (3.13 had compatibility problems with some compiled packages).

```powershell
cd 4.4_capstone-ai-agent
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Get a free API key at <https://console.groq.com/keys> and put it in a `.env` file:

```
GROQ_API_KEY=your-key-here
```

Then start the app:

```powershell
streamlit run app.py
```

Or use the command-line version:

```powershell
python agent.py
```

The first run downloads the embedding model (about 90 MB) and builds the `faiss_index` folder.

## Deploy on Streamlit Community Cloud (free)

1. Push this folder to a GitHub repo. Keep `.env`, `venv/` and `faiss_index/` out of it with a `.gitignore`.
2. On <https://share.streamlit.io>, create an app from the repo with `app.py` as the main file.
3. In **Advanced settings**, choose Python 3.12 and add your key under **Secrets**:

   ```toml
   GROQ_API_KEY = "your-key-here"
   ```

Never commit your real key. If it leaks, delete it in the Groq console and create a new one.

## Troubleshooting

| Problem | Fix |
|---|---|
| App says "Sorry, I couldn't get an answer" | The real error is printed in the terminal where `streamlit run app.py` is running. |
| `403 Access denied. Please check your network settings` | Groq is blocking your network. Turn off any VPN or proxy, or try another network or a mobile hotspot. |
| `numpy` version conflict while installing | `langchain==0.3.7` needs `numpy<2`, so pin `numpy==1.26.4` in `requirements.txt` on Python 3.12. |
| `torch` DLL error on Windows (WinError 126) | Reinstall the CPU build: `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cpu` |
| "GROQ_API_KEY is not set" | Add it to `.env` locally, or to Secrets on Streamlit Cloud (with quotes around the value). |

## Example prompts

- What does the capstone need to include?
- 15% discount on 4500?
- Convert 5 marla to sq ft and 100 USD to PKR
- Solve x^2 - 5x + 6 = 0
- EMI on 1,000,000 at 15% for 3 years
- Weather in Lahore
- Latest news about AI

## Tech stack

LangGraph, LangChain, Groq, FAISS, sentence-transformers, Streamlit, SymPy, plus free public APIs (wttr.in, Wikipedia, DDGS web search).
