import os
import html
import time
import uuid
import traceback
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

ICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "bot_avatar.png")
FAVICON = ICON_PATH if os.path.exists(ICON_PATH) else "🧩"

st.set_page_config(page_title="Sage", page_icon=FAVICON, layout="centered")

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass

NODE_SVG = """<svg width="18" height="18" viewBox="0 0 256 256">
<line x1="88" y1="168" x2="168" y2="92" stroke="currentColor" stroke-width="16"/>
<line x1="88" y1="168" x2="168" y2="176" stroke="currentColor" stroke-width="16"/>
<circle cx="88" cy="168" r="26" fill="currentColor"/>
<circle cx="168" cy="92" r="20" fill="currentColor"/>
<circle cx="168" cy="176" r="20" fill="currentColor"/>
</svg>"""

USER_SVG = """<svg width="14" height="14" viewBox="0 0 24 24" fill="none">
<circle cx="12" cy="8" r="4" stroke="currentColor" stroke-width="2"/>
<path d="M4 20c0-4.4 3.6-7 8-7s8 2.6 8 7" stroke="currentColor" stroke-width="2" fill="none"/>
</svg>"""

# ---------------------------------------------------------------------------
# Brand: seagreen + white/near-black, adapting to light or dark mode.
# Button weight is deliberately NOT uniform: one primary action (New chat),
# quiet text-style rows for chat history, and muted low-key chips for
# suggestions - so nothing shouts all at once.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
  --bg: #FFFFFF; --bg-alt: #F6FAF7; --sea: #2E8B57; --sea-soft: #3A9C65;
  --ink: #16221C; --muted: #5E6B63; --border: #E0E6E2;
  --user-bg: #2E8B57; --user-ink: #FFFFFF;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --bg: #15181A; --bg-alt: #1D2122; --sea: #45B07E; --sea-soft: #3A9C65;
    --ink: #ECEFEC; --muted: #98A39C; --border: #262B29;
    --user-bg: #2A6B4B; --user-ink: #FFFFFF;
  }}
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: var(--ink); }}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: var(--bg) !important; }}
h1, h2, h3 {{ font-family: 'Fraunces', serif; }}

[data-testid="stSidebar"] {{ background: var(--bg-alt) !important; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] h3 {{ color: var(--ink); font-size: 0.95rem; font-family:'Fraunces',serif; margin-bottom: 2px; }}
[data-testid="stSidebar"] label {{ color: var(--ink); font-weight: 500; }}
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--bg); border-radius: 12px; border-color: var(--border) !important; }}

/* default button reset - specific groups below override this */
[data-testid="stSidebar"] .stButton button {{ border-radius: 8px; font-size: 0.85rem; }}

/* primary: New chat */
[data-testid="stSidebar"] button[kind="primary"] {{
  background: var(--sea) !important; border: none !important; color: #FFFFFF !important;
  border-radius: 10px !important; font-weight: 600;
}}
[data-testid="stSidebar"] button[kind="primary"]:hover {{ background: var(--sea-soft) !important; }}

/* chat history rows - quiet text rows, left aligned */
.st-key-chat-list-wrap .stButton button:not([kind="primary"]) {{
  background: transparent; border: none; color: var(--ink); font-weight: 500;
  padding: 0.4rem 0.7rem; justify-content: flex-start; text-align: left;
}}
.st-key-chat-list-wrap .stButton button:not([kind="primary"]) div,
.st-key-chat-list-wrap .stButton button:not([kind="primary"]) p {{ text-align: left; justify-content: flex-start; width: 100%; }}
.st-key-chat-list-wrap .stButton button:not([kind="primary"]):hover {{ background: color-mix(in srgb, var(--sea) 12%, transparent); color: var(--sea); }}
.st-key-chat-active .stButton button:not([kind="primary"]) {{
  background: color-mix(in srgb, var(--sea) 16%, transparent); color: var(--sea); font-weight: 600;
}}

/* suggestion chips - low-key until hovered */
.st-key-examples-wrap .stButton button {{
  background: transparent; border: 1px solid var(--border); color: var(--muted);
  border-radius: 999px; font-size: 0.8rem; text-align: left; padding: 0.35rem 0.8rem;
}}
.st-key-examples-wrap .stButton button:hover {{ border-color: var(--sea); color: var(--sea); }}

/* chat input: no separate dark strip, green focus ring */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
[data-testid="stChatInput"] {{ border: 1px solid var(--border) !important; border-radius: 14px !important; background: var(--bg-alt) !important; box-shadow: none !important; }}
[data-testid="stChatInput"]:focus-within {{ border-color: var(--sea) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--sea) 25%, transparent) !important; }}
[data-testid="stChatInput"] textarea {{ font-family: 'Inter', sans-serif; color: var(--ink) !important; }}

/* header: small mark + name, no big color block */
.brand-row {{ display: flex; align-items: center; gap: 12px; margin: 2px 0 2px; }}
.brand-mark {{
  width: 40px; height: 40px; border-radius: 50%; background: var(--bg-alt);
  border: 1px solid var(--border); color: var(--sea);
  display: flex; align-items: center; justify-content: center; flex: none;
}}
.brand-name {{ font-family: 'Fraunces', serif; font-size: 1.45rem; font-weight: 600; color: var(--ink); line-height: 1.15; }}
.brand-tag {{ color: var(--muted); font-size: 0.85rem; margin-top: 2px; }}
.brand-rule {{ border: none; border-top: 2px solid var(--sea); width: 54px; margin: 14px 0 18px; }}

/* chat: user = solid green bubble on the right; assistant = clean text with avatar */
.msg-row {{ display: flex; align-items: flex-end; gap: 8px; margin: 14px 0; }}
.msg-row.user {{ justify-content: flex-end; }}
.avatar {{ width: 26px; height: 26px; border-radius: 50%; flex: none; display: flex; align-items: center; justify-content: center; }}
.avatar.user {{ background: var(--bg-alt); border: 1px solid var(--border); color: var(--muted); }}
.avatar.assistant {{ background: var(--sea); color: #FFFFFF; }}
.bubble.user {{ max-width: 75%; padding: 10px 16px; border-radius: 18px 18px 4px 18px; background: var(--user-bg); color: var(--user-ink); font-size: 0.95rem; line-height: 1.55; word-break: break-word; }}
[class*="st-key-asst-"] {{ position: relative; padding-left: 40px; margin: 16px 0; }}
[class*="st-key-asst-"]::before {{
  content: ""; position: absolute; left: 0; top: 2px; width: 26px; height: 26px; border-radius: 50%;
  background: var(--sea) url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 256 256'><g stroke='white' stroke-width='16'><line x1='88' y1='168' x2='168' y2='92'/><line x1='88' y1='168' x2='168' y2='176'/></g><g fill='white'><circle cx='88' cy='168' r='26'/><circle cx='168' cy='92' r='20'/><circle cx='168' cy='176' r='20'/></g></svg>") center / 16px no-repeat;
}}
[class*="st-key-asst-"] p, [class*="st-key-asst-"] li {{ font-size: 0.95rem; line-height: 1.7; }}

.thinking-row {{ display: flex; align-items: center; gap: 8px; margin: 10px 0; color: var(--muted); font-size: 0.9rem; font-style: italic; }}

[data-testid="stExpander"] {{ border: 1px solid var(--border) !important; border-radius: 10px !important; background: var(--bg-alt) !important; }}
[data-testid="stExpander"] summary {{ color: var(--sea); font-size: 0.85rem; }}

input[type="checkbox"] {{ accent-color: var(--sea); }}

.block-container {{ padding-top: 2.5rem; padding-bottom: 7rem; }}
.stAppDeployButton, .stDeployButton, [data-testid="stAppDeployButton"] {{ display: none !important; }}
.st-key-examples-wrap .stButton button {{ color: var(--ink); opacity: 0.75; }}
.st-key-examples-wrap .stButton button:hover {{ opacity: 1; }}
.brand-rule {{ display: none; }}

/* force readable colors, whatever theme Streamlit itself picked */
[class*="st-key-asst-"], [class*="st-key-asst-"] * {{ color: var(--ink) !important; }}
[class*="st-key-asst-"] code {{ background: var(--bg-alt) !important; }}
[class*="st-key-asst-"] pre {{ background: var(--bg-alt) !important; }}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] [data-baseweb] {{ background: transparent !important; }}
[data-testid="stChatInput"] textarea {{ background: transparent !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; caret-color: var(--sea); }}
[data-testid="stChatInput"] textarea::placeholder {{ color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: 1; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: var(--bg) !important; color: var(--ink) !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: var(--ink) !important; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {{ color: var(--ink) !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="brand-row">
  <div class="brand-mark">{NODE_SVG}</div>
  <div>
    <div class="brand-name">Sage</div>
    <div class="brand-tag">AI agent for the Skill Set Go internship — calculations, conversions, live facts.</div>
  </div>
</div>
<hr class="brand-rule"/>
""", unsafe_allow_html=True)

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it in the app's secrets settings.")
    st.stop()

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agent import build_agent, get_vectorstore, SYSTEM_PROMPT, AVAILABLE_MODELS, DEFAULT_MODEL


@st.cache_resource(show_spinner="Loading knowledge base...")
def warmup():
    get_vectorstore()
    return True


@st.cache_resource
def get_agent(model_name: str, temperature: float):
    return build_agent(model_name, temperature)


warmup()


def new_chat_id():
    return uuid.uuid4().hex[:8]


if "chats" not in st.session_state:
    first_id = new_chat_id()
    st.session_state.chats = {first_id: {"title": "New chat", "messages": []}}
    st.session_state.active_chat_id = first_id
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

active_id = st.session_state.active_chat_id
active_chat = st.session_state.chats[active_id]

EXAMPLES = [
    "What does the capstone need to include?",
    "15% discount on 4500?",
    "Convert 5 marla to sq ft",
    "100 USD to PKR",
    "Solve x^2 - 5x + 6 = 0",
    "EMI on 1,000,000 at 15% for 3 years",
    "Mean and std of 4, 8, 15, 16, 23, 42",
    "BMI for 70 kg, 175 cm",
    "Weather in Lahore",
    "Time in New York now",
    "Days between 2026-01-01 and 2026-12-25",
    "255 to binary",
    "Latest news about AI",
]

with st.sidebar:
    with st.container(border=True):
        st.markdown("### Settings")
        model_name = st.selectbox("Model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))
        temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.1)
        show_steps = st.checkbox("Show agent steps", value=False,
                                  help="See which tool the agent used behind the scenes")

    st.write("")
    with st.container(key="chat-list-wrap", border=True):
        st.markdown("### Chats")
        if st.button("+ New chat", key="new_chat", type="primary", use_container_width=True):
            cid = new_chat_id()
            st.session_state.chats[cid] = {"title": "New chat", "messages": []}
            st.session_state.active_chat_id = cid
            st.rerun()

        for cid, chat in sorted(st.session_state.chats.items(), key=lambda kv: kv[0], reverse=True):
            with st.container(key=("chat-active" if cid == active_id else f"chat-row-{cid}")):
                col1, col2 = st.columns([0.82, 0.18])
                with col1:
                    if st.button(chat["title"], key=f"open_{cid}", use_container_width=True):
                        st.session_state.active_chat_id = cid
                        st.rerun()
                with col2:
                    if len(st.session_state.chats) > 1 and st.button("✕", key=f"del_{cid}"):
                        del st.session_state.chats[cid]
                        if st.session_state.active_chat_id == cid:
                            st.session_state.active_chat_id = next(iter(st.session_state.chats))
                        st.rerun()

    st.write("")
    with st.container(key="examples-wrap", border=True):
        st.markdown("### Try asking")
        for ex in EXAMPLES:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.pending_question = ex


_msg_i = [0]


def render_message(role, text):
    _msg_i[0] += 1
    if role == "user":
        safe = html.escape(text).replace("\n", "<br>")
        st.markdown(
            f'<div class="msg-row user"><div class="bubble user">{safe}</div>'
            f'<div class="avatar user">{USER_SVG}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        with st.container(key=f"asst-{_msg_i[0]}"):
            st.markdown(text)


for role, text in active_chat["messages"]:
    render_message(role, text)

typed_question = st.chat_input("Ask anything: internship, coding, science, calculations, latest news...")
question = typed_question or st.session_state.pending_question
st.session_state.pending_question = None


def run_agent(model, msgs, step_log):
    agent = get_agent(model, temperature)
    answer = ""
    for update in agent.stream({"messages": msgs}, config={"recursion_limit": 12}, stream_mode="updates"):
        for node, payload in update.items():
            for m in payload.get("messages", []):
                if node == "agent":
                    calls = getattr(m, "tool_calls", None)
                    if calls:
                        for c in calls:
                            step_log.append(f"🔧 `{c['name']}` {c['args']}")
                    else:
                        answer = m.content or ""
                elif node == "tools":
                    step_log.append(f"↳ `{m.name}`: {str(m.content)[:200]}")
    return answer


if question:
    active_chat["messages"].append(("user", question))
    if active_chat["title"] == "New chat":
        active_chat["title"] = question[:28] + ("…" if len(question) > 28 else "")
    render_message("user", question)

    placeholder = st.empty()
    placeholder.markdown(
        f'<div class="thinking-row"><div class="avatar assistant">{NODE_SVG}</div> Thinking…</div>',
        unsafe_allow_html=True,
    )

    start = time.time()
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, text in active_chat["messages"][-8:]:
        msgs.append(HumanMessage(content=text) if role == "user" else AIMessage(content=text))

    answer, step_log = "", []
    fallbacks = [model_name] + [m for m in AVAILABLE_MODELS if m != model_name]
    for candidate in fallbacks:
        try:
            answer = run_agent(candidate, msgs, step_log)
            if answer:
                break
        except Exception as e:
            # Kept out of the chat UI on purpose, but printed to the terminal
            # running `streamlit run app.py` so the real cause is debuggable.
            print(f"\n[Sage] model '{candidate}' failed: {e}")
            traceback.print_exc()

    placeholder.empty()

    if not answer:
        answer = "Sorry, I couldn't get an answer right now. Please try again in a moment."

    render_message("assistant", answer)

    if show_steps and step_log:
        with st.expander("Show agent steps"):
            for line in step_log:
                st.write(line)
            st.caption(f"Answered in {time.time() - start:.1f}s")

    active_chat["messages"].append(("assistant", answer))