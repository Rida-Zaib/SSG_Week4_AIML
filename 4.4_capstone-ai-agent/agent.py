import os
import re
import ast
import math
import random
import string
import operator
import statistics
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence
from urllib.parse import quote

import requests
from dotenv import load_dotenv

load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

DEFAULT_MODEL = "openai/gpt-oss-120b"
AVAILABLE_MODELS = [
    "openai/gpt-oss-120b",        # strongest reasoning, Groq's recommended default
    "openai/gpt-oss-20b",         # lighter and faster, still very capable
]
HEADERS = {"User-Agent": "AIAgentSystem/1.0 (student project)"}


# =====================================================================
# 1. INTERNSHIP TOOLS
# =====================================================================

_vectorstore = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    docs_path = "documents" if os.path.exists("documents") else "knowledge_base"
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_path = "faiss_index"
    if os.path.exists(index_path):
        _vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
        chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(loader.load())
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        _vectorstore.save_local(index_path)
    return _vectorstore


@tool
def search_knowledge_base(query: str) -> str:
    """Search the Skill Set Go EduTech FAQ about the AI/ML internship: tasks, weeks,
    capstone, submission standards, learning cycle. ONLY for internship questions."""
    docs = get_vectorstore().similarity_search(query, k=4)
    if not docs:
        return "No relevant information found in the knowledge base."
    seen, out = set(), []
    for d in docs:
        text = d.page_content.strip()
        if text not in seen:
            seen.add(text)
            out.append(text)
    return "\n\n".join(out)


@tool
def count_remaining_tasks(completed: int) -> str:
    """How many internship tasks remain (17 total = 16 tasks + capstone) given how many
    a student completed. Returns remaining count and percentages."""
    total = 17
    remaining = total - completed
    return (f"{remaining} of {total} tasks remaining; "
            f"{completed / total * 100:.1f}% done, {remaining / total * 100:.1f}% left")


# =====================================================================
# 2. CALCULATOR TOOLS
# =====================================================================

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "factorial": math.factorial, "ceil": math.ceil, "floor": math.floor,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


def _fmt(x):
    if isinstance(x, float):
        return f"{x:,.6g}" if abs(x) >= 1e6 or abs(x) < 1e-4 else f"{x:,.4f}".rstrip("0").rstrip(".")
    return str(x)


@tool
def calculator(expression: str) -> str:
    """Exact math. Supports + - * / ** % //, parentheses, sqrt, log, log10, log2, sin, cos,
    tan, abs, round, factorial, ceil, floor, pi, e. Trig uses radians.
    Examples: '5/17*100', '5e9*365.25*24*3600', 'sqrt(144)'."""
    try:
        return _fmt(_safe_eval(ast.parse(expression.strip(), mode="eval")))
    except Exception as e:
        return f"error: {e}"


@tool
def percentage_calculator(kind: str, a: float, b: float = 0) -> str:
    """Percentage math. kind must be one of:
    'percent_of'      -> a% of b            (e.g. 15% of 200)
    'what_percent'    -> a is what % of b   (e.g. 30 is what % of 120)
    'percent_change'  -> change from a to b (increase/decrease %)
    'add_percent'     -> b increased by a%  (tax/markup)
    'subtract_percent'-> b decreased by a%  (discount)"""
    try:
        if kind == "percent_of":
            return f"{a}% of {b} = {_fmt(a / 100 * b)}"
        if kind == "what_percent":
            return f"{a} is {_fmt(a / b * 100)}% of {b}"
        if kind == "percent_change":
            ch = (b - a) / a * 100
            return f"Change from {a} to {b}: {_fmt(abs(ch))}% {'increase' if ch >= 0 else 'decrease'}"
        if kind == "add_percent":
            return f"{b} + {a}% = {_fmt(b * (1 + a / 100))}"
        if kind == "subtract_percent":
            return f"{b} - {a}% = {_fmt(b * (1 - a / 100))}"
        return "error: unknown kind"
    except ZeroDivisionError:
        return "error: division by zero"


_UNITS = {
    "length": {"m": 1, "km": 1000, "cm": 0.01, "mm": 0.001, "mile": 1609.344, "yard": 0.9144,
               "foot": 0.3048, "inch": 0.0254},
    "weight": {"kg": 1, "g": 0.001, "mg": 1e-6, "lb": 0.45359237, "oz": 0.028349523, "ton": 1000,
               "tola": 0.01166638},
    "volume": {"l": 1, "ml": 0.001, "gallon": 3.785411784, "cup": 0.2365882365,
               "tbsp": 0.01478676, "tsp": 0.00492892},
    "area": {"sqm": 1, "sqkm": 1e6, "sqft": 0.09290304, "acre": 4046.8564224, "hectare": 10000,
             "marla": 25.2929538, "kanal": 505.857},
    "speed": {"mps": 1, "kmh": 0.2777777778, "mph": 0.44704, "knot": 0.5144444},
    "time": {"sec": 1, "min": 60, "hour": 3600, "day": 86400, "week": 604800, "year": 31557600},
    "data": {"byte": 1, "kb": 1024, "mb": 1024 ** 2, "gb": 1024 ** 3, "tb": 1024 ** 4},
}
_ALIAS = {
    "meter": "m", "metre": "m", "meters": "m", "kilometer": "km", "kilometre": "km",
    "centimeter": "cm", "millimeter": "mm", "miles": "mile", "mi": "mile", "feet": "foot",
    "ft": "foot", "inches": "inch", "in": "inch", "yd": "yard", "kilogram": "kg", "gram": "g",
    "pound": "lb", "lbs": "lb", "ounce": "oz", "liter": "l", "litre": "l", "gal": "gallon",
    "sq m": "sqm", "sq ft": "sqft", "square feet": "sqft", "square meter": "sqm",
    "km/h": "kmh", "kph": "kmh", "m/s": "mps", "second": "sec", "seconds": "sec", "s": "sec",
    "minute": "min", "minutes": "min", "hr": "hour", "hours": "hour", "days": "day",
    "weeks": "week", "years": "year", "yr": "year", "bytes": "byte", "kilobyte": "kb",
    "megabyte": "mb", "gigabyte": "gb", "terabyte": "tb",
    "celsius": "c", "fahrenheit": "f", "kelvin": "k", "°c": "c", "°f": "f",
}


def _norm_unit(u: str) -> str:
    u = u.lower().strip()
    u = _ALIAS.get(u, u)
    return u


def _find_unit(u: str):
    for cat, table in _UNITS.items():
        if u in table:
            return cat
        if u.endswith("s") and u[:-1] in table:
            return cat
    return None


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert units. Length (m, km, cm, mm, mile, yard, foot, inch), weight (kg, g, mg, lb, oz,
    ton, tola), volume (l, ml, gallon, cup, tbsp, tsp), area (sqm, sqkm, sqft, acre, hectare,
    marla, kanal), speed (mps, kmh, mph, knot), time (sec, min, hour, day, week, year),
    data (byte, kb, mb, gb, tb), temperature (c, f, k)."""
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    if f in ("c", "f", "k") and t in ("c", "f", "k"):
        c = value if f == "c" else (value - 32) * 5 / 9 if f == "f" else value - 273.15
        res = c if t == "c" else c * 9 / 5 + 32 if t == "f" else c + 273.15
        return f"{value} {f.upper()} = {_fmt(res)} {t.upper()}"
    cf, ct = _find_unit(f), _find_unit(t)
    if not cf or cf != ct:
        return f"error: cannot convert '{from_unit}' to '{to_unit}'"
    tf = _UNITS[cf]
    f = f if f in tf else f[:-1]
    t = t if t in tf else t[:-1]
    return f"{value} {f} = {_fmt(value * tf[f] / tf[t])} {t}"


@tool
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert money using live exchange rates. Use 3-letter codes: USD, PKR, EUR, GBP, SAR, AED, INR..."""
    try:
        f, t = from_currency.upper().strip(), to_currency.upper().strip()
        r = requests.get(f"https://open.er-api.com/v6/latest/{f}", timeout=10, headers=HEADERS).json()
        if r.get("result") != "success" or t not in r["rates"]:
            return "error: currency not found"
        rate = r["rates"][t]
        return f"{_fmt(amount)} {f} = {_fmt(amount * rate)} {t} (1 {f} = {_fmt(rate)} {t}, live rate)"
    except Exception as e:
        return f"error: could not fetch rates ({e})"


@tool
def stats_calculator(numbers: str) -> str:
    """Statistics for a list of numbers (comma or space separated): count, sum, mean, median,
    mode, min, max, range, standard deviation, variance."""
    try:
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE]-?\d+)?", numbers)]
        if not nums:
            return "error: no numbers found"
        out = {
            "count": len(nums), "sum": sum(nums), "mean": statistics.mean(nums),
            "median": statistics.median(nums), "min": min(nums), "max": max(nums),
            "range": max(nums) - min(nums),
        }
        try:
            out["mode"] = statistics.mode(nums)
        except statistics.StatisticsError:
            out["mode"] = "no unique mode"
        if len(nums) > 1:
            out["std_dev (sample)"] = statistics.stdev(nums)
            out["variance (sample)"] = statistics.variance(nums)
            out["std_dev (population)"] = statistics.pstdev(nums)
        return "; ".join(f"{k}: {_fmt(v) if not isinstance(v, str) else v}" for k, v in out.items())
    except Exception as e:
        return f"error: {e}"


@tool
def symbolic_math(task: str, expression: str, variable: str = "x") -> str:
    """Algebra and calculus. task must be one of: 'solve' (equation, e.g. 'x^2 - 5x + 6 = 0'),
    'simplify', 'expand', 'factor', 'differentiate', 'integrate'. Use ^ for powers."""
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                                implicit_multiplication_application, convert_xor)
        if "__" in expression or not re.fullmatch(r"[\w\s\+\-\*/\^\(\)\.=,]+", expression):
            return "error: unsupported characters in expression"
        tr = standard_transformations + (implicit_multiplication_application, convert_xor)
        var = sp.Symbol(variable)
        if task == "solve":
            if "=" in expression:
                lhs, rhs = expression.split("=", 1)
                expr = parse_expr(lhs, transformations=tr) - parse_expr(rhs, transformations=tr)
            else:
                expr = parse_expr(expression, transformations=tr)
            return f"Solutions for {variable}: {sp.solve(expr, var)}"
        expr = parse_expr(expression, transformations=tr)
        if task == "simplify":
            return str(sp.simplify(expr))
        if task == "expand":
            return str(sp.expand(expr))
        if task == "factor":
            return str(sp.factor(expr))
        if task == "differentiate":
            return f"d/d{variable}: {sp.diff(expr, var)}"
        if task == "integrate":
            return f"Integral: {sp.integrate(expr, var)} + C"
        return "error: unknown task"
    except Exception as e:
        return f"error: {e}"


@tool
def finance_calculator(kind: str, principal: float, annual_rate_percent: float, years: float,
                       times_per_year: int = 12) -> str:
    """Finance math. kind: 'emi' (monthly loan payment), 'simple_interest', or 'compound_interest'
    (times_per_year = compounding frequency, default 12). Rate is annual percent, e.g. 12 for 12%."""
    try:
        r = annual_rate_percent / 100
        if kind == "emi":
            n, mr = int(years * 12), r / 12
            emi = principal / n if mr == 0 else principal * mr * (1 + mr) ** n / ((1 + mr) ** n - 1)
            return (f"Monthly EMI: {_fmt(emi)}; total payment: {_fmt(emi * n)}; "
                    f"total interest: {_fmt(emi * n - principal)}")
        if kind == "simple_interest":
            i = principal * r * years
            return f"Interest: {_fmt(i)}; total: {_fmt(principal + i)}"
        if kind == "compound_interest":
            total = principal * (1 + r / times_per_year) ** (times_per_year * years)
            return f"Final amount: {_fmt(total)}; interest earned: {_fmt(total - principal)}"
        return "error: unknown kind"
    except Exception as e:
        return f"error: {e}"


@tool
def health_calculator(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI from weight in kg and height in cm, with the standard category."""
    try:
        bmi = weight_kg / ((height_cm / 100) ** 2)
        cat = ("Underweight" if bmi < 18.5 else "Normal" if bmi < 25
               else "Overweight" if bmi < 30 else "Obese")
        return f"BMI: {bmi:.1f} ({cat}). General guide only, not medical advice."
    except Exception as e:
        return f"error: {e}"


@tool
def date_calculator(kind: str, date1: str, date2: str = "", days: int = 0) -> str:
    """Date math. Dates in YYYY-MM-DD. kind: 'days_between' (date1 to date2), 'add_days'
    (date1 + days, negative to subtract), 'weekday' (day of week of date1), 'age' (years/months/days
    from date1 to today)."""
    try:
        d1 = datetime.strptime(date1, "%Y-%m-%d")
        if kind == "days_between":
            d2 = datetime.strptime(date2, "%Y-%m-%d")
            return f"{abs((d2 - d1).days)} days between {date1} and {date2}"
        if kind == "add_days":
            return (d1 + timedelta(days=days)).strftime("%A, %d %B %Y")
        if kind == "weekday":
            return d1.strftime("%A")
        if kind == "age":
            t = datetime.now()
            y, m, d = t.year - d1.year, t.month - d1.month, t.day - d1.day
            if d < 0:
                m -= 1
                d += 30
            if m < 0:
                y -= 1
                m += 12
            return f"{y} years, {m} months, {d} days"
        return "error: unknown kind"
    except Exception as e:
        return f"error: {e} (use YYYY-MM-DD)"


@tool
def number_base_convert(value: str, from_base: int, to_base: int) -> str:
    """Convert a number between bases 2 to 36 (binary=2, octal=8, decimal=10, hex=16)."""
    try:
        n = int(value.strip().lower().replace("0x", "").replace("0b", ""), from_base)
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        if n == 0:
            return "0"
        s, x = "", abs(n)
        while x:
            s = digits[x % to_base] + s
            x //= to_base
        return f"{value} (base {from_base}) = {'-' if n < 0 else ''}{s.upper()} (base {to_base})"
    except Exception as e:
        return f"error: {e}"


# =====================================================================
# 3. GENERAL TOOLS
# =====================================================================

@tool
def get_current_datetime(timezone: str = "Asia/Karachi") -> str:
    """Current date and time in any IANA timezone, e.g. 'Asia/Karachi', 'America/New_York',
    'Europe/London', 'Asia/Dubai', 'Asia/Tokyo'. Default is Pakistan time."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(timezone)).strftime(f"%A, %d %B %Y, %I:%M %p ({timezone})")
    except Exception:
        return datetime.now().strftime("%A, %d %B %Y, %I:%M %p (local)")


@tool
def weather(city: str) -> str:
    """Current weather for any city (temperature, feels like, humidity, wind, condition)."""
    try:
        r = requests.get(f"https://wttr.in/{quote(city)}?format=j1", timeout=10, headers=HEADERS).json()
        c = r["current_condition"][0]
        return (f"{city}: {c['weatherDesc'][0]['value']}, {c['temp_C']}°C (feels like {c['FeelsLikeC']}°C), "
                f"humidity {c['humidity']}%, wind {c['windspeedKmph']} km/h")
    except Exception as e:
        return f"error: could not fetch weather ({e})"


@tool
def wikipedia_summary(topic: str) -> str:
    """Short factual summary of a person, place, concept, or event from Wikipedia."""
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(topic.replace(' ', '_'))}",
            timeout=10, headers=HEADERS)
        if r.status_code != 200:
            return "No Wikipedia page found. Try web_search instead."
        j = r.json()
        return f"{j.get('title')}: {j.get('extract', '')[:900]} (Source: Wikipedia)"
    except Exception as e:
        return f"error: {e}"


@tool
def web_search(query: str) -> str:
    """Search the web for current or recent info: news, prices, latest versions, recent
    events, or facts you are unsure about. NOT for internship questions."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
    except ImportError:
        return "Web search unavailable (run: pip install ddgs)."
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return "No web results found."
        return "\n\n".join(
            f"{r.get('title', '')}: {r.get('body', '')[:300]} (Source: {r.get('href', '')})"
            for r in results
        )
    except Exception as e:
        return f"Web search failed: {e}"


@tool
def text_tools(action: str, text: str = "", length: int = 12) -> str:
    """Text utilities. action: 'word_count', 'char_count', 'reverse', 'upper', 'lower', 'title',
    or 'password' (generates a strong random password of given length; text not needed)."""
    if action == "word_count":
        return f"{len(text.split())} words"
    if action == "char_count":
        return f"{len(text)} characters ({len(text.replace(' ', ''))} without spaces)"
    if action == "reverse":
        return text[::-1]
    if action == "upper":
        return text.upper()
    if action == "lower":
        return text.lower()
    if action == "title":
        return text.title()
    if action == "password":
        length = max(8, min(length, 64))
        pool = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(random.SystemRandom().choice(pool) for _ in range(length))
    return "error: unknown action"


tools = [
    search_knowledge_base, count_remaining_tasks,
    calculator, percentage_calculator, unit_converter, currency_convert, stats_calculator,
    symbolic_math, finance_calculator, health_calculator, date_calculator, number_base_convert,
    get_current_datetime, weather, wikipedia_summary, web_search, text_tools,
]
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def build_agent(model_name: str = DEFAULT_MODEL, temperature: float = 0.2):
    llm = ChatGroq(model=model_name, temperature=temperature, max_tokens=900,
                   timeout=40, max_retries=2)
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState) -> AgentState:
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def should_continue(state: AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


SYSTEM_PROMPT = """You are a sharp, fast, expert AI assistant. You can discuss any topic (science, tech, \
coding, careers, study, health basics, money, general knowledge) and you are the official helper for the \
Skill Set Go EduTech AI/ML internship.

TOOLS (call only when needed; several at once if independent)
- Internship, tasks, weeks, capstone, submission, learning cycle -> search_knowledge_base; answer only from \
its results; if missing, say you don't know. "I finished N tasks" -> count_remaining_tasks.
- Plain arithmetic -> calculator. Percent, discount, tax, % change -> percentage_calculator.
- Unit conversion (length, weight, area incl. marla/kanal, temperature, data, speed, time) -> unit_converter.
- Money conversion -> currency_convert (live rates). Loans/EMI/interest -> finance_calculator.
- List of numbers stats -> stats_calculator. Equations, derivatives, integrals, factoring -> symbolic_math.
- BMI -> health_calculator. Date differences, age, weekday, date + N days -> date_calculator.
- Binary/hex/octal -> number_base_convert. Time in any city/timezone -> get_current_datetime.
- Weather -> weather. Quick facts about a person/place/concept -> wikipedia_summary.
- News, latest versions, prices, recent or uncertain facts -> web_search; name the source briefly.
- Word count, text case, reverse, password generation -> text_tools.
- Never compute numbers in your head; use a tool. Everything else: answer directly, no tools.

ANSWER STYLE
- First sentence = the direct answer. Then only the details that matter. No filler, no repeating the question.
- Precise and honest: if unsure or possibly outdated, say so. Never guess numbers or facts.
- Use conversation history for follow-ups ("give me the actual number", "explain more").
- Reply in the user's language; Roman Urdu/Hinglish in, same style out.
- Short paragraphs or bullets; a table only to compare several items. No LaTeX (write 1.6 x 10^17).
- For coding: minimal working code in a code block, then a one-line explanation.
- Health/finance answers are general information only; add one short line to consult a professional \
when the stakes are real.
- If the question is truly ambiguous, ask one short clarifying question instead of guessing."""


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set your GROQ_API_KEY environment variable before running this.")
        return
    app = build_agent()
    for q in [
        "What is the capstone project supposed to include?",
        "If I've completed 12 tasks, how many are left, and what percentage is that of the total?",
        "Convert 5 marla to square feet and 100 USD to PKR",
        "Solve x^2 - 5x + 6 = 0",
    ]:
        print("\n" + "=" * 60)
        print("User:", q)
        result = app.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=q)]})
        print("Agent:", result["messages"][-1].content)


if __name__ == "__main__":
    main()