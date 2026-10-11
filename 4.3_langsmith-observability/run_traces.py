import os
from dotenv import load_dotenv

load_dotenv()
import time
from langchain_groq import ChatGroq

# LangSmith tracing is enabled just by setting these environment variables -
# no code changes needed to the LLM calls themselves once they're set.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "skillsetgo-week4-observability"
# also required: LANGCHAIN_API_KEY (free from smith.langchain.com)
# and GROQ_API_KEY (free, no credit card, from console.groq.com/keys)

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


test_cases = [
    {"id": "reasoning_1", "prompt": "If a train travels 60 miles in 1.5 hours, what is its average speed?"},
    {"id": "extraction_1", "prompt": "Extract the name and age from: 'John is 34 years old.' Reply as JSON."},
    {"id": "summarization_1", "prompt": "Summarize in one sentence: The internet is a global network of interconnected computers that allows people to share information instantly across the world."},
    {"id": "edge_case_1", "prompt": ""},  # deliberately empty, to test a failure case
]


def run_evaluation():
    results = []
    for case in test_cases:
        start = time.time()
        try:
            response = llm.invoke(case["prompt"])
            latency = time.time() - start
            results.append({
                "id": case["id"],
                "prompt": case["prompt"],
                "output": response.content,
                "latency_sec": round(latency, 2),
                "status": "success"
            })
            print(f"[{case['id']}] OK, {latency:.2f}s")
        except Exception as e:
            latency = time.time() - start
            results.append({
                "id": case["id"],
                "prompt": case["prompt"],
                "output": None,
                "latency_sec": round(latency, 2),
                "status": f"failed: {e}"
            })
            print(f"[{case['id']}] FAILED: {e}")
    return results


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set your GROQ_API_KEY environment variable before running this.")
        print("Get a free key (no credit card) at https://console.groq.com/keys")
        return
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("Set your LANGCHAIN_API_KEY (free from smith.langchain.com) to enable tracing.")
        return

    results = run_evaluation()

    print("\nAll traces are now visible at https://smith.langchain.com")
    print(f"under the '{os.environ['LANGCHAIN_PROJECT']}' project.")
    print("\nSummary:")
    for r in results:
        print(r["id"], "-", r["status"], "-", r["latency_sec"], "sec")


if __name__ == "__main__":
    main()