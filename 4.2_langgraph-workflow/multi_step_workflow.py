import os
from dotenv import load_dotenv

load_dotenv()
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq


class WorkflowState(TypedDict):
    question: str
    question_type: Optional[str]
    draft_answer: Optional[str]
    final_answer: Optional[str]
    needs_review: Optional[bool]


llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


def classify_question(state: WorkflowState) -> WorkflowState:
    prompt = (
        "Classify this question as either 'factual' or 'opinion'. "
        f"Reply with just one word.\n\nQuestion: {state['question']}"
    )
    response = llm.invoke(prompt)
    q_type = response.content.strip().lower()
    print(f"[classify] question type: {q_type}")
    return {"question_type": q_type}


def draft_answer(state: WorkflowState) -> WorkflowState:
    prompt = f"Answer this question concisely: {state['question']}"
    response = llm.invoke(prompt)
    print("[draft] draft answer generated")
    return {"draft_answer": response.content}


def review_answer(state: WorkflowState) -> WorkflowState:
    prompt = (
        "Does this answer look complete and correct for the question below? "
        "Reply with just 'yes' or 'no'.\n\n"
        f"Question: {state['question']}\nAnswer: {state['draft_answer']}"
    )
    response = llm.invoke(prompt)
    needs_review = "no" in response.content.strip().lower()
    print(f"[review] needs revision: {needs_review}")
    return {"needs_review": needs_review}


def revise_answer(state: WorkflowState) -> WorkflowState:
    prompt = (
        f"Improve and expand this answer to fully address the question.\n\n"
        f"Question: {state['question']}\nCurrent answer: {state['draft_answer']}"
    )
    response = llm.invoke(prompt)
    print("[revise] answer revised")
    return {"final_answer": response.content}


def finalize_answer(state: WorkflowState) -> WorkflowState:
    print("[finalize] using draft as final answer")
    return {"final_answer": state["draft_answer"]}


def route_after_review(state: WorkflowState) -> str:
    return "revise_answer" if state["needs_review"] else "finalize_answer"


def build_workflow():
    graph = StateGraph(WorkflowState)

    graph.add_node("classify_question", classify_question)
    graph.add_node("draft_node", draft_answer)
    graph.add_node("review_answer", review_answer)
    graph.add_node("revise_answer", revise_answer)
    graph.add_node("finalize_answer", finalize_answer)

    graph.set_entry_point("classify_question")
    graph.add_edge("classify_question", "draft_node")
    graph.add_edge("draft_node", "review_answer")
    graph.add_conditional_edges(
        "review_answer",
        route_after_review,
        {
            "revise_answer": "revise_answer",
            "finalize_answer": "finalize_answer"
        }
    )
    graph.add_edge("revise_answer", END)
    graph.add_edge("finalize_answer", END)

    return graph.compile()


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set your GROQ_API_KEY environment variable before running this.")
        print("Get a free key (no credit card) at https://console.groq.com/keys")
        return

    app = build_workflow()

    questions = [
        "What is the boiling point of water at sea level?",
        "What's the best programming language?"
    ]

    for q in questions:
        print("\n" + "=" * 50)
        print("Question:", q)
        result = app.invoke({"question": q})
        print("\nFinal answer:", result["final_answer"])


if __name__ == "__main__":
    main()