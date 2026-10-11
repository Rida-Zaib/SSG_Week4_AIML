import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


def build_vectorstore(docs_path="documents"):
    loader = DirectoryLoader(docs_path, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"loaded {len(documents)} document(s), split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print("FAISS index saved to faiss_index/")
    return vectorstore


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


def build_rag_chain(vectorstore):
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("Set your GROQ_API_KEY environment variable before running this.")
        print("Get a free key (no credit card) at https://console.groq.com/keys")
        return

    if os.path.exists("faiss_index"):
        vectorstore = load_vectorstore()
        print("loaded existing FAISS index")
    else:
        vectorstore = build_vectorstore()

    qa_chain = build_rag_chain(vectorstore)

    questions = [
        "What is the AI/ML track?",
        "How many total tasks are in the internship?",
        "What does the capstone project need to include?"
    ]

    for question in questions:
        print("\nQuestion:", question)
        result = qa_chain.invoke({"query": question})
        print("Answer:", result["result"])
        print("Sources:", [doc.metadata.get("source") for doc in result["source_documents"]])


if __name__ == "__main__":
    main()