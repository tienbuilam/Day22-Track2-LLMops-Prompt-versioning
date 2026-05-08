"""
Step 2 — Prompt Hub & A/B Routing
"""
import os
import sys
import hashlib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if "LANGSMITH_PROJECT" in os.environ and "LANGCHAIN_PROJECT" not in os.environ:
    os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
if not os.environ.get("LANGCHAIN_API_KEY"):
    print("Error: LANGCHAIN_API_KEY missing from .env")
    sys.exit(1)
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY missing from .env")
    sys.exit(1)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langsmith import Client, traceable

SYSTEM_V1 = (
    "You are a strict AI assistant. "
    "Answer the question using STRICTLY and ONLY the provided context. "
    "Do not include any prior knowledge, outside information, or extra explanations. "
    "Keep your answer extremely concise (1-2 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI tutor. Provide a structured, accurate answer based STRICTLY on the context provided.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify ONLY the key facts explicitly stated in the context.\n"
    "3. Write a clear, well-organized answer (2-4 sentences) without adding external knowledge.\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPT_V1_NAME = "my-rag-prompt-v1-tlam"
PROMPT_V2_NAME = "my-rag-prompt-v2-tlam"

def push_prompts_to_hub(client):
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise answers")
        print(f"✅ Pushed V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured answers")
        print(f"✅ Pushed V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2: {e}")

def pull_prompts_from_hub(client):
    prompts = {}
    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts

def get_prompt_version(request_id: str) -> str:
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME

def build_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    text = Path("data/knowledge_base.txt").read_text(encoding="utf-8")
    headers_to_split_on = [("##", "Section"), ("###", "Topic")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_docs = markdown_splitter.split_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(md_docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}

def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )

    import importlib.util
    spec = importlib.util.spec_from_file_location("step1", "01_langsmith_rag_pipeline.py")
    step1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(step1)
    SAMPLE_QUESTIONS = step1.SAMPLE_QUESTIONS
    
    v1_count = 0
    v2_count = 0
    log_lines = []

    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        line = f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}..."
        print(line)
        log_lines.append(line)

    summary = f"\nRouting Summary: {v1_count} queries to V1, {v2_count} queries to V2"
    print(summary)
    log_lines.append(summary)

    with open("02_ab_routing_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print("\n💾 Saved routing log to 02_ab_routing_log.txt")

if __name__ == "__main__":
    main()
