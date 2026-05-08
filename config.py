import os
from dotenv import load_dotenv

load_dotenv()

def check_config():
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {os.environ.get('LANGCHAIN_PROJECT', 'day22-langsmith-lab')}")
    print(f"   OpenAI endpoint   : {os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
    print(f"   Default LLM model : {os.environ.get('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   Embedding model   : {os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')}")
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY is not set in .env")
    if not os.environ.get("LANGCHAIN_API_KEY"):
        print("⚠️  Warning: LANGCHAIN_API_KEY is not set in .env")

if __name__ == "__main__":
    check_config()
